module NotPluto

import Pluto

"""
    NotPluto.run(; kwargs...)

Start a Pluto server in non-reactive mode (Neptune-equivalent behaviour):

  1. `run_notebook_on_load = false` — opening a notebook does not auto-run cells.
  2. Running a cell does **not** propagate to its downstream dependents.
  3. Multiple cells may define the same global variable without an error.

All other keyword arguments are forwarded to `Pluto.run`.

Loading `NotPluto` mutates `Pluto`: it rewrites `Pluto.run_reactive_core!` in
place via `Pluto.eval`, so any code path that calls it (not just `NotPluto.run`)
becomes non-reactive for the rest of the Julia session.
"""
run(; kwargs...) = Pluto.run(; run_notebook_on_load=false, kwargs...)

function _apply_patches!()
    m = first(methods(Pluto.run_reactive_core!))
    file = String(m.file)
    isfile(file) || error("NotPluto: source file $file is not on disk; cannot patch.")
    parsed = Meta.parseall(read(file, String); filename=file)
    expr = _find_function_named(parsed, m.name)
    expr === nothing && error("NotPluto: could not locate `$(m.name)` in $file.")

    transformed, n_kwarg, n_filter = _transform(expr)
    n_kwarg >= 2 || error(
        "NotPluto: expected ≥2 `topological_order_cached(...)` calls without " *
        "`allow_multiple_defs`, found $n_kwarg. Pluto's body may have changed."
    )
    n_filter == 1 || error(
        "NotPluto: expected exactly 1 `to_run = ...::Vector{Cell}` assignment, " *
        "found $n_filter. Pluto's body may have changed."
    )
    Pluto.eval(transformed)
    return nothing
end

function _find_function_named(node, name::Symbol)
    if node isa Expr
        if node.head === :function && _signature_name(node) === name
            return node
        end
        for a in node.args
            r = _find_function_named(a, name)
            r === nothing || return r
        end
    end
    return nothing
end

function _signature_name(funcexpr::Expr)
    sig = funcexpr.args[1]
    while sig isa Expr && sig.head !== :call
        sig = sig.args[1]
    end
    return sig isa Expr ? sig.args[1] : sig
end

function _transform(expr)
    n_kwarg = Ref(0)
    n_filter = Ref(0)
    out = _postwalk(expr) do e
        e isa Expr || return e
        if e.head === :call && _is_call_to(e, :topological_order_cached)
            new_e, added = _add_kwarg(e, :allow_multiple_defs, true)
            added && (n_kwarg[] += 1)
            return new_e
        elseif e.head === :block
            return _inject_filter_after_to_run(e, n_filter)
        end
        return e
    end
    return out, n_kwarg[], n_filter[]
end

function _postwalk(f, expr)
    if expr isa Expr
        new_args = Any[_postwalk(f, a) for a in expr.args]
        return f(Expr(expr.head, new_args...))
    else
        return f(expr)
    end
end

function _is_call_to(call::Expr, name::Symbol)
    isempty(call.args) && return false
    g = call.args[1]
    g === name && return true
    return g isa Expr && g.head === :. && length(g.args) == 2 && g.args[2] == QuoteNode(name)
end

# Returns (new_expr, added::Bool) — `added` is false if the kwarg was already there.
function _add_kwarg(call::Expr, name::Symbol, val)
    args = copy(call.args)
    if length(args) >= 2 && args[2] isa Expr && args[2].head === :parameters
        params = args[2]
        for kw in params.args
            if kw isa Expr && kw.head === :kw && kw.args[1] === name
                return call, false
            end
        end
        args[2] = Expr(:parameters, params.args..., Expr(:kw, name, val))
    else
        insert!(args, 2, Expr(:parameters, Expr(:kw, name, val)))
    end
    return Expr(:call, args...), true
end

function _is_to_run_typed_assignment(a)
    a isa Expr || return false
    a.head === :(=) || return false
    length(a.args) == 2 || return false
    a.args[1] === :to_run || return false
    rhs = a.args[2]
    rhs isa Expr || return false
    rhs.head === :(::) || return false
    return length(rhs.args) == 2 && rhs.args[2] == :(Vector{Cell})
end

function _inject_filter_after_to_run(blk::Expr, counter::Ref{Int})
    new_args = Any[]
    for a in blk.args
        push!(new_args, a)
        if _is_to_run_typed_assignment(a)
            push!(new_args, :(filter!(in(roots), to_run)))
            counter[] += 1
        end
    end
    return Expr(:block, new_args...)
end

__init__() = _apply_patches!()

end # module NotPluto
