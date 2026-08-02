# ──────────────────────────────────────────────────────────────────────────────
# Shared argument checking
# ──────────────────────────────────────────────────────────────────────────────

"""
    _check_mul_args(L, w, v)

Validate the state arguments of `mul!(w, L, v)`, where `w` and `v` may each be a vector or a
matrix (a batch of column states).

`SciMLOperators` has no generic version of this — each concrete operator asserts its own sizes
(`block.jl`, `basic.jl`) and `FunctionOperator` has a private `_sizecheck` — so operators here have
to do it themselves. It matters more here than for a `MatrixOperator`, whose underlying `mul!`
would catch a bad size anyway: these operators address the state through views like `v[2:N, :]`, so
an over-long `v` produces a plausible answer computed from a prefix of it rather than an error.
"""
function _check_mul_args(L, w::AbstractVecOrMat, v::AbstractVecOrMat)
    size(v, 1) == size(L, 2) || throw(
        DimensionMismatch(
            "operator has $(size(L, 2)) columns but the input state has leading dimension $(size(v, 1))",
        ),
    )
    size(w, 1) == size(L, 1) || throw(
        DimensionMismatch(
            "operator has $(size(L, 1)) rows but the output state has leading dimension $(size(w, 1))",
        ),
    )
    size(w, 2) == size(v, 2) || throw(
        DimensionMismatch(
            "input state has $(size(v, 2)) columns but the output state has $(size(w, 2))",
        ),
    )
    return nothing
end
