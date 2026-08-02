# ──────────────────────────────────────────────────────────────────────────────
# LocalTensorProductOperator — lazy I ⊗ … ⊗ Aⱼ ⊗ … ⊗ I
# ──────────────────────────────────────────────────────────────────────────────

@doc raw"""
    LocalTensorProductOperator{T, M, N, D, I, O, C} <: AbstractSciMLOperator{T}

Lazy tensor (Kronecker) product acting on a composite Hilbert space, in which only the
*non-identity* factors are stored:

```math
\hat{O} = \hat{\mathbb{1}}_{d_1} \otimes \cdots \otimes \hat{A}_j \otimes \cdots \otimes \hat{\mathbb{1}}_{d_N}
```

Identity factors are never materialized and are never applied.

# Why not `SciMLOperators.TensorProductOperator`?

`SciMLOperators` already provides `kron` for operators, which builds a nested tree of binary
`TensorProductOperator`s. `LocalTensorProductOperator` is a flat, sparse alternative that is
better suited to lattice/multi-mode Hamiltonians, where a term touches one or two sites out of
many:

  * **Identity factors are free.** Only `indices`/`ops` for the active subsystems are stored, so a
    two-site term in a 20-site chain applies exactly two operators. A `TensorProductOperator`
    built from `reduce(kron, (A, I, I, …))` still descends through — and applies — every identity
    factor.
  * **The number of active operators `M` is a type parameter**, so the application loop unrolls.
    A nested `TensorProductOperator`'s depth is a property of the runtime tree.
  * **Exactly two scratch buffers**, allocated once by [`cache_operator`](@ref), rather than one
    per nesting level.
  * **A genuine fast path** for the last subsystem — see the memory-layout note below.
  * **It compiles under Reactant/XLA**, whereas tracing a `TensorProductOperator` fails with
    "Scalar indexing is disallowed".

Its sub-operators are *not* adapted to the state's array type. `cache_operator` moves the work
buffers to the device, but a sub-operator wrapping a host matrix (e.g. `MatrixOperator(::Matrix)`)
stays on the host and `mul!` will fail on a GPU state. Build such sub-operators with device arrays
directly. `BosonicOperator`s are immune — they store only a dimension.

# Memory layout

The state vector is interpreted as `reshape(v, reverse(dims)...)`. Physics mode `j`
(`1` = leftmost/outermost in the Kronecker product, `N` = rightmost/innermost) therefore maps to
Julia array dimension `N - j + 1`:

  * physics mode `N` (innermost) → Julia dimension 1, i.e. **contiguous in memory**, so applying
    an operator there is a single `mul!` on a reshaped matrix — no permutation, no scratch buffer;
  * physics mode `1` (outermost) → Julia dimension `N`, the most strided case, which requires a
    `permutedims!` there and back.

# Acting on a matrix

A matrix state is a batch of column states, so `mul!(W, L, V)` is `L` applied to each column of `V`
— which is left multiplication `L * ρ` when `V` is a density matrix. The batch is a trailing axis
that is never contracted and never permuted, so this costs no more than the columns themselves.

The work buffers must be sized for the batch, so `cache_operator` has to be given a state of the
shape you intend to apply `L` to: an operator cached against a vector cannot be applied to a
matrix, and `mul!` throws rather than silently going wrong.

Right multiplication `ρ * L` works too — SciMLOperators rewrites it as `adjoint(L' * ρ')`
(`left.jl:7`), which lands back here. It is roughly 3× slower than `L * ρ` because the state it
forwards is an `Adjoint`, whose `reshape` is a `ReshapedArray` rather than a shared-memory view,
so `permutedims!` loses its fast path.

# Constructor

    LocalTensorProductOperator(dims::Tuple, idx₁ => op₁, idx₂ => op₂, …)

`dims` gives the dimension of *every* subsystem; the pairs give only the non-identity ones, of which
there must be at least one — for an identity on the whole space use
`SciMLOperators.IdentityOperator(prod(dims))`. Indices must be **strictly increasing** (operators on
distinct subsystems commute, so this is a normalization, not a restriction).

# Fields

  * `dims::D`    — `NTuple{N, Int}`, dimension of every subsystem, in physics order.
  * `indices::I` — `NTuple{M, Int}`, strictly increasing physics-mode indices of the active operators.
  * `ops::O`     — `NTuple{M, <:AbstractSciMLOperator}`, the active operators.
  * `cache::C`   — `nothing`, or the work buffers allocated by [`cache_operator`](@ref).

# Example

```julia
a = DestroyOperator{ComplexF64}(10)
L = LocalTensorProductOperator((10, 10, 10), 1 => a', 3 => a)   # a†₁ ⊗ 1 ⊗ a₃
L = cache_operator(L, randn(ComplexF64, 1000))
mul!(similar(v), L, v)
```
"""
struct LocalTensorProductOperator{T, M, N, D <: NTuple, I <: NTuple, O <: Tuple, C} <: AbstractSciMLOperator{T}
    dims::D
    indices::I
    ops::O
    cache::C

    function LocalTensorProductOperator(
            dims::D, indices::I, ops::O, cache::C,
        ) where {
            M, N,
            D <: NTuple{N, Int},
            I <: NTuple{M, Int},
            O <: Tuple{Vararg{AbstractSciMLOperator, M}},
            C,
        }
        M > 0 || throw(
            ArgumentError(
                "at least one operator is required; for an identity on the whole space use " *
                    "`SciMLOperators.IdentityOperator(prod(dims))`",
            ),
        )

        prev = 0
        for (idx, op) in zip(indices, ops)
            (1 <= idx <= N) || throw(ArgumentError("subsystem index $idx is out of range 1:$N"))
            idx > prev || throw(ArgumentError("indices must be strictly increasing, got $indices"))
            size(op, 1) == size(op, 2) ||
                throw(ArgumentError("operator on subsystem $idx must be square, got size $(size(op))"))
            size(op, 1) == dims[idx] || throw(
                ArgumentError("operator on subsystem $idx has size $(size(op, 1)) but dims[$idx] = $(dims[idx])"),
            )
            prev = idx
        end

        return new{_promote_op_eltype(ops), M, N, D, I, O, C}(dims, indices, ops, cache)
    end
end

# TODO: replace this with `Base.promote_eltype(ops...)` once the `SciMLOperators` compat lower
# bound includes the fix for `eltype` on operator *types* — then this helper can be deleted.
#
# It cannot be used as of SciMLOperators 1.25: `Base.promote_eltype` has a method
# `promote_eltype(v1::T, vs::T...) = eltype(T)` that wins whenever the sub-operators share a
# concrete type — two identical `MatrixOperator`s in a spin chain, say. That calls `eltype` on the
# *type*, and SciMLOperators only defines `Base.eltype(::Type{AbstractSciMLOperator{T}})` for the
# exact abstract type rather than `::Type{<:AbstractSciMLOperator{T}}`, so a concrete subtype falls
# through to `Base.eltype(::Type) = Any`. The operator would then silently be built as `{Any, …}`.
_promote_op_eltype(ops::Tuple) = promote_type(map(eltype, ops)...)

function LocalTensorProductOperator(dims::NTuple{N, Int}, pairs::Pair{Int, <:AbstractSciMLOperator}...) where {N}
    # Keeping these as tuples is what makes `M` a type parameter, so `mul!` can unroll.
    return LocalTensorProductOperator(dims, map(first, pairs), map(last, pairs), nothing)
end

Base.size(L::LocalTensorProductOperator) = (prod(L.dims), prod(L.dims))

SciMLOperators.islinear(::LocalTensorProductOperator) = true
SciMLOperators.has_adjoint(L::LocalTensorProductOperator) = all(has_adjoint, L.ops)


SciMLOperators.getops(L::LocalTensorProductOperator) = L.ops

Base.adjoint(L::LocalTensorProductOperator) =
    LocalTensorProductOperator(L.dims, L.indices, map(adjoint, L.ops), L.cache)

Base.:*(L::LocalTensorProductOperator, v::AbstractVecOrMat) = mul!(similar(v, Base.promote_eltype(L, v)), L, v)

SciMLOperators.isconvertible(::LocalTensorProductOperator) = false

function Base.show(io::IO, L::LocalTensorProductOperator{T, M, N}) where {T, M, N}
    return print(io, "LocalTensorProductOperator{$T}(dims=$(L.dims), $M active of $N subsystems)")
end

# ─── Caching ─────────────────────────────────────────────────────────────────
#
# A scratch buffer is needed whenever `mul!` has to permute, or has to chain several operators.
# The one exception is a single operator on the LAST subsystem: that subsystem is the
# fastest-varying axis of `reshape(v, reverse(dims)...)`, so `mul!` is one contiguous
# matrix-matrix product straight from `v` into `w`.

_is_contiguous_single(::LocalTensorProductOperator{T, M, N}) where {T, M, N} = false
_is_contiguous_single(L::LocalTensorProductOperator{T, 1, N}) where {T, N} = L.indices[1] == N

_needs_cache(L::LocalTensorProductOperator) = !_is_contiguous_single(L)

SciMLOperators.iscached(L::LocalTensorProductOperator) =
    (!_needs_cache(L) || !isnothing(L.cache)) && all(iscached, L.ops)

# `SciMLOperators.cache_operator` is defined as `cache_internals(cache_self(L, u), u)`, so these
# are the two hooks to implement rather than overriding `cache_operator` itself.
#
# `cache_self` allocates this operator's own work buffers. The one configuration that needs none —
# a single operator on the last (contiguous) subsystem — keeps `nothing`, which is what makes `mul!`
# take its buffer-free fast path.

"""
    cache_self(L::LocalTensorProductOperator, u::AbstractVecOrMat)

Allocate `L`'s two work buffers, sized for `u`. Returns `L` unchanged for a single operator acting
on the last subsystem, which needs none; see [`LocalTensorProductOperator`](@ref).

`u` may be a matrix — a batch of states, e.g. a density matrix — in which case the buffers are
sized for the whole batch. An operator cached against a vector therefore cannot be applied to a
matrix; `mul!` says so rather than silently going wrong.
"""
function SciMLOperators.cache_self(L::LocalTensorProductOperator, u::AbstractVecOrMat)
    _needs_cache(L) || return L
    n = length(u)
    return LocalTensorProductOperator(L.dims, L.indices, L.ops, (similar(u, n), similar(u, n)))
end

"""
    cache_internals(L::LocalTensorProductOperator, u::AbstractVecOrMat)

Cache each sub-operator against a slice of `u` of that subsystem's dimension. A slice rather than a
fresh array so that nothing is allocated: `cache_operator` uses its second argument only as a size
and type prototype, and allocates its own buffers from it with `similar`. Linear indexing makes the
same expression work whether `u` is a vector or a matrix.
"""
function SciMLOperators.cache_internals(L::LocalTensorProductOperator{T, M, N}, u::AbstractVecOrMat) where {T, M, N}
    cached_ops = ntuple(Val(M)) do j
        dk = L.dims[L.indices[j]]
        cache_operator(L.ops[j], @view(u[1:dk]))
    end

    return LocalTensorProductOperator(L.dims, L.indices, cached_ops, L.cache)
end

# The buffers are stored flat, so one cache serves both the vector and the matrix path. No error
# path here: `_check_cache` has already established that the length matches, which keeps the return
# type concrete — a `Union{Vector, SubArray}` here would propagate through the `iseven(j) ? buf : w`
# ping-pong below and cost an allocation on every call.
#
# The vector case returns the buffer untouched: `reshape` allocates a fresh array header even when
# it changes nothing, and this sits in the inner loop of every `sesolve` step.
_work_buffer(L::LocalTensorProductOperator, k::Int, ::AbstractVector) = L.cache[k]
_work_buffer(L::LocalTensorProductOperator, k::Int, v::AbstractMatrix) = reshape(L.cache[k], size(v))

# Work buffers are sized for the state they were cached against (`cache_self`), so applying `L` to
# a state of a different shape — the easy mistake being to cache with `ψ` and then apply to `ρ` —
# has to be caught here. Without it the failure surfaces as a `reshape` `DimensionMismatch` deep in
# `_apply_single_op_tensor_prod!`, which says nothing about the cache.
function _check_cache(L::LocalTensorProductOperator, v::AbstractVecOrMat)
    _needs_cache(L) || return nothing
    iscached(L) || throw(ArgumentError("Operator is not cached. Call `cache_operator(L, v)` first."))
    n = length(L.cache[1])
    n == length(v) || throw(
        ArgumentError(
            "operator was cached for $n elements but the state has $(length(v)). Call " *
                "`cache_operator(L, v)` with a state of the shape you intend to apply it to — " *
                "caching against a vector is not enough to apply `L` to a matrix.",
        ),
    )
    return nothing
end

_state_axes(L::LocalTensorProductOperator, ::AbstractVector) = reverse(L.dims)
_state_axes(L::LocalTensorProductOperator, v::AbstractMatrix) = (reverse(L.dims)..., size(v, 2))

# ─── 3-arg mul!: w = L * v ───────────────────────────────────────────────────

function LinearAlgebra.mul!(w::AbstractVector, L::LocalTensorProductOperator, v::AbstractVector)
    _check_mul_args(L, w, v)
    return _tensor_prod_mul!(w, L, v)
end

function LinearAlgebra.mul!(w::AbstractMatrix, L::LocalTensorProductOperator, v::AbstractMatrix)
    _check_mul_args(L, w, v)
    return _tensor_prod_mul!(w, L, v)
end

function _tensor_prod_mul!(w::AbstractVecOrMat, L::LocalTensorProductOperator{T, M, N}, v::AbstractVecOrMat) where {T, M, N}
    dims_rev = _state_axes(L, v)
    ops = L.ops

    if _is_contiguous_single(L)
        return _apply_single_op_tensor_prod!(w, ops[1], v, dims_rev, 1, nothing)
    end

    _check_cache(L, v)
    buf = _work_buffer(L, 1, v)

    current_src = v
    for j in 1:M
        idx = N - L.indices[j] + 1

        # Ping-pong between buf and w to avoid unnecessary allocations
        current_dst = iseven(j) ? buf : w
        current_buf = iseven(j) ? w : buf

        _apply_single_op_tensor_prod!(current_dst, ops[j], current_src, dims_rev, idx, current_buf)
        current_src = current_dst
    end

    # If M is even, the last write went to buf — copy to w
    if iseven(M)
        copyto!(w, buf)
    end

    return w
end

# Fast path: the target subsystem is already contiguous, so no permutation is needed and no
# scratch buffer is required.
function _apply_single_op_tensor_prod!(
        dst::AbstractVecOrMat, op, src::AbstractVecOrMat,
        dims_rev::NTuple{N, Int}, idx::Int, ::Nothing,
    ) where {N}
    dk = dims_rev[idx]
    rest = length(src) ÷ dk

    mul!(reshape(dst, dk, rest), op, reshape(src, dk, rest))
    return dst
end

function _apply_single_op_tensor_prod!(
        dst::AbstractVecOrMat, op, src::AbstractVecOrMat,
        dims_rev::NTuple{N, Int}, idx::Int, perm_buf::AbstractVecOrMat,
    ) where {N}
    dk = dims_rev[idx]
    rest = length(src) ÷ dk

    if idx == 1
        # Fast path: target dimension is already contiguous
        mul!(reshape(dst, dk, rest), op, reshape(src, dk, rest))
    else
        perm = ntuple(Val(N)) do j
            if j == 1
                return idx
            else
                return (j ≤ idx) ? j - 1 : j
            end
        end
        inv_perm = ntuple(Val(N)) do j
            if j == idx
                return 1
            else
                return (j < idx) ? j + 1 : j
            end
        end
        perm_dims = ntuple(i -> dims_rev[perm[i]], Val(N))

        permutedims!(reshape(dst, perm_dims...), reshape(src, dims_rev...), perm)
        mul!(reshape(perm_buf, dk, rest), op, reshape(dst, dk, rest))
        permutedims!(reshape(dst, dims_rev...), reshape(perm_buf, perm_dims...), inv_perm)
    end
    return dst
end

# ─── 5-arg mul!: w = α * L * v + β * w ───────────────────────────────────────

function LinearAlgebra.mul!(w::AbstractVector, L::LocalTensorProductOperator, v::AbstractVector, α, β)
    _check_mul_args(L, w, v)
    return _tensor_prod_mul!(w, L, v, α, β)
end

function LinearAlgebra.mul!(w::AbstractMatrix, L::LocalTensorProductOperator, v::AbstractMatrix, α, β)
    _check_mul_args(L, w, v)
    return _tensor_prod_mul!(w, L, v, α, β)
end

function _tensor_prod_mul!(
        w::AbstractVecOrMat,
        L::LocalTensorProductOperator{T, M, N},
        v::AbstractVecOrMat,
        α,
        β,
    ) where {T, M, N}
    # Fast exits for scalar coefficients
    if iszero(α)
        if iszero(β)
            fill!(w, zero(eltype(w)))
        elseif !isone(β)
            rmul!(w, β)
        end
        return w
    end

    # If β == 0, compute w <- L*v first, then scale by α if needed
    if iszero(β)
        _tensor_prod_mul!(w, L, v)
        if !isone(α)
            rmul!(w, α)
        end
        return w
    end

    # Contiguous single-operator case: fold α and β straight into the sub-operator's own 5-arg
    # `mul!` on the reshaped state. One pass, and no scratch buffer.
    if _is_contiguous_single(L)
        dk = L.dims[N]
        rest = length(v) ÷ dk
        mul!(reshape(w, dk, rest), L.ops[1], reshape(v, dk, rest), α, β)
        return w
    end

    # General case: compute tmp = L*v into an internal cached buffer, then w <- α*tmp + β*w
    _check_cache(L, v)
    buf2 = _work_buffer(L, 2, v)

    _tensor_prod_mul!(buf2, L, v)
    axpby!(α, buf2, β, w)
    return w
end

# ─── concretize: materialize the full Kronecker product ──────────────────────

function SciMLOperators.concretize(L::LocalTensorProductOperator{T, M, N}) where {T, M, N}
    # `ntuple` over `Val(N)` keeps the subsystem index a compile-time constant, so each factor is
    # built by its own specialization and the result is a tuple of concrete matrix types — unlike
    # accumulating into a `Vector{AbstractMatrix}`.
    factors = ntuple(Val(N)) do j
        i = findfirst(==(j), L.indices)
        isnothing(i) ? sparse(one(T) * I, L.dims[j], L.dims[j]) : concretize(L.ops[i])
    end

    return reduce(kron, factors)
end

# SciMLOperators reaches concretization of composite operators through `convert`, so an
# `AddedOperator` of these would otherwise fail to concretize. See the note in `bosonic.jl`.
Base.convert(::Type{AbstractMatrix}, L::LocalTensorProductOperator) = concretize(L)
SciMLOperators.has_concretization(L::LocalTensorProductOperator) = all(SciMLOperators.has_concretization, L.ops)
