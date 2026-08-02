# Reference for `LocalTensorProductOperator`: materialize the full Kronecker product, inserting a
# sparse identity wherever no operator was given. Deliberately independent of `L.indices`.
function _kron_ref(dims::NTuple{N, Int}, pairs::Pair{Int, <:AbstractMatrix}...) where {N}
    lookup = Dict(pairs)
    return reduce(kron, ntuple(j -> get(() -> sparse((1.0 + 0im)I, dims[j], dims[j]), lookup, j), N))
end

# Unequal dims throughout, so an index that is reversed or off by one cannot pass by coincidence.
const DIMS3 = (2, 3, 4)
const DIMS4 = (3, 4, 5, 2)

_matop(d) = MatrixOperator(randn(ComplexF64, d, d))

@testset "M = 1, each subsystem position" begin
    # Position N is the contiguous fast path (no permutation, no cache); position 1 is the most
    # strided. The interior positions are neither.
    for p in 1:length(DIMS4)
        A = _matop(DIMS4[p])
        L = LocalTensorProductOperator(DIMS4, p => A)
        ref = _kron_ref(DIMS4, p => A.A)
        check_operator(cache_operator(L, randn(ComplexF64, prod(DIMS4))), ref; label = "op on subsystem $p of 4")
    end
end

@testset "M = 2 and M = N" begin
    A1, A2, A3, A4 = _matop.(DIMS4)

    L = LocalTensorProductOperator(DIMS4, 1 => A1, 2 => A2)
    check_operator(cache_operator(L, randn(ComplexF64, prod(DIMS4))), _kron_ref(DIMS4, 1 => A1.A, 2 => A2.A); label = "adjacent (1,2)")

    L = LocalTensorProductOperator(DIMS4, 1 => A1, 4 => A4)
    check_operator(cache_operator(L, randn(ComplexF64, prod(DIMS4))), _kron_ref(DIMS4, 1 => A1.A, 4 => A4.A); label = "first and last (1,4)")

    L = LocalTensorProductOperator(DIMS4, 1 => A1, 2 => A2, 3 => A3, 4 => A4)
    ref = _kron_ref(DIMS4, 1 => A1.A, 2 => A2.A, 3 => A3.A, 4 => A4.A)
    check_operator(cache_operator(L, randn(ComplexF64, prod(DIMS4))), ref; label = "all four active")
end

@testset "Sum over subsystems (AddedOperator)" begin
    # This is the construction every user writes for a lattice Hamiltonian, and it is what
    # `AddedOperator` drives with `mul!(w, op, v, true, true)`. The n = N term takes the
    # contiguous fast path, which used to crash.
    dims = DIMS4
    v = randn(ComplexF64, prod(dims))
    ops = ntuple(j -> _matop(dims[j]), length(dims))

    H = cache_operator(sum(n -> LocalTensorProductOperator(dims, n => ops[n]), 1:length(dims)), v)
    H_ref = sum(n -> _kron_ref(dims, n => ops[n].A), 1:length(dims))

    @test mul!(similar(v), H, v) ≈ H_ref * v

    w = randn(ComplexF64, prod(dims))
    w0 = copy(w)
    @test mul!(w, H, v, true, true) ≈ H_ref * v + w0
end

@testset "Agrees with SciMLOperators.TensorProductOperator" begin
    # An oracle that does not go through our own `concretize`.
    dims = DIMS3
    A = _matop(dims[2])
    v = randn(ComplexF64, prod(dims))

    L = cache_operator(LocalTensorProductOperator(dims, 2 => A), v)
    S = cache_operator(reduce(kron, (IdentityOperator(dims[1]), A, IdentityOperator(dims[3]))), v)

    @test S isa TensorProductOperator

    # Compare the output buffers, not the return values: `TensorProductOperator`'s `mul!` writes
    # into `w` but returns a reshaped `Matrix` view of it, whereas ours returns `w` itself.
    w_local, w_sciml = similar(v), similar(v)
    mul!(w_local, L, v)
    mul!(w_sciml, S, v)
    @test w_local ≈ w_sciml
end

@testset "Mixed operator and element types" begin
    dims = (4, 3)
    a = DestroyOperator{Float64}(dims[1])          # real eltype
    A = MatrixOperator(randn(ComplexF64, dims[2], dims[2]))
    L = LocalTensorProductOperator(dims, 1 => a, 2 => A)

    @test eltype(L) == ComplexF64                  # promoted across the sub-operators
    check_operator(
        cache_operator(L, randn(ComplexF64, prod(dims))),
        _kron_ref(dims, 1 => concretize(a), 2 => A.A);
        label = "DestroyOperator ⊗ MatrixOperator",
    )
end

@testset "Matrix action never materializes the operator" begin
    # `isconvertible` is `false`, so SciMLOperators' convert-based `mul!` fallback announces itself
    # with a warning before materializing (`interface.jl:479`). A matrix state used to reach it
    # silently; `@test_nowarn` is the direct assertion that it no longer does.
    for p in (1, 3)                       # 1 = most strided (permutes), 3 = contiguous fast path
        L = cache_operator(
            LocalTensorProductOperator(DIMS3, p => _matop(DIMS3[p])),
            randn(ComplexF64, prod(DIMS3), 3),
        )
        V = randn(ComplexF64, prod(DIMS3), 3)
        W = randn(ComplexF64, size(V)...)

        @test !SciMLOperators.isconvertible(L)
        @test_nowarn mul!(similar(V), L, V)
        @test_nowarn mul!(W, L, V, true, true)
        @test_nowarn L * V
    end
end

@testset "Right multiplication" begin
    # `ρ * L` is serviced by SciMLOperators' `*(::AbstractVecOrMat, ::AbstractSciMLOperator)`
    # (`left.jl:7`), which rewrites it as `adjoint(L' * ρ')` — so it needs `*` to accept a matrix,
    # and the state it forwards is an `Adjoint`, not a plain `Matrix`.
    A = _matop(DIMS3[1])
    ref = _kron_ref(DIMS3, 1 => A.A)
    ρ = randn(ComplexF64, prod(DIMS3), prod(DIMS3))
    L = cache_operator(LocalTensorProductOperator(DIMS3, 1 => A), ρ)

    @test ρ * L ≈ ρ * ref
    @test L * ρ ≈ ref * ρ
    @test ρ * ref ≉ ref * ρ      # the two products differ, so the check above is not vacuous
end

@testset "Matrix action does not allocate per call" begin
    # `concretize(L)` here is a 1024×1024 matrix with no zero blocks — ~16 MiB, rebuilt on every
    # call by the convert-based fallback this replaced. The state is 32 KiB, so the threshold sits
    # two orders of magnitude below the regression and well above the real path's own overhead.
    d = (32, 32)
    V = randn(ComplexF64, prod(d), 2)
    L = cache_operator(LocalTensorProductOperator(d, 1 => _matop(d[1]), 2 => _matop(d[2])), V)

    W = similar(V)
    mul!(W, L, V)                          # compile before measuring
    @test (@allocated mul!(W, L, V)) < 100_000

    W2 = copy(V)
    mul!(W2, L, V, true, true)
    @test (@allocated mul!(W2, L, V, true, true)) < 100_000
end

@testset "Cache contract" begin
    v = randn(ComplexF64, prod(DIMS3))

    # Needs a cache: permutation and/or several operators.
    L = LocalTensorProductOperator(DIMS3, 1 => _matop(DIMS3[1]))
    @test !iscached(L)
    @test_throws ArgumentError mul!(similar(v), L, v)
    @test iscached(cache_operator(L, v))

    # Work buffers are sized for the state they were cached against, so a vector-cached operator
    # cannot be applied to a matrix. It must say so, not silently fall back to a materialized form.
    V = randn(ComplexF64, prod(DIMS3), 3)
    @test_throws ArgumentError mul!(similar(V), cache_operator(L, v), V)
    @test mul!(similar(V), cache_operator(L, V), V) ≈ concretize(cache_operator(L, V)) * V

    # Mismatched state shapes are rejected rather than reinterpreted.
    Lc = cache_operator(L, V)
    @test_throws DimensionMismatch mul!(randn(ComplexF64, prod(DIMS3), 2), Lc, V)
    @test_throws DimensionMismatch mul!(similar(V), Lc, randn(ComplexF64, prod(DIMS3) + 1, 3))

    # Needs no cache: single operator on the last subsystem, contiguous in memory.
    A = _matop(DIMS3[3])
    L_fast = LocalTensorProductOperator(DIMS3, 3 => A)
    @test iscached(L_fast)
    ref = _kron_ref(DIMS3, 3 => A.A)
    @test mul!(similar(v), L_fast, v) ≈ ref * v
    w = randn(ComplexF64, prod(DIMS3))
    w0 = copy(w)
    @test mul!(w, L_fast, v, true, true) ≈ ref * v + w0
end

@testset "Constructor rejects invalid input" begin
    A = _matop(3)
    @test_throws ArgumentError LocalTensorProductOperator(DIMS3)                            # no operators
    @test_throws ArgumentError LocalTensorProductOperator(DIMS3, 0 => A)                    # index below range
    @test_throws ArgumentError LocalTensorProductOperator(DIMS3, 4 => A)                    # index above range
    @test_throws ArgumentError LocalTensorProductOperator(DIMS3, 2 => A, 2 => A)            # duplicate index
    @test_throws ArgumentError LocalTensorProductOperator(DIMS3, 3 => A, 2 => _matop(3))    # not increasing
    @test_throws ArgumentError LocalTensorProductOperator(DIMS3, 1 => A)                    # dims[1] = 2 ≠ 3
    @test_throws ArgumentError LocalTensorProductOperator(DIMS3, 2 => MatrixOperator(randn(3, 4)))  # not square
end

@testset "Type stability" begin
    for state in (randn(ComplexF64, prod(DIMS4)), randn(ComplexF64, prod(DIMS4), 3))
        u = similar(state)

        L = cache_operator(LocalTensorProductOperator(DIMS4, 1 => _matop(DIMS4[1]), 3 => _matop(DIMS4[3])), state)
        @test (@inferred mul!(u, L, state)) === u
        @test (@inferred mul!(u, L, state, 1.0, 0.0)) === u

        L_fast = LocalTensorProductOperator(DIMS4, 4 => _matop(DIMS4[4]))
        @test (@inferred mul!(u, L_fast, state)) === u
        @test (@inferred mul!(u, L_fast, state, 1.0, 1.0)) === u
    end
end
