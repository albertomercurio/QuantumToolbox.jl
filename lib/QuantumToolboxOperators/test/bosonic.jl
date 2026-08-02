# Reference matrices are built from the truncated sparse â, so `check_operator` compares against
# an oracle that never touches this package's own coefficient formulas.
_sparse_destroy(N, T = ComplexF64) = spdiagm(1 => T[sqrt(i) for i in 1:(N - 1)])

@testset "Bosonic operators vs. sparse reference" begin
    # Nothing in the implementation branches on N, so two sizes are enough: a small one where the
    # truncation boundary is close, and a larger one.
    for N in (5, 20)
        A = _sparse_destroy(N)
        a = DestroyOperator{ComplexF64}(N)

        check_operator(a, A; label = "â (N=$N)")
        check_operator(a', A'; label = "â† (N=$N)")
        check_operator(NumberOperator{ComplexF64}(N), A' * A; label = "n̂ (N=$N)")

        for k in (2, 3)
            check_operator(DestroyPowerOperator{ComplexF64}(N, k), A^k; label = "â^$k (N=$N)")
            check_operator(adjoint(DestroyPowerOperator{ComplexF64}(N, k)), (A^k)'; label = "(â†)^$k (N=$N)")
        end

        for (k, n) in ((2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3))
            check_operator(
                NormalOrderedOperator{ComplexF64}(N, k, n), (A')^k * A^n;
                label = "(â†)^$k â^$n (N=$N)",
            )
        end
    end
end

@testset "ââ† at the truncation boundary" begin
    # `a * a'` simplifies to NumberOperator(shift = 1) = diag(1, 2, …, N), i.e. the *untruncated*
    # ââ† = n̂ + 1. The truncated sparse product instead gives diag(1, …, N-1, 0), because â† maps
    # the top Fock state out of the retained space. The two agree everywhere except that last
    # entry. This is pinned so the difference stays visible; see STATUS.md.
    N = 5
    A = _sparse_destroy(N)
    lazy = concretize(DestroyOperator{ComplexF64}(N) * adjoint(DestroyOperator{ComplexF64}(N)))

    @test diag(lazy) ≈ ComplexF64.(1:N)
    @test diag(A * A') ≈ ComplexF64[1:(N - 1)..., 0]
    @test lazy[1:(N - 1), 1:(N - 1)] ≈ (A * A')[1:(N - 1), 1:(N - 1)]
end

@testset "Algebraic simplification" begin
    N = 10
    a = DestroyOperator(N)
    ad = adjoint(a)

    @test (ad * a) isa NumberOperator
    @test (ad * a).shift == 0
    @test (a * ad) isa NumberOperator
    @test (a * ad).shift == 1

    @test (a^1) === a
    @test adjoint(ad) === a
    @test adjoint(NumberOperator(N)) === NumberOperator(N) || adjoint(NumberOperator(N)).shift == 0

    for k in (2, 3, 5)
        @test (a^k) isa DestroyPowerOperator
        @test (a^k).k == k
        @test (ad^k).L isa DestroyPowerOperator
        @test (ad^k).L.k == k
    end

    # powers add under composition
    @test (a * a).k == 2
    @test (a * DestroyPowerOperator(N, 2)).k == 3
    @test (DestroyPowerOperator(N, 2) * a).k == 3
    @test (DestroyPowerOperator(N, 2) * DestroyPowerOperator(N, 3)).k == 5
    @test (ad * ad).L.k == 2

    # mixed creation/annihilation → NormalOrderedOperator
    for (k, n) in ((2, 3), (3, 2), (2, 1), (1, 3))
        L = (ad^k) * (a^n)
        @test L isa NormalOrderedOperator
        @test (L.k, L.n) == (k, n)
    end

    # a scalar in the middle must not block the rewrite
    Δ = 2.5
    @test ((Δ * ad) * a) isa ScaledOperator
    @test ((Δ * ad) * a).L isa NumberOperator

    # eltype promotes across mixed-precision operands
    a32, a64 = DestroyOperator{Float32}(N), DestroyOperator{Float64}(N)
    @test (adjoint(a32) * a64) isa NumberOperator{Float64}
    @test (a32 * a64) isa DestroyPowerOperator{Float64}
    @test (a32^2) isa DestroyPowerOperator{Float32}
    @test ((adjoint(a32)^2) * (a64^3)) isa NormalOrderedOperator{Float64}
end

@testset "SciMLOperators composition" begin
    N = 10
    T = ComplexF64
    a = DestroyOperator{T}(N)
    A = _sparse_destroy(N)
    v = randn(T, N)

    # No rule for `+`, so this stays a generic AddedOperator — still has to be correct.
    @test (a + a') isa SciMLOperators.AddedOperator
    check_operator(cache_operator(a + a', v), A + A'; label = "â + â†")
    check_operator(cache_operator(2.0 * a, v), 2.0 * A; label = "2â")

    # The single-photon-driven Kerr Hamiltonian from the workshop notebook.
    Δ, U, F = 0.1, 0.2, 0.3
    H = Δ * a' * a + U * (a'^2 * a^2) + F * (a + a')
    H_ref = Δ * (A' * A) + U * (A'^2 * A^2) + F * (A + A')
    check_operator(cache_operator(H, v), H_ref; label = "Kerr Hamiltonian")
end

@testset "Caching as a factor of a TensorProductOperator" begin
    # A `TensorProductOperator` caches its factors by handing each one the *whole* state vector.
    # SciMLOperators' generic `AdjointOperator` caching reshapes that down to the factor's own size
    # and throws; our no-op `cache_internals` is what prevents it.
    d = 4
    ad = adjoint(DestroyOperator{ComplexF64}(d))
    S = cache_operator(TensorProductOperator(ad, IdentityOperator(d)), randn(ComplexF64, d^2))

    v = randn(ComplexF64, d^2)
    w = similar(v)
    mul!(w, S, v)
    @test w ≈ kron(concretize(ad), sparse((1.0 + 0im)I, d, d)) * v
end

@testset "Element type is preserved" begin
    N = 10
    v32 = randn(ComplexF32, N)
    w32 = similar(v32)

    for op in (
            DestroyOperator(N), adjoint(DestroyOperator(N)), NumberOperator(N),
            DestroyPowerOperator(N, 2), NormalOrderedOperator{Float32}(N, 2, 3),
        )
        mul!(w32, op, v32)
        @test eltype(w32) == ComplexF32
    end

    # Float32 and Float64 paths must agree to Float32 precision
    a = DestroyOperator(N)
    v64 = ComplexF64.(v32)
    @test ComplexF64.(mul!(similar(v32), a, v32)) ≈ mul!(similar(v64), a, v64) atol = 1.0e-6
end

@testset "Action on matrices" begin
    # `check_operator` already compares the matrix path against the sparse reference. This adds a
    # second, independent oracle — the matrix result must equal the columns computed one at a time
    # — plus the two things only a matrix state can exercise: right multiplication `ρ Â`, and the
    # requirement that neither side allocate, since a density matrix is where a fallback that
    # materializes the operator would hurt most.
    N, ncols = 10, 4
    V = randn(ComplexF64, N, ncols)
    ρ = randn(ComplexF64, N, N)

    for op in (
            DestroyOperator(N), adjoint(DestroyOperator(N)),
            NumberOperator(N), NumberOperator(N; shift = 1),
            DestroyPowerOperator(N, 2), adjoint(DestroyPowerOperator(N, 2)),
            NormalOrderedOperator(N, 2, 3),
        )
        ref = concretize(op)

        W, W_col = similar(V), similar(V)
        mul!(W, op, V)
        for c in 1:ncols
            mul!(view(W_col, :, c), op, view(V, :, c))
        end
        @test W ≈ W_col

        α, β = randn(ComplexF64), randn(ComplexF64)
        W2 = randn(ComplexF64, N, ncols)
        W2_col = copy(W2)
        mul!(W2, op, V, α, β)
        for c in 1:ncols
            mul!(view(W2_col, :, c), op, view(V, :, c), α, β)
        end
        @test W2 ≈ W2_col

        # Left and right multiplication of a density matrix. `ρ * op` is serviced by
        # SciMLOperators' `*(::AbstractVecOrMat, ::AbstractSciMLOperator)`, which routes through
        # `adjoint` back into the methods here — so it stays matrix-free too.
        @test op * ρ ≈ ref * ρ
        @test ρ * op ≈ ρ * ref

        Wρ = similar(ρ)
        mul!(Wρ, op, ρ)                    # compile before measuring
        @test (@allocated mul!(Wρ, op, ρ)) == 0
    end
end

@testset "Composite operators act on matrices" begin
    # An `AddedOperator` of `ScaledOperator`s over a matrix state: the case a density-matrix solver
    # actually builds. Correctness comes from `check_operator`; what is pinned here is that the
    # composite stays lazy rather than concretizing each term.
    N = 10
    a = DestroyOperator{ComplexF64}(N)
    A = _sparse_destroy(N)
    ρ = randn(ComplexF64, N, N)

    Δ, U, F = 0.1, 0.2, 0.3
    H = cache_operator(Δ * a' * a + U * (a'^2 * a^2) + F * (a + a'), ρ)
    H_ref = Δ * (A' * A) + U * (A'^2 * A^2) + F * (A + A')

    W = similar(ρ)
    @test mul!(W, H, ρ) ≈ H_ref * ρ
    @test H * ρ ≈ H_ref * ρ
    @test ρ * H ≈ ρ * H_ref

    mul!(W, H, ρ)
    @test (@allocated mul!(W, H, ρ)) == 0
end

@testset "Type stability" begin
    N = 10

    for state in (randn(ComplexF64, N), randn(ComplexF64, N, 3))
        u = similar(state)

        for op in (
                DestroyOperator(N), adjoint(DestroyOperator(N)), NumberOperator(N),
                DestroyPowerOperator(N, 2), adjoint(DestroyPowerOperator(N, 2)),
                NormalOrderedOperator(N, 2, 3),
            )
            @test (@inferred mul!(u, op, state)) === u
            @test (@inferred mul!(u, op, state, 1.0, 0.0)) === u
        end
    end
end

@testset "Mismatched state sizes are rejected" begin
    # These operators address the state through views like `v[2:N, :]`, so an over-long `v` used to
    # produce a plausible answer computed from a prefix of it, with no error at all.
    N = 10
    ops = (
        DestroyOperator(N), adjoint(DestroyOperator(N)), NumberOperator(N),
        DestroyPowerOperator(N, 2), adjoint(DestroyPowerOperator(N, 2)),
        NormalOrderedOperator(N, 2, 3),
    )
    for op in ops
        @test_throws DimensionMismatch mul!(randn(ComplexF64, N), op, randn(ComplexF64, 2N))
        @test_throws DimensionMismatch mul!(randn(ComplexF64, 2N), op, randn(ComplexF64, N))
        @test_throws DimensionMismatch mul!(randn(ComplexF64, N, 3), op, randn(ComplexF64, N, 4))
        @test_throws DimensionMismatch mul!(randn(ComplexF64, N), op, randn(ComplexF64, 2N), 1.0, 0.0)
        @test_throws DimensionMismatch mul!(randn(ComplexF64, N, 3), op, randn(ComplexF64, N, 4), 1.0, 0.0)
    end
end

@testset "Constructors reject invalid input" begin
    @test_throws ArgumentError DestroyOperator(0)
    @test_throws ArgumentError DestroyPowerOperator(10, 0)
    @test_throws ArgumentError NormalOrderedOperator(10, 0, 1)
    @test_throws ArgumentError NormalOrderedOperator(10, 1, 0)
    @test_throws ArgumentError NormalOrderedOperator(3, 3, 3)   # N too small for (â†)³â³
    @test_throws ArgumentError DestroyOperator(10)^0
end
