# Plain `Test.jl` for now. If this subpackage is later wired into the root test harness (which
# uses TestItemRunner), the migration is mechanical: `@testset "X" begin` → `@testitem "X" begin`,
# move the `using` lines into each item, and delete this file.

using QuantumToolboxOperators
using LinearAlgebra
using SparseArrays
using SciMLOperators
using SciMLOperators: AdjointOperator, IdentityOperator, MatrixOperator, ScaledOperator, TensorProductOperator
using Random
using Test

Random.seed!(20260802)

const ALPHA_BETA = ((true, true), (0.7 + 0.2im, 0.0), (0.7 + 0.2im, 1.3 - 0.4im))
# `(true, true)` is in the list deliberately — it is what `SciMLOperators.AddedOperator` passes for
# every term after the first, so it is the combination real code hits most.

"""
    check_operator(L, ref; label, ncols = 3)

Compare a matrix-free operator against a materialized reference matrix: 3-arg `mul!`, 5-arg `mul!`
over a few `(α, β)` pairs, the adjoint, the allocating `*`, and `concretize`.

Everything is checked twice — once on a vector state and once on an `ncols`-column matrix state,
which is what a density matrix is under left multiplication. `ncols` differs from every subsystem
dimension used in the tests, so a batch axis confused with a subsystem axis cannot pass.

The matrix arm re-caches `L` against the matrix, because work buffers have to be sized for the
whole batch.
"""
function check_operator(L, ref; label, ncols = 3)
    return @testset "$label" begin
        @testset "vector state" begin
            v = randn(ComplexF64, size(L, 2))

            @test mul!(similar(v), L, v) ≈ ref * v

            for (α, β) in ALPHA_BETA
                w = randn(ComplexF64, size(L, 1))
                w0 = copy(w)
                @test mul!(w, L, v, α, β) ≈ α * (ref * v) + β * w0
            end

            # `v` must not be clobbered by any of the above
            v0 = copy(v)
            mul!(similar(v), L, v)
            @test v == v0

            @test mul!(similar(v), L', v) ≈ ref' * v
            @test concretize(L) ≈ ref

            # Allocating form, which `LinearAlgebra.dot(v, L, v)` and `QuantumToolbox.expect` need.
            @test L * v ≈ ref * v
            @test dot(v, L, v) ≈ dot(v, ref * v)
        end

        @testset "matrix state ($ncols columns)" begin
            V = randn(ComplexF64, size(L, 2), ncols)
            Lm = cache_operator(L, V)

            @test mul!(similar(V), Lm, V) ≈ ref * V

            for (α, β) in ALPHA_BETA
                W = randn(ComplexF64, size(L, 1), ncols)
                W0 = copy(W)
                @test mul!(W, Lm, V, α, β) ≈ α * (ref * V) + β * W0
            end

            V0 = copy(V)
            mul!(similar(V), Lm, V)
            @test V == V0

            @test mul!(similar(V), cache_operator(L', V), V) ≈ ref' * V
            @test Lm * V ≈ ref * V

            # Right multiplication, via SciMLOperators' `*(::AbstractVecOrMat, ::AbstractSciMLOperator)`.
            @test V' * Lm ≈ V' * ref

            # Column c of the result must be `L` applied to column c and nothing else.
            W = mul!(similar(V), Lm, V)
            Lv = cache_operator(L, V[:, 1])
            @test W[:, 1] ≈ mul!(similar(V[:, 1]), Lv, V[:, 1])
        end
    end
end

include("bosonic.jl")
include("tensor_product.jl")
