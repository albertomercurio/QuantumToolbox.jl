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

"""
    check_operator(L, ref; label)

Compare a matrix-free operator against a materialized reference matrix: 3-arg `mul!`, 5-arg `mul!`
over a few `(α, β)` pairs, the adjoint, and `concretize`.

`(true, true)` is in the `(α, β)` list deliberately — it is what `SciMLOperators.AddedOperator`
passes for every term after the first, so it is the combination real code hits most.
"""
function check_operator(L, ref; label)
    return @testset "$label" begin
        v = randn(ComplexF64, size(L, 2))

        @test mul!(similar(v), L, v) ≈ ref * v

        for (α, β) in ((true, true), (0.7 + 0.2im, 0.0), (0.7 + 0.2im, 1.3 - 0.4im))
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
end

include("bosonic.jl")
include("tensor_product.jl")
