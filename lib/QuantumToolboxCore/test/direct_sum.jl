using Test
using QuantumToolboxCore
using LinearAlgebra
using SparseArrays

@testset "Direct sum" begin
    k2, k3 = basis(2, 0), basis(3, 1)
    b2, b3 = k2', k3'
    A, B, C = sigmax(), num(3), sigmaz()
    v2, v3 = mat2vec(A), mat2vec(B)
    L2, L3 = spre(A), spre(B)

    @testset "SumSpace" begin
        S = SumSpace(Space(2), Space(3))
        @test get_size(S) == 5
        @test length(S) == 2
        @test collect(S) == [Space(2), Space(3)]
        @test length(SumSpace(Space(2))) == 1 # a single summand stays a SumSpace
        @test SumSpace(Space(2)) != Space(2)
        @test SumSpace([Space(2), Space(3)]) == S # Vector storage
        @test_throws DomainError SumSpace(())
        @test_throws ArgumentError SumSpace(TensorSpace(Space(2), Space(2)), Space(3))
        @test_throws ArgumentError TensorSpace(S, Space(2))
        @test_throws ArgumentError k2 ⊗ (k2 ⊕ k3)

        # ⊕ of spaces is flat, the constructor keeps the nesting, and == ignores it
        @test Space(2) ⊕ Space(3) ⊕ Space(2) == SumSpace(Space(2), Space(3), Space(2))
        @test length(Space(2) ⊕ Space(3) ⊕ Space(2)) == 3
        nested = SumSpace(S, Space(2))
        @test length(nested) == 2
        @test nested == SumSpace(Space(2), Space(3), Space(2))
        @test hash(nested) == hash(SumSpace(Space(2), Space(3), Space(2)))
        @test SumSpace(Space(3), Space(2)) != S
    end

    @testset "⊕ (several arguments)" begin
        ψ = k2 ⊕ k3
        @test isket(ψ) && size(ψ) == (5,)
        @test ψ.dimensions == Dimensions(SumSpace(Space(2), Space(3)), Space(1))
        @test ψ.data == [1, 0, 0, 1, 0]
        @test ψ == direct_sum(k2, k3) == [k2; k3] == direct_sum([k2, k3])
        @test (k2 ⊕ k3)' == b2 ⊕ b3 == [b2 b3]
        @test v2 ⊕ v3 == direct_sum([A, B])
        @test isoperket(v2 ⊕ v3) && isoperbra(v2' ⊕ v3')

        H = A ⊕ B
        @test isoper(H) && size(H) == (5, 5)
        @test H.data isa SparseMatrixCSC
        @test Matrix(H.data) == cat(Matrix(A.data), Matrix(B.data); dims = (1, 2))
        @test (to_dense(A) ⊕ to_dense(B)).data isa SparseMatrixCSC
        @test issuper(L2 ⊕ L3) && size(L2 ⊕ L3) == (13, 13)

        # numbers are 1-dimensional summands, and take the type of the other arguments
        @test (k2 ⊕ 1).data == [1, 0, 1]
        @test isbra(b2 ⊕ 1) && (b2 ⊕ 1)' == k2 ⊕ 1
        @test isoper(A ⊕ 1) && isoper(1 ⊕ 2)

        # flat chains, a single argument, mixed types
        @test (k2 ⊕ k3 ⊕ k2).dimensions.to == SumSpace(Space(2), Space(3), Space(2))
        @test length((k2 ⊕ k3 ⊕ k2).dimensions.to) == 3
        @test direct_sum(k2).dimensions.to == SumSpace(Space(2)) && direct_sum(k2).data == k2.data
        @test direct_sum(b2) == direct_sum(k2)'
        @test direct_sum(A).dimensions == Dimensions(SumSpace(Space(2)), SumSpace(Space(2)))
        @test direct_sum(Space(2)) == SumSpace(Space(2))
        @test isoper(direct_sum(1)) && direct_sum(1).dimensions.to == SumSpace(Space(1))
        @test_throws ArgumentError k2 ⊕ A
        @test_throws ArgumentError k2 ⊕ b2
        @test_throws ArgumentError A ⊕ L2
        @test_throws ArgumentError direct_sum()

        @test @inferred(k2 ⊕ k3) isa QuantumObject
        @test @inferred(A ⊕ B) isa QuantumObject
    end

    @testset "1D list" begin
        @test direct_sum([k2, 1]).data == [1, 0, 1]
        @test isbra(direct_sum([b2, b3]))
        @test isoperket(direct_sum([A, v3]))
        @test direct_sum([k2]) == direct_sum(k2)
        @test_throws ArgumentError direct_sum([])
        @test_throws ArgumentError direct_sum([k2, b3])
        @test_throws ArgumentError direct_sum([L2, L3])
    end

    @testset "2D matrix of blocks" begin
        X = Qobj(rand(ComplexF64, 2, 3); dims = Dimensions(Space(2), Space(3)))
        H = direct_sum([A X; nothing B])
        @test isoper(H) && H.data isa SparseMatrixCSC
        @test Matrix(H.data) ≈ [Matrix(A.data) Matrix(X.data); zeros(3, 2) Matrix(B.data)]
        @test direct_sum([[A, X], [nothing, B]]) == H # QuTiP form
        @test direct_sum(Any[Any[A, X], Any[nothing, B]]) == H
        @test direct_sum([A nothing; nothing B]) == A ⊕ B
        @test size(direct_sum([k2 nothing; nothing k3])) == (5, 2)
        @test direct_sum([1 2; 3 4]).data == [1 2; 3 4]
        @test issuper(direct_sum([L2 nothing; nothing L3]))

        # blocks that are direct sums themselves are placed by their size
        @test direct_sum([(A ⊕ C) nothing; nothing sigmay()]) == A ⊕ C ⊕ sigmay()
        @test direct_sum([(A ⊕ B) nothing; nothing destroy(4)]) == A ⊕ B ⊕ destroy(4)

        @test_throws ArgumentError direct_sum([[A, B]]) # a single row
        @test_throws ArgumentError direct_sum([[A], [B]]) # a single column
        @test_throws ArgumentError direct_sum([[A, nothing], [B]]) # rows of different lengths
        @test_throws ArgumentError direct_sum([A nothing; nothing nothing]) # empty row
        @test_throws ArgumentError direct_sum([A B; nothing B]) # inconsistent row
        @test_throws ArgumentError direct_sum([A L2; nothing L3]) # Hilbert and Liouville blocks
        @test_throws ArgumentError direct_sum([v2 nothing; nothing v3]) # both space and superspace dims
    end

    @testset "direct_component" begin
        ψ = k2 ⊕ k3 ⊕ k2
        @test direct_component(ψ, 2) == k3
        @test direct_component(ψ, 1:2) == k2 ⊕ k3
        @test direct_component(ψ, 1:1) == direct_sum(k2)
        @test direct_component(ψ', 2) == k3'
        @test direct_component(v2 ⊕ v3, 2) == v3
        @test direct_component((v2 ⊕ v3)', 1) == v2'
        H = A ⊕ B ⊕ A
        @test direct_component(H, 2, 2) == B
        @test iszero(direct_component(H, 1, 2).data) && size(direct_component(H, 1, 2)) == (2, 3)
        @test direct_component(H, 2:3, 2:3) == B ⊕ A
        @test direct_component(L2 ⊕ L3, 2, 2) == L3
        nested = Qobj(ψ.data; dims = Dimensions(SumSpace(SumSpace(Space(2), Space(3)), Space(2)), Space(1)))
        @test direct_component(nested, 1) == k2 ⊕ k3

        @test_throws ArgumentError direct_component(ψ, 4)
        @test_throws ArgumentError direct_component(ψ, 2:1)
        @test_throws ArgumentError direct_component(k2, 1)
        @test_throws ArgumentError direct_component(ψ, 1, 1)
        @test_throws ArgumentError direct_component(H, 1)
        @test @inferred(direct_component(H, 2, 2)) isa QuantumObject
    end

    @testset "set_direct_component" begin
        ψ = k2 ⊕ k3
        @test set_direct_component(ψ, 2k3, 2) == k2 ⊕ 2k3
        @test ψ == k2 ⊕ k3 # unchanged
        @test set_direct_component(ψ, nothing, 1) == 0k2 ⊕ k3
        @test set_direct_component(ψ', 2k3', 2) == (k2 ⊕ 2k3)'
        @test set_direct_component(k2 ⊕ k3 ⊕ k2, 2(k2 ⊕ k3), 1:2) == 2k2 ⊕ 2k3 ⊕ k2
        @test set_direct_component(k2 ⊕ k3 ⊕ k2, 2k2, 1, 3k2, 3) == 2k2 ⊕ k3 ⊕ 3k2
        real_ψ = Qobj([1.0, 0, 0, 1, 0]; dims = Dimensions(SumSpace(Space(2), Space(3)), Space(1)))
        @test eltype(set_direct_component(real_ψ, 1im * k3, 2)) == ComplexF64

        H = A ⊕ B
        @test set_direct_component(H, 2B, 2, 2) == A ⊕ 2B
        @test set_direct_component(H, nothing, 1, 1, 2B, 2, 2) == 0A ⊕ 2B
        X = Qobj(ones(2, 3); dims = Dimensions(Space(2), Space(3)))
        @test direct_component(set_direct_component(H, X, 1, 2), 1, 2) == X
        @test set_direct_component(L2 ⊕ L3, 2L3, 2, 2) == L2 ⊕ 2L3

        @test_throws DimensionMismatch set_direct_component(ψ, k2, 2)
        @test_throws ArgumentError set_direct_component(ψ, k2, 3)
        @test_throws ArgumentError set_direct_component(k2, k2, 1)
        @test_throws ArgumentError set_direct_component(H, A, 1)
    end

    @testset "building from components" begin
        S = SumSpace(Space(2), Space(3))
        @test direct_sum(S, 2 => k3) == 0k2 ⊕ k3
        X = Qobj(ones(2, 3); dims = Dimensions(Space(2), Space(3)))
        @test direct_component(direct_sum(S, S, (1, 2) => X), 1, 2) == X
    end

    @testset "dims (SumDims)" begin
        ψ = k2 ⊕ k3
        @test ψ.dims == (SumDims(2, 3), [1])
        @test string(ψ.dims[1]) == "{2, 3}"
        @test (A ⊕ B).dims == (SumDims(2, 3), SumDims(2, 3))
        @test direct_sum(k2).dims == (SumDims(2), [1])
        # no clash with the dims of tensor products and of Liouville spaces
        @test (k2 ⊕ k2).dims != (k2 ⊗ k2).dims
        @test (k2 ⊕ k2).dims != mat2vec(sigmax()).dims
        @test (A ⊕ A).dims != spre(A).dims
        nested = SumSpace(SumSpace(Space(2), Space(3)), Space(2))
        @test QuantumToolboxCore.dimensions_to_dims(Dimensions(nested, Space(1))) == (SumDims(2, 3, 2), [1])

        # the dims can be read back
        for x in (ψ, ψ', A ⊕ B, v2 ⊕ v3, L2 ⊕ L3, direct_sum(k2))
            @test Qobj(x.data; type = x.type, dims = x.dims).dimensions == x.dimensions
        end
        @test Qobj(ψ.data; dims = SumDims(2, 3)) == ψ
        @test Qobj((A ⊕ B).data; dims = SumDims(2, 3)) == A ⊕ B
        @test_throws DimensionMismatch Qobj((A ⊕ B).data; dims = SumDims(2, 2))
    end

    @testset "functions that need subsystems" begin
        ψ = normalize(k2 ⊕ k3)
        ρ = ket2dm(ψ)
        ψ22 = normalize(k2 ⊕ basis(2, 1))
        @test_throws ArgumentError ptrace(ρ, 1)
        @test_throws ArgumentError ptrace(ψ', 1)
        @test_throws ArgumentError permute(ψ, (1,))
        @test_throws ArgumentError partial_transpose(ρ, [true])
        @test_throws ArgumentError negativity(ρ, 1)
        @test_throws ArgumentError entanglement(ψ, 1)
        @test_throws ArgumentError entropy_conditional(ρ, 1)
        @test_throws ArgumentError entropy_mutual(ρ, 1, 2)
        @test_throws ArgumentError concurrence(ψ22)
        @test_throws ArgumentError multisite_operator(2, 1 => A ⊕ B)
        @test expect(A ⊕ B, ψ) ≈ 0.5
    end
end
