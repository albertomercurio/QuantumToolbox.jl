#=
Direct sums of quantum objects, which live in (or act on) a `SumSpace`.
=#

export direct_sum, direct_component, set_direct_component

@doc raw"""
    direct_sum(A₁, A₂, ...)
    A₁ ⊕ A₂

Return the direct sum of spaces, or of [`QuantumObject`](@ref)s of the same type:
- spaces: the [`SumSpace`](@ref) of them;
- [`Ket`](@ref)s (or [`OperatorKet`](@ref)s): the vector made of the given ones, one after the other, and in the same way for [`Bra`](@ref)s (or [`OperatorBra`](@ref)s);
- [`Operator`](@ref)s (or [`SuperOperator`](@ref)s): the block-diagonal matrix made of the given ones.

Numbers are summands of dimension 1, and take the type of the other arguments (only numbers give an [`Operator`](@ref)). A single argument gives a direct sum with a single summand.

The direct sum is associative, so a direct sum of direct sums is flattened, as [`kron`](@ref) does for the tensor products: `A₁ ⊕ A₂ ⊕ A₃` is the same as `direct_sum(A₁, A₂, A₃)`. A nested direct sum can only be built with the [`SumSpace`](@ref) constructor.

`⊕` (`\oplus` + `<tab>`) is a synonym of `direct_sum`. See also [`direct_component`](@ref) and [`set_direct_component`](@ref).

# Examples

```jldoctest
julia> H = sigmaz() ⊕ qeye(3) ⊕ sigmax();

julia> H.dimensions
Dimensions(SumSpace(Space(2), Space(3), Space(2)), SumSpace(Space(2), Space(3), Space(2)))

julia> ψ = basis(2, 0) ⊕ basis(3, 1);

julia> direct_component(ψ, 2) == basis(3, 1)
true
```
"""
direct_sum(s::AbstractSpace, ss::AbstractSpace...) = SumSpace(_summands(s, ss...)...)

# the summands of the spaces, one level deep (a space that is not a direct sum is its only summand)
_summands() = ()
_summands(s::SumSpace, ss::AbstractSpace...) = (s.spaces..., _summands(ss...)...)
_summands(s::AbstractSpace, ss::AbstractSpace...) = (s, _summands(ss...)...)

const _DirectSumArg{T} = Union{QuantumObject{T}, Number}
const _DirectSumAnyArg = Union{QuantumObject{<:Union{Ket, Bra, Operator, OperatorKet, OperatorBra, SuperOperator}}, Number}

direct_sum(A::Number, B::Number, Cs::Number...) = _direct_sum_block(Operator(), (A, B, Cs...))

# stacked, same as [A; B] and [A B]
direct_sum(A::_DirectSumArg{Ket}, B::_DirectSumArg{Ket}, Cs::_DirectSumArg{Ket}...) = vcat(map(x -> _num_to_Qobj(x, Ket()), (A, B, Cs...))...)
direct_sum(A::_DirectSumArg{OperatorKet}, B::_DirectSumArg{OperatorKet}, Cs::_DirectSumArg{OperatorKet}...) = vcat(map(x -> _num_to_Qobj(x, OperatorKet()), (A, B, Cs...))...)
direct_sum(A::_DirectSumArg{Bra}, B::_DirectSumArg{Bra}, Cs::_DirectSumArg{Bra}...) = hcat(map(x -> _num_to_Qobj(x, Bra()), (A, B, Cs...))...)
direct_sum(A::_DirectSumArg{OperatorBra}, B::_DirectSumArg{OperatorBra}, Cs::_DirectSumArg{OperatorBra}...) = hcat(map(x -> _num_to_Qobj(x, OperatorBra()), (A, B, Cs...))...)

# block diagonal
direct_sum(A::_DirectSumArg{Operator}, B::_DirectSumArg{Operator}, Cs::_DirectSumArg{Operator}...) = _direct_sum_block(Operator(), (A, B, Cs...))
direct_sum(A::_DirectSumArg{SuperOperator}, B::_DirectSumArg{SuperOperator}, Cs::_DirectSumArg{SuperOperator}...) = _direct_sum_block(SuperOperator(), (A, B, Cs...))

direct_sum() = throw(ArgumentError("direct_sum needs at least one argument."))
# a single argument gives a direct sum with a single summand
direct_sum(A::Number) = _direct_sum_block(Operator(), (A,))
direct_sum(A::QuantumObject{<:Union{Ket, OperatorKet}}) = vcat(A)
direct_sum(A::QuantumObject{<:Union{Bra, OperatorBra}}) = hcat(A)
direct_sum(A::QuantumObject{<:Union{Operator, SuperOperator}}) = _direct_sum_block(A.type, (A,))
function direct_sum(A::_DirectSumAnyArg, B::_DirectSumAnyArg, Cs::_DirectSumAnyArg...)
    types = unique(nameof(typeof(x.type)) for x in (A, B, Cs...) if x isa QuantumObject)
    throw(ArgumentError("direct_sum needs Qobjs of the same type (numbers allowed), got: $(join(types, ", "))."))
end

function _direct_sum_block(type, xs)
    qs = map(_num_to_Qobj, xs)
    data = blockdiag(map(q -> sparse(q.data), qs)...)
    to = direct_sum(map(q -> q.dimensions.to, qs)...)
    from = direct_sum(map(q -> q.dimensions.from, qs)...)
    return QuantumObject(data, type, Dimensions(to, from))
end

# a number as a quantum object of dimension 1 (an operator, or a ket or bra of the given type)
_num_to_Qobj(x::Number) = QuantumObject(fill(x, 1, 1))
_num_to_Qobj(x::Number, type) = QuantumObject(type isa Union{Bra, OperatorBra} ? [conj(x)]' : [x], type, Dimensions(Space(1), Space(1))) # bras are adjoint vectors
_num_to_Qobj(A::QuantumObject) = A
_num_to_Qobj(A::QuantumObject, type) = A

@doc raw"""
    direct_sum([A₁, A₂, ...])

Return the direct sum of the elements of a vector, which are stacked as in QuTiP: [`Ket`](@ref)s (and numbers) give a [`Ket`](@ref), [`Bra`](@ref)s give a [`Bra`](@ref), and [`Operator`](@ref)s are vectorized (see [`mat2vec`](@ref)) into an [`OperatorKet`](@ref), or into an [`OperatorBra`](@ref) together with operator-bras. `[A₁; A₂]` and `[A₁ A₂]` stack in the same way.

A vector of rows is a matrix of blocks, see the next method.
"""
function direct_sum(av::AbstractVector)
    isempty(av) && throw(ArgumentError("direct_sum needs at least one element in the list."))
    # a vector of rows, e.g. Any[Any[A, nothing], Any[nothing, B]]
    all(x -> x isa AbstractVector, av) && return direct_sum(collect(AbstractVector, av))
    any(x -> !(x isa _DirectSum1DArg), av) &&
        throw(ArgumentError("Invalid type used in direct_sum list. Only Ket, Bra, Operator, OperatorKet, OperatorBra and numbers allowed."))
    to_bra = any(_is_bra, av)
    any(_is_ket, av) && to_bra && throw(ArgumentError("Cannot combine kets or operator-kets with bras or operator-bras in a direct sum."))

    if to_bra
        type = any(x -> x isa QuantumObject{<:Union{Operator, OperatorBra}}, av) ? OperatorBra() : Bra()
        return hcat(map(x -> _num_to_Qobj(x, type), av)...)
    else # operators alone are vectorized into operator-kets
        type = any(x -> x isa QuantumObject{<:Union{Operator, OperatorKet}}, av) ? OperatorKet() : Ket()
        return vcat(map(x -> _num_to_Qobj(x, type), av)...)
    end
end

const _DirectSum1DArg = Union{QuantumObject{<:Union{Ket, Bra, Operator, OperatorKet, OperatorBra}}, Number}

_is_ket(x) = x isa QuantumObject{<:Union{Ket, OperatorKet}}
_is_bra(x) = x isa QuantumObject{<:Union{Bra, OperatorBra}}

# stacking kets, or operators (vectorized) and operator-kets
function Base.vcat(A::QuantumObject{Ket}, Bs::QuantumObject{Ket}...)
    data = vcat(A.data, map(B -> B.data, Bs)...)
    to = direct_sum(A.dimensions.to, map(B -> B.dimensions.to, Bs)...)::SumSpace # to is always a SumSpace
    return QuantumObject(data, Ket(), Dimensions(to, Space(1)))
end

const _DirectSumVcatArg = QuantumObject{<:Union{Operator, OperatorKet}}
function Base.vcat(A::_DirectSumVcatArg, Bs::_DirectSumVcatArg...)
    vs = map(O -> O isa QuantumObject{Operator} ? mat2vec(O) : O, (A, Bs...))
    to = direct_sum(map(v -> v.dimensions.to, vs)...)
    return QuantumObject(vcat(map(v -> v.data, vs)...), OperatorKet(), Dimensions(to, Space(1)))
end

Base.vcat(::QuantumObject, ::QuantumObject...) = throw(ArgumentError("Invalid combination of quantum object types for direct sum. Only Ket or Operator and OperatorKet allowed."))

# stacking bras, or operators (vectorized) and operator-bras
function Base.hcat(A::QuantumObject{Bra}, Bs::QuantumObject{Bra}...)
    data = hcat(A.data, map(B -> B.data, Bs)...)
    from = direct_sum(A.dimensions.from, map(B -> B.dimensions.from, Bs)...)::SumSpace # from is always a SumSpace
    return QuantumObject(data, Bra(), Dimensions(Space(1), from))
end

const _DirectSumHcatArg = QuantumObject{<:Union{Operator, OperatorBra}}
function Base.hcat(A::_DirectSumHcatArg, Bs::_DirectSumHcatArg...)
    vs = map(O -> O isa QuantumObject{Operator} ? mat2vec(O)' : O, (A, Bs...))
    from = direct_sum(map(v -> v.dimensions.from, vs)...)
    return QuantumObject(hcat(map(v -> v.data, vs)...), OperatorBra(), Dimensions(Space(1), from))
end

Base.hcat(::QuantumObject, ::QuantumObject...) = throw(ArgumentError("Invalid combination of quantum object types for direct sum. Only Bra or Operator and OperatorBra allowed."))

@doc raw"""
    direct_sum(S::SumSpace, k => v, ...)

Return the vector of the [`SumSpace`](@ref) `S` that is equal to the [`Ket`](@ref) (or [`OperatorKet`](@ref)) `v` from the `k`-th component on, and which is zero in the components that are not given.

The space of `v` can be the `k`-th summand of `S`, or the direct sum of some consecutive summands of `S` from the `k`-th on. For example, `v` can be an eigenvector of a block returned by [`block_diagonal_form`](@ref QuantumToolbox.block_diagonal_form).
"""
function direct_sum(S::SumSpace, c1::Pair{Int, <:QuantumObject}, cs::Pair{Int, <:QuantumObject}...) # (at least one component, otherwise `direct_sum(S)` would be the direct sum of a single space)
    components = (c1, cs...)
    data = zeros(promote_type(map(c -> eltype(c.second), components)...), get_size(S))
    for (k, v) in components
        _check_component(S, k, v.dimensions.to)
        offset = first(_summand_range(S, k)) - 1
        data[(offset + 1):(offset + length(v.data))] .= v.data
    end
    return QuantumObject(data, last(first(components)).type, Dimensions(S, Space(1)))
end

function _check_component(S::SumSpace, k::Int, s::AbstractSpace)
    (S.spaces[k] == s) && return nothing

    # a direct sum of consecutive summands of S
    run = _summands(s)
    r = k:(k + length(run) - 1)
    ((last(r) <= length(S.spaces)) && all(i -> S.spaces[r[i]] == run[i], eachindex(run))) ||
        throw(DimensionMismatch("The dims of the component $k do not match the summands of the SumSpace."))
    return nothing
end

@doc raw"""
    direct_sum(blocks::AbstractMatrix)
    direct_sum([[A₁₁, A₁₂], [A₂₁, A₂₂]])

Return the [`Operator`](@ref) (or [`SuperOperator`](@ref)) made of a matrix of blocks, with at least 2 rows and 2 columns, where `nothing` stands for a zero block, e.g. `direct_sum([A B; nothing D])`. As in QuTiP, the matrix can also be given as a vector of rows, and the blocks can also be [`Ket`](@ref)s, [`Bra`](@ref)s and numbers (or [`OperatorKet`](@ref)s and [`OperatorBra`](@ref)s together with [`SuperOperator`](@ref)s).

The blocks in the same row must have the same `to` space, and the blocks in the same column the same `from` space, so each row and each column needs at least one block (otherwise, see the method with `(i, j) => A` below). The result acts from the direct sum of the `from` spaces of the columns to the direct sum of the `to` spaces of the rows, and `direct_component(result, i, j)` is the block `(i, j)`.

# Examples

```jldoctest
julia> a = destroy(4);

julia> direct_sum([qeye(2) nothing; nothing a]) == qeye(2) ⊕ a
true

julia> B = direct_sum([sigmax() basis(2, 0) * basis(4, 1)'; nothing a]);

julia> direct_component(B, 1, 2) == basis(2, 0) * basis(4, 1)'
true
```
"""
function direct_sum(blocks::AbstractMatrix)
    (size(blocks, 1) >= 2 && size(blocks, 2) >= 2) || throw(ArgumentError("A 2D direct sum needs at least 2 rows and 2 columns of blocks. For a single row or column, use direct_sum([...]) or ⊕."))
    # all blocks Hilbert-like (kets, bras, operators) or all Liouville-like (operator-kets, operator-bras, super-operators)
    liouville = all(b -> b isa _DirectSum2DLiouvilleArg, blocks) && any(b -> b isa QuantumObject, blocks)
    (liouville || all(b -> b isa _DirectSum2DHilbertArg, blocks)) || throw(ArgumentError("Invalid combination of Qobj types for direct sum."))

    to = [_line_space(blocks[i, :], :to, "row $i") for i in axes(blocks, 1)]
    from = [_line_space(blocks[:, j], :from, "column $j") for j in axes(blocks, 2)]
    S_to, S_from = direct_sum(to...), direct_sum(from...)
    # operator-kets/-bras and numbers have a plain Space(1) side: a side made only of those is not a Liouville space
    liouville && !(_has_liouville(S_to) && _has_liouville(S_from)) && throw(ArgumentError("Operator with both space and superspace dimensions are not supported. Please open an issue if you have an use case for these."))

    # sparse zero blocks instead of `nothing`, then glue the blocks together
    mats = [isnothing(blocks[i, j]) ? spzeros(Bool, get_size(to[i]), get_size(from[j])) : sparse(_block_matrix(blocks[i, j])) for i in axes(blocks, 1), j in axes(blocks, 2)]
    return QuantumObject(hvcat(size(mats, 2), permutedims(mats)...), liouville ? SuperOperator() : Operator(), Dimensions(S_to, S_from))
end

function direct_sum(rows::AbstractVector{<:AbstractVector})
    allequal(map(length, rows)) || throw(ArgumentError("The rows of a 2D direct sum must have the same length."))
    return direct_sum(permutedims(reduce(hcat, rows)))
end

const _DirectSum2DHilbertArg = Union{Nothing, Number, QuantumObject{<:Union{Ket, Bra, Operator}}}
const _DirectSum2DLiouvilleArg = Union{Nothing, Number, QuantumObject{<:Union{OperatorKet, OperatorBra, SuperOperator}}}

# the `to` (or `from`) space of the blocks in a row (or column), which must be the same
function _line_space(line, side::Symbol, name::String)
    spaces = [getproperty(_num_to_Qobj(b).dimensions, side) for b in line if !isnothing(b)]
    isempty(spaces) && throw(ArgumentError("Direct sum: empty $name."))
    k = findfirst(!=(first(spaces)), spaces)
    isnothing(k) || throw(ArgumentError("Direct sum: inconsistent dimensions in $name. Expected $(first(spaces)), got $(spaces[k])."))
    return first(spaces)
end

_has_liouville(::LiouvilleSpace) = true
_has_liouville(s::SumSpace) = any(_has_liouville, s.spaces)
_has_liouville(::AbstractSpace) = false

_block_matrix(b::Number) = fill(b, 1, 1)
_block_matrix(b::QuantumObject) = b.data isa AbstractVector ? reshape(b.data, :, 1) : b.data

@doc raw"""
    direct_sum(S_to::SumSpace, S_from::SumSpace, (i, j) => A, ...)

Return the [`Operator`](@ref) (or [`SuperOperator`](@ref)) from `S_from` to `S_to` whose block `(i, j)` is `A`, and which is zero in the blocks that are not given. As for the vectors, the dims of `A` can also span several consecutive summands of `S_to` and `S_from`, starting from the `i`-th and the `j`-th.
"""
function direct_sum(S_to::SumSpace, S_from::SumSpace, b1::Pair{<:NTuple{2, Int}, <:QuantumObject}, bs::Pair{<:NTuple{2, Int}, <:QuantumObject}...) # (at least one block, otherwise `direct_sum(S_to, S_from)` would be the direct sum of two spaces)
    blocks = (b1, bs...)
    I, J, V = Int[], Int[], promote_type(map(b -> eltype(b.second), blocks)...)[]
    for ((i, j), A) in blocks
        _check_component(S_to, i, A.dimensions.to)
        _check_component(S_from, j, A.dimensions.from)
        Ai, Aj, Av = findnz(sparse(A.data))
        append!(I, Ai .+ (first(_summand_range(S_to, i)) - 1))
        append!(J, Aj .+ (first(_summand_range(S_from, j)) - 1))
        append!(V, Av)
    end
    type = last(first(blocks)).type
    all(b -> b.second.type == type, blocks) || throw(ArgumentError("The blocks must have the same type."))
    return QuantumObject(sparse(I, J, V, get_size(S_to), get_size(S_from)), type, Dimensions(S_to, S_from))
end

const _SumIndex = Union{Integer, AbstractUnitRange{<:Integer}}
const _SumBra = Union{QuantumObject{Bra, <:Dimensions{Space, <:SumSpace}}, QuantumObject{OperatorBra, <:Dimensions{Space, <:SumSpace}}}

@doc raw"""
    direct_component(v::QuantumObject, k)
    direct_component(A::QuantumObject, i, j)

Return the `k`-th component of a [`Ket`](@ref), [`Bra`](@ref), [`OperatorKet`](@ref) or [`OperatorBra`](@ref) `v` of a [`SumSpace`](@ref), or the block `(i, j)` of an [`Operator`](@ref) (or [`SuperOperator`](@ref)) `A` between two [`SumSpace`](@ref)s. The result has the dims of the corresponding summands.

The indices can also be ranges of consecutive summands, e.g. `direct_component(A, 1:2, 1:2)`, in which case the result acts on the direct sum of these summands.

See also [`direct_sum`](@ref) and [`set_direct_component`](@ref).
"""
function direct_component(v::QuantumObject{<:Union{Ket, OperatorKet}, <:Dimensions{<:SumSpace}}, k::_SumIndex)
    S = v.dimensions.to
    return QuantumObject(v.data[_summand_range(S, k)], v.type, Dimensions(_summand_space(S, k), Space(1)))
end
direct_component(v::_SumBra, k::_SumIndex) = direct_component(v', k)'
function direct_component(A::QuantumObject{<:Union{Operator, SuperOperator}, <:Dimensions{<:SumSpace, <:SumSpace}}, i::_SumIndex, j::_SumIndex)
    to, from = A.dimensions.to, A.dimensions.from
    data = A.data[_summand_range(to, i), _summand_range(from, j)]
    return QuantumObject(data, A.type, Dimensions(_summand_space(to, i), _summand_space(from, j)))
end
direct_component(::QuantumObject, idx...) =
    throw(ArgumentError("direct_component needs a direct sum, with one index (integer or range) for kets and bras, and two for operators."))

# the space of the k-th summand, or the direct sum of the summands in the range k
_summand_space(S::SumSpace, k::Integer) = S.spaces[k]
_summand_space(S::SumSpace, k::AbstractUnitRange{<:Integer}) = SumSpace(S.spaces[k])

@doc raw"""
    set_direct_component(v::QuantumObject, c, k)
    set_direct_component(A::QuantumObject, c, i, j)
    set_direct_component(v::QuantumObject, c1, k1, c2, k2, ...)
    set_direct_component(A::QuantumObject, c1, i1, j1, c2, i2, j2, ...)

Return a copy of `v` where the `k`-th component is replaced by `c`, or a copy of `A` where the block `(i, j)` is replaced by `c`. If `c` is `nothing`, that component (or block) is set to zero. `c` must have the dims of the component (or block), and the indices can also be ranges of consecutive summands. Several components (or blocks) can be set in one call by repeating the component and its indices.

See also [`direct_component`](@ref).

# Examples

```jldoctest
julia> set_direct_component(basis(2, 0) ⊕ basis(3, 1), 2 * basis(3, 1), 2) == basis(2, 0) ⊕ (2 * basis(3, 1))
true

julia> set_direct_component(sigmax() ⊕ num(3) ⊕ sigmaz(), nothing, 1, 1, sigmax(), 3, 3) == (0 * sigmax()) ⊕ num(3) ⊕ sigmax()
true
```
"""
function set_direct_component(v::QuantumObject{<:Union{Ket, OperatorKet}, <:Dimensions{<:SumSpace}}, c::Union{Nothing, QuantumObject}, k::_SumIndex)
    S = v.dimensions.to
    r = _summand_range(S, k) # also checks k
    _check_component_dims(c, Dimensions(_summand_space(S, k), Space(1)))
    data = _promoted_copy(v.data, c)
    data[r] .= isnothing(c) ? 0 : c.data
    return QuantumObject(data, v.type, v.dimensions)
end
set_direct_component(v::_SumBra, c::Union{Nothing, QuantumObject}, k::_SumIndex) =
    set_direct_component(v', isnothing(c) ? c : c', k)'
function set_direct_component(A::QuantumObject{<:Union{Operator, SuperOperator}, <:Dimensions{<:SumSpace, <:SumSpace}}, c::Union{Nothing, QuantumObject}, i::_SumIndex, j::_SumIndex)
    to, from = A.dimensions.to, A.dimensions.from
    r1, r2 = _summand_range(to, i), _summand_range(from, j) # also checks i and j
    _check_component_dims(c, Dimensions(_summand_space(to, i), _summand_space(from, j)))
    data = _promoted_copy(A.data, c)
    data[r1, r2] .= isnothing(c) ? 0 : c.data
    return QuantumObject(data, A.type, A.dimensions)
end
# several components in one call: repeat (c, k) for kets and bras, (c, i, j) for operators
set_direct_component(v::QuantumObject{<:Union{Ket, OperatorKet, Bra, OperatorBra}}, c, k, c2, k2, rest...) =
    set_direct_component(set_direct_component(v, c, k), c2, k2, rest...)
set_direct_component(A::QuantumObject{<:Union{Operator, SuperOperator}}, c, i, j, c2, i2, j2, rest...) =
    set_direct_component(set_direct_component(A, c, i, j), c2, i2, j2, rest...)
set_direct_component(::QuantumObject, c, idx...) =
    throw(ArgumentError("set_direct_component needs a direct sum, with one index (integer or range) for kets and bras, and two for operators."))

_check_component_dims(::Nothing, ::Dimensions) = nothing
_check_component_dims(c::QuantumObject, D::Dimensions) =
    (c.dimensions == D) ? nothing : throw(DimensionMismatch("The component has $(c.dimensions), but the summand has $D."))

# a copy of the data, with an element type that can also hold the elements of c (e.g. real data, complex c)
_promoted_copy(data, c) = copyto!(similar(data, promote_type(eltype(data), isnothing(c) ? Bool : eltype(c))), data)
# structured matrices (e.g. Diagonal or Hermitian) cannot hold every block, so they are copied into a sparse matrix
_promoted_copy(data::Union{Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal, Symmetric, Hermitian, LinearAlgebra.AbstractTriangular}, c) = _promoted_copy(sparse(data), c)