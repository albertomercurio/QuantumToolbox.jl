"""
    QuantumToolboxOperators

Matrix-free quantum operators for [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl).

Operators store the *rule* for acting on a state rather than its matrix elements. Each one is a
`SciMLOperators.AbstractSciMLOperator` implementing `mul!`, so it works throughout the SciML
ecosystem. See `README.md` for a quick start and `STATUS.md` for the current integration status.
"""
module QuantumToolboxOperators

using LinearAlgebra
using SparseArrays

import SciMLOperators
import SciMLOperators: AbstractSciMLOperator, AdjointOperator, IdentityOperator, ScaledOperator
import SciMLOperators: islinear, has_adjoint, concretize, cache_operator, iscached

export BosonicOperator, DestroyOperator, NumberOperator, DestroyPowerOperator, NormalOrderedOperator
export LocalTensorProductOperator

include("bosonic.jl")
include("tensor_product.jl")

end
