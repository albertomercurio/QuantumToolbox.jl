# QuantumToolboxOperators.jl

Matrix-free quantum operators for [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl).

`QuantumToolbox.jl` stores operators as sparse matrices, which stops scaling once the Hilbert space
gets large — a system of `M` modes truncated at `d` levels has dimension `dᴹ`. But solvers only ever
need the *action* `Ô|ψ⟩`, never the matrix elements. This package stores the **rule** instead of the
matrix: each operator is a `SciMLOperators.AbstractSciMLOperator` implementing `mul!`, so it plugs
straight into the SciML ecosystem.

> **Experimental.** Unregistered, API subject to change, and not yet wired into CI. See
> [STATUS.md](STATUS.md) for what works with `QuantumToolbox.jl` today and what does not.

## Installation

```julia
import Pkg
Pkg.develop(path = "lib/QuantumToolboxOperators")
```

## Quick start

```julia
using QuantumToolboxOperators, SciMLOperators, LinearAlgebra

N = 1_000_000
a = DestroyOperator{ComplexF32}(N)

# Products simplify at *construction* time, via multiple dispatch —
# `a' * a` is a single diagonal pass, not a two-pass ComposedOperator.
typeof(a' * a)          # NumberOperator{ComplexF32}
typeof(a'^2 * a^2)      # NormalOrderedOperator{ComplexF32}

# A single-photon-driven Kerr Hamiltonian
Δ, U, F = 0.1f0, 0.2f0, 0.3f0
H = Δ * a' * a + U * (a'^2 * a^2) + F * (a + a')

ψ = normalize(randn(ComplexF32, N))
mul!(similar(ψ), H, ψ)

Base.summarysize(H)                 # 112 bytes …
Base.summarysize(concretize(H))     # … versus 53.4 MiB as a sparse matrix
```

## What's provided

| | |
|---|---|
| `DestroyOperator{T}(N)` | annihilation `â`; `a'` gives `â†` |
| `NumberOperator{T}(N; shift)` | `n̂ + shift` |
| `DestroyPowerOperator{T}(N, k)` | `â^k` |
| `NormalOrderedOperator{T}(N, k, n)` | `(â†)^k âⁿ` |
| `LocalTensorProductOperator(dims, j => A, …)` | `1 ⊗ … ⊗ Aⱼ ⊗ … ⊗ 1`, without materializing or traversing the identity factors |

Products are rewritten as they are built:

| Expression | Becomes |
|---|---|
| `a' * a` | `NumberOperator` |
| `a * a'` | `NumberOperator(shift = 1)` |
| `a * a`, `a^k` | `DestroyPowerOperator` |
| `a' * a'`, `(a')^k` | `adjoint(DestroyPowerOperator)` |
| `(a')^k * a^n` | `NormalOrderedOperator` |
| `(λ * a') * a` | `λ * NumberOperator` |

Anything without a rule falls back to SciMLOperators' generic lazy types.

## Using it with QuantumToolbox.jl

A matrix-free operator cannot be a `QuantumObject` (whose data must be an `AbstractArray`), so wrap
it in a `QobjEvo`:

```julia
using QuantumToolbox

N = 200
a = DestroyOperator{ComplexF64}(N)
H = QobjEvo(0.1 * a' * a + 0.2 * (a'^2 * a^2); type = Operator(), dims = N)

sol = sesolve(H, fock(N, 1), 0.0:0.1:1.0)
```

`sesolve`, `mesolve` without collapse operators, and `expect` work today. `mesolve` *with* `c_ops`,
the stochastic solvers, `steadystate`, `spectrum` and `eigenstates` do not — all for the same
reason, which [STATUS.md](STATUS.md) explains.

## Performance at a glance

Memory savings are large and unconditional: a single-mode operator is a handful of bytes against
tens of megabytes sparse. Simple operators (`â`, `â†`, `n̂`) are also ~1.7–1.9× *faster* than sparse
on both CPU and GPU. Composite operators are not yet: the Kerr Hamiltonian above is 3× slower than
sparse on CPU, and multi-mode tensor products are slower still — though under Reactant/XLA they are
the fastest option measured, and `SciMLOperators.TensorProductOperator` cannot compile there at all.

Numbers, method, and what is being worked on: [STATUS.md](STATUS.md).

## Limitations

  * Bosonic single-mode operators only — no spin/qubit types (use `MatrixOperator`).
  * `LocalTensorProductOperator` acts on `AbstractVector`, not `AbstractMatrix`.
  * No lazy superoperators, which is what limits the open-system solvers.
  * A cached operator is not thread-safe; its work buffers are shared.

## Tests

```bash
julia --project=lib/QuantumToolboxOperators -e 'import Pkg; Pkg.test()'
```
