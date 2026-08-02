# QuantumToolboxOperators — status

What works, what does not, and what the numbers actually say. Written 2026-08-02.

For a quick start see [README.md](README.md). The performance figures in §4 are a point-in-time
measurement, recorded here so the document stands on its own; §4.1 gives the hardware and method.

---

## 1. TL;DR

| | |
|---|---|
| **Works today** | `sesolve`, `mesolve` without `c_ops`, `expect`/`dot`, CPU + CUDA + Reactant/XLA, tensor products via `LocalTensorProductOperator` |
| **Works, with caveats** | Composite single-mode operators are still slower than sparse on CPU; multi-mode operators are slower than sparse on CPU and much slower on CUDA — but far smaller in memory, and the fastest of all under XLA |
| **Blocked** | `mesolve` with `c_ops`, all stochastic solvers, `steadystate` (except the ODE solver), `spectrum`, `eigenstates` — every one of them for the same underlying reason: there is no lazy superoperator |

The single highest-leverage missing piece is a reshape-based `_spre`/`_spost` — see §3.3 for
exactly what it would unblock.

---

## 2. What is implemented

### 2.1 Bosonic operators

All store only a dimension (and a power); none allocate matrix data.

| Type | Operator | 3-arg `mul!` | 5-arg `mul!` | `adjoint` | `concretize` | `AbstractMatrix` action |
|---|---|---|---|---|---|---|
| `DestroyOperator{T}(N)` | ``â`` | ✅ | ✅ | `AdjointOperator` | ✅ | ✅ |
| `NumberOperator{T}(N; shift)` | ``n̂ + \text{shift}`` | ✅ | ✅ | itself (Hermitian) | ✅ | ✅ |
| `DestroyPowerOperator{T}(N, k)` | ``â^k`` | ✅ | ✅ | `AdjointOperator` | ✅ | ✅ |
| `NormalOrderedOperator{T}(N, k, n)` | ``(â†)^k â^n`` | ✅ | ✅ | swaps `k`, `n` | ✅ | ✅ |

Creation operators are `SciMLOperators.AdjointOperator` wrappers — there is deliberately no
separate `CreateOperator` type.

### 2.2 Algebraic simplification

These fire at *construction* time, so they cost nothing per `mul!`:

| Expression | Result |
|---|---|
| `a' * a` | `NumberOperator` (``n̂``) |
| `a * a'` | `NumberOperator(shift = 1)` |
| `a * a`, `a^k` | `DestroyPowerOperator` |
| `a' * a'`, `(a')^k` | `adjoint(DestroyPowerOperator)` |
| `(a')^k * a^n` | `NormalOrderedOperator` |
| `(λ * a') * a` | `λ * NumberOperator` — the scalar is peeled first |
| `a * (λ * a')` | `λ * NumberOperator(shift = 1)` — likewise from the right |

The two `ScaledOperator` unwrap rules are easy to miss but are what make a physical Hamiltonian
like `Δ * a' * a + U * (a'^2 * a^2)` simplify at all. Anything without a rule (e.g. `a + a'`)
falls back to SciMLOperators' generic lazy types — correct, just slower.

### 2.3 `LocalTensorProductOperator`

Applies ``\mathbb{1} ⊗ ⋯ ⊗ A_j ⊗ ⋯ ⊗ \mathbb{1}`` one subsystem at a time by reshaping the state,
`permutedims!`-ing the target axis to the front, `mul!`-ing, and permuting back — ping-ponging
between two cached buffers so no allocation happens per application.

**Memory layout.** The state is read as `reshape(v, reverse(dims)...)`, so physics mode `j`
(1 = leftmost in the Kronecker product) is Julia array dimension `N - j + 1`. Mode `N` is therefore
contiguous and needs no permutation at all; mode 1 is the most strided. This one fact explains
every branch in `mul!`.

**Caching.** `cache_operator` is required except for the one configuration that genuinely needs no
buffers: a single operator acting on the last subsystem. `mul!` throws a clear `ArgumentError`
otherwise.

At least one operator is required. An identity on the whole space is
`SciMLOperators.IdentityOperator(prod(dims))`, so there is no reason for this type to represent it.

### 2.4 Why not `SciMLOperators.TensorProductOperator`?

Measured, not asserted (18-spin XX chain, `ComplexF32`, dimension 262 144):

| | `TensorProductOperator` | `LocalTensorProductOperator` | sparse |
|---|---|---|---|
| CPU time | 36.94 ms | 42.74 ms | 2.65 ms |
| Cached memory | **264.0 MiB** | 68.0 MiB (4.0 MiB shared) | 70.0 MiB |
| XLA | **fails to compile** | 604.78 µs | — |

Two things matter here. Its cache is ~4× larger than ours and ~66× larger than ours with a shared
cache, because it allocates buffers at every nesting level while we allocate exactly two. And it
cannot be traced by Reactant at all — `Scalar indexing is disallowed` — whereas
`LocalTensorProductOperator` compiles and is then the fastest option measured for this problem.
Raw CPU throughput is a wash.

### 2.5 Coverage versus the workshop notebook

Everything the "Matrix Free Operations" notebook demonstrated is implemented: all four operator
types, every simplification rule including the `ScaledOperator` unwraps, `concretize`, the Kerr
Hamiltonian example, CUDA, and Reactant. The package has since added `NormalOrderedOperator` and
`LocalTensorProductOperator`. The notebook's `bosonic_operators.jl` is now `src/bosonic.jl`.

---

## 3. Integration with QuantumToolbox.jl

### 3.1 Why `QuantumObjectEvolution`, not `QuantumObject`

`QuantumObject` constrains `DataType <: AbstractArray`
(`src/qobj/quantum_object.jl:46,52`), so a matrix-free operator **cannot** be one.
`QuantumObjectEvolution` constrains `DataType <: AbstractSciMLOperator`
(`src/qobj/quantum_object_evo.jl:111-131`) and `QobjEvo(data::AbstractSciMLOperator; type, dims)`
(`:159-173`) passes it through untouched. That is the entry point:

```julia
H = QobjEvo(0.1 * a' * a + 0.2 * (a'^2 * a^2); type = Operator(), dims = N)
```

### 3.2 Status by feature

Rows marked **verified** were executed against this build; the rest are from reading the source.

| Feature | Status | Evidence |
|---|---|---|
| `sesolve` | ✅ works | **verified** — agrees with the sparse Hamiltonian to `max abs 1.4e-17`; `sesolve.jl:82` hands the cached operator to `ODEProblem` |
| `mesolve`, no `c_ops` | ✅ works | **verified** — `mesolve.jl:113` |
| `expect(QobjEvo, ψ)`, `dot(ψ, L, ψ)` | ✅ works | **verified** — needed the allocating `L * v` added in this change (see §5.1) |
| CUDA | ✅ works | **verified** — bosonic operators are array-agnostic; see §4 |
| Reactant / XLA | ✅ works | **verified** — `mul!` traces and compiles for both bosonic and `LocalTensorProductOperator` |
| `mesolve` + `c_ops` | ❌ blocked | **verified** — `DimensionMismatch: parent has 400 elements, which is incompatible with length 20`. `liouvillian` has no lazy path, so `_spre` falls through to `kron(Eye, A)` (`src/qobj/superoperators.jl:23-27`) and the inner operator then receives the vectorized density matrix at the wrong shape |
| `steadystate` (non-ODE) | ❌ blocked | **verified** — `ArgumentError: SteadyStateDirectSolver does not support QobjEvo.` (`src/steadystate.jl:237-242`) |
| `eigenstates` / `eigen` | ❌ blocked | **verified** — `MethodError`; these are `::QuantumObject`-typed and call `to_dense` (`src/qobj/eigsolve.jl:558-568,621-633`) |
| `ssesolve` / `smesolve` | ❌ blocked | `src/time_evolution/ssesolve.jl:5-8`, `smesolve.jl:108` hardcode `op.A`, the `MatrixOperator` field (there is a `TODO` in the source saying as much) |
| `spectrum` / correlations | ❌ blocked | `src/spectrum.jl:76` explicitly rejects a non-`QuantumObject` Liouvillian |
| `e_ops` in `mesolve` | ❌ blocked | `mesolve_callback_helpers.jl:45` needs `mat2vec(adjoint(get_data(op)))`, i.e. a concrete matrix |
| `cu(::QuantumObjectEvolution)` | ❌ absent | `ext/QuantumToolboxCUDAExt.jl` is `::QuantumObject`-typed throughout |
| Reactant extension in QuantumToolbox | ❌ absent | no Reactant code anywhere in the main package |

### 3.3 The one unlock

`src/qobj/superoperators.jl` has no reshape-based `_spre`/`_spost`. Its `AbstractSciMLOperator`
fallback materializes `kron(Eye, A)` and emits a lazy-tensor performance warning. A superoperator
that instead acted as ``ρ ↦ Aρ`` by reshaping the vectorized density matrix would unblock three
things at once:

  * `mesolve` with `c_ops` — directly;
  * `steadystate(…, SteadyStateODESolver())` — it runs through `mesolve`;
  * the `Lanczos` spectrum solver — it needs only `mul!` on the Liouvillian plus a steady state.

The rest of the blocked rows need their own work and would *not* follow: `ssesolve`/`smesolve`
hardcode the `MatrixOperator` field, `mesolve`'s `e_ops` need a concrete `mat2vec`, and
`eigenstates`/`spectrum`'s other solvers are `::QuantumObject`-typed or need a factorization.

---

## 4. Benchmarks

### 4.1 Method

Measured 2026-08-02 on an NVIDIA RTX 4090 with Julia 1.12.6, SciMLOperators 1.25.2, CUDA 5.10.0 and
Reactant 0.2.278. `ComplexF32` throughout, `BLAS.set_num_threads(1)`. Times are the **minimum** over
Chairmarks samples; every case was checked against a sparse reference before being recorded.

Two things about how these were taken:

  * **Every GPU measurement is wrapped in `CUDA.@sync`.** CUDA kernel launches are asynchronous, so
    an unwrapped `@be mul!(dψ, a, ψ)` on device arrays times the *launch*, not the work. The
    workshop notebook's GPU figures were not synchronized, which is why they showed a ~10-kernel
    lazy Hamiltonian "losing" to a 1-kernel sparse one by exactly the ratio of kernel counts. Those
    numbers are **not** comparable with the ones below.
  * `Base.summarysize` on a device-backed operator counts the host-side wrapper only, not device
    memory. GPU memory figures are therefore not reported.

### 4.2 Single mode, N = 10⁶

| Case | CPU lazy | CPU sparse | CUDA lazy | CUDA sparse | XLA lazy |
|---|---|---|---|---|---|
| `â` | **682 µs** | 1.26 ms | **14.0 µs** | 23.7 µs | **9.6 µs** |
| `â†` | **675 µs** | 1.26 ms | **14.6 µs** | 23.7 µs | — |
| `n̂ = â†â` | **761 µs** | 1.26 ms | **13.7 µs** | 23.8 µs | — |
| `â²` | 8.09 ms | **1.26 ms** | **23.0 µs** | 24.1 µs | — |
| `(â†)²â²` | 2.55 ms | **1.26 ms** | **18.9 µs** | 23.8 µs | — |
| `Kerr H` | 7.80 ms | **2.54 ms** | 63.4 µs | **54.7 µs** | **13.3 µs** |

XLA compile cost: 16.3 s for `â`, 8.3 s for `Kerr H` — one-off, and excluded from the times above.

**Run-to-run stability.** Two full runs agreed within ~2% on every CPU and CUDA figure. The XLA
figures did not: `Kerr H` measured 19.6 µs and 13.3 µs across the two runs, a 32% spread. Treat the
XLA column as indicative of magnitude, not precise. The qualitative conclusion (XLA beats plain
CUDA here) holds comfortably in both runs.

### 4.3 Memory, single mode (N = 10⁶)

| Case | matrix-free | sparse | ratio |
|---|---|---|---|
| `â` | 8 B | 19.1 MiB | 2.5 × 10⁶ |
| `n̂`, `â²`, `(â†)²â²` | 16 B | 31.7 MiB | 2.1 × 10⁶ |
| `Kerr H` | 112 B | 53.4 MiB | 5.0 × 10⁵ |

Caching is a no-op for single-mode operators, so these are the whole cost.

### 4.4 Multi-mode chains

`H = Σₙ Aₙ Bₙ₊₁`, nearest neighbor.

| Case | Variant | CPU | CUDA | XLA |
|---|---|---|---|---|
| 18 spins (dim 262 144) | sparse | **2.65 ms** | **52.2 µs** | — |
| | `TensorProductOperator` | 36.94 ms | — | fails to compile |
| | `LocalTensorProduct` | 42.74 ms | 2.82 ms | **605 µs** |
| | `LocalTensorProduct`, shared cache | 41.45 ms | 2.80 ms | — |
| 4 cavities, d = 30 (dim 810 000) | sparse | **2.03 ms** | **31.2 µs** | — |
| | `TensorProductOperator` | 14.93 ms | — | fails to compile |
| | `LocalTensorProduct` | 12.21 ms | 488 µs | **16.4 µs** |
| | `LocalTensorProduct`, shared cache | 11.39 ms | 488 µs | — |

Memory:

| Case | Variant | operator | cached |
|---|---|---|---|
| 18 spins | sparse | 70.0 MiB | — |
| | `TensorProductOperator` | 608 B | 264.0 MiB |
| | `LocalTensorProduct` | 3.0 KiB | 68.0 MiB |
| | `LocalTensorProduct`, shared cache | 3.0 KiB | **4.0 MiB** |
| 4 cavities | sparse | 40.8 MiB | — |
| | `TensorProductOperator` | 80 B | 123.6 MiB |
| | `LocalTensorProduct` | 192 B | 37.1 MiB |
| | `LocalTensorProduct`, shared cache | 192 B | **12.4 MiB** |

### 4.5 Reading the numbers

**Memory is an unconditional and very large win for single-mode operators** — six orders of
magnitude, and it is the entire reason the approach exists. For multi-mode operators the honest
figure is far smaller: a cached `LocalTensorProductOperator` holds two full state vectors, so it is
~1× sparse until the terms share a cache, at which point it is 3–17× smaller (17× for the spin
chain, 3× for the cavity chain). Quoting the
single-mode ratio for a tensor-product Hamiltonian would be misleading.

**Simple single-mode operators are faster than sparse** on both CPU (~1.7–1.9×) and CUDA (~1.7×).
That is the expected result: one pass over the state with computed coefficients beats one pass plus
an index-array gather.

**Powers are slower, and the reason is now localized.** `â²` costs 8.09 ms against sparse's 1.26 ms,
while `(â†)²â²` costs only 2.55 ms — the *more complicated* operator is 3× faster than the simpler
one. The difference is a single `sqrt`: this change added a `k == n` fast path to
`_normal_ordered_coeff`, where the two square roots cancel exactly, but `â^k` still evaluates
`sqrt` per element inside a loop whose length is a struct field rather than a type parameter, so it
neither unrolls nor vectorizes. Making `k` a type parameter is the obvious next step (§6, P1).

**The Kerr Hamiltonian is now 3.1× slower than sparse on CPU, down from 9.8× before this change**
(the notebook recorded 27.2 ms against 2.78 ms at the same size — a 3.5× improvement). On CUDA it
is 1.16× slower, not the 4.2× the notebook reported; that gap was the missing `CUDA.@sync`.

**XLA is the surprise.** Under Reactant the Kerr Hamiltonian runs in 13.3 µs against CUDA's 63.4 µs
— roughly 5× *faster*, where the notebook reported XLA being 2.4× slower. Same correction: the CUDA
baseline it was compared against had not been synchronized. For the multi-mode chains XLA is
5–30× faster than plain CUDA, and it is the only backend on which
`SciMLOperators.TensorProductOperator` cannot run at all.

**Multi-mode chains remain slower than sparse on CPU (~6–16×) and much slower on CUDA (~16–54×).**
The `permutedims!` round trip dominates, and it is especially costly on the GPU. Matrix-free
tensor products are currently a memory play and an XLA play, not a CPU/CUDA throughput play.

---

## 5. Known defects and limitations

### 5.1 Fixed in this change

  * `LocalTensorProductOperator` with a single operator on the **last** subsystem crashed with
    `MethodError: getindex(::Nothing, ::Int64)` in both 3-arg and 5-arg `mul!` — which meant the
    idiomatic `sum(n -> LocalTensorProductOperator(dims, n => op), 1:N)` crashed on its last term,
    since `AddedOperator` drives every term after the first with `mul!(w, op, v, true, true)`.
  * The test suite did not run at all: it referenced `KroneckerOperator` from a file the module
    never included.
  * `concretize` on any *composite* built from these operators threw a `MethodError`, because
    SciMLOperators reaches composite concretization through `convert(AbstractMatrix, …)` rather
    than `concretize`.
  * `L * v` did not exist, so `LinearAlgebra.dot(ψ, L, ψ)` and hence `QuantumToolbox.expect` failed.
    SciMLOperators defines `*(L, v)` once per concrete type, not generically.
  * The package was unresolvable on Julia 1.10/1.11 (stdlib compat bounds pinned to `1.12.0`).
  * `size(L, d)` was wrong or threw for `d ≥ 3`.
  * `isconstant` was unconditionally `true` for `LocalTensorProductOperator` because it exposed no
    `getops`, which would silently change ODE solver stepping for a time-dependent sub-operator.

### 5.2 Still open

  * **`ââ†` disagrees with the truncated sparse product at the cutoff.** `a * a'` simplifies to
    `NumberOperator(shift = 1)` = `diag(1, …, N)`, the *untruncated* ``n̂ + 1``. The sparse product
    gives `diag(1, …, N-1, 0)`, because ``â†`` maps the top Fock state out of the retained space.
    They agree everywhere else. This is pinned by a test so it cannot drift silently, but it means
    results differ from QuTiP/QuantumToolbox for states with population at the cutoff — where the
    truncation is already suspect.
  * **Sub-operators are not adapted to the state's array type.** `cache_operator` moves the work
    buffers to the device, but a `MatrixOperator` wrapping a host `Matrix` stays on the host and
    `mul!` fails on a GPU state. Build such sub-operators with device arrays directly. Bosonic
    operators are immune — they store only a dimension.
  * **`LocalTensorProductOperator` acts on `AbstractVector` only**, not `AbstractMatrix`.
  * **An `M`-term sum allocates `2M` state vectors.** `AddedOperator` evaluates its terms strictly
    one at a time, so one buffer pair would do. Rebuilding the terms by hand against a single shared
    cache is worth up to ~17× memory (§4.4), but it is not thread-safe and is not something a user
    would discover.
  * **A cached operator is not thread-safe** — concurrent `mul!` calls share the same buffers.
  * **`β = 0` does not clear `NaN`.** The 5-arg `mul!` methods use `lmul!(β, w)`, so a `NaN` already
    in `w` survives `β = 0`. This matches `SciMLOperators.AddedOperator` and is an ecosystem-wide
    convention rather than a local bug, but it does depart from the strict `LinearAlgebra.mul!`
    contract.
  * **No spin/qubit operator types.** Pauli operators must go through `MatrixOperator`.
  * **The subpackage is not wired into CI.** See P3 below.

---

## 6. Roadmap

**P0 — correctness (done in this change).** Everything in §5.1.

**P1 — performance, ordered by measured payoff.**

  1. Make `k`/`n` type parameters of `DestroyPowerOperator`/`NormalOrderedOperator` so
     `_rising_product`'s loop unrolls and the surrounding broadcast can vectorise. §4.5 shows this
     is worth roughly 3× on `â²`, and `â²`-class terms dominate the Kerr Hamiltonian.
  2. Share scratch buffers across `AddedOperator` terms — worth ~17× memory on the chains measured.
  3. Single-pass 3-arg `mul!`: write the zero tail explicitly instead of `fill!` followed by a
     partial assignment.
  4. Reduce the `permutedims!` round trip in `LocalTensorProductOperator`, which is what makes the
     multi-mode CUDA numbers 17–52× worse than sparse.

Note that coefficient *precomputation* was tried and reverted before (commits `953e0ffd`,
`886c1c7d`). The win identified here is *cheaper* coefficients, not cached ones.

**P2 — features.**

  1. Reshape-based lazy `_spre`/`_spost` in QuantumToolbox — the single unlock of §3.3.
  2. `cu(::QuantumObjectEvolution)`, plus `Adapt.adapt_structure` for these operator types.
  3. Spin/qubit matrix-free operator types.
  4. `AbstractMatrix` action for `LocalTensorProductOperator`.

**P3 — ecosystem.** Wiring the subpackage into the repository, deliberately deferred because this
branch is behind `upstream/main` and `upstream/lib/core` already restructures `src/` into
`lib/QuantumToolboxCore/`. When rebasing onto that branch:

  1. Add `QuantumToolboxOperators` to the root `Project.toml` `[deps]` and `[sources]`.
  2. Add it to `LIBRARY_NAME_AND_PATH` in `test/runtests.jl` and convert `@testset`s to
     `@testitem`s (mechanical: rename the macro, move the `using` lines into each item, delete
     `test/runtests.jl`).
  3. Add `'lib/**'` to the path filters in `.github/workflows/CI.yml` and `Code-Quality.yml`, plus
     the "Dev local libs for Julia < 1.11" step that `upstream/lib/core` uses.
  4. Add the module to `makedocs(modules = […])` in `docs/make.jl` and add a docs page.
  5. Add a Reactant extension to QuantumToolbox proper — §4.5 suggests it is the strongest backend
     for this workload.

**Pending an upstream fix.** `LocalTensorProductOperator` computes its element type with a local
`_promote_op_eltype` helper instead of `Base.promote_eltype`, because as of SciMLOperators 1.25 the
latter silently returns `Any` when the sub-operators share a concrete type (see the comment in
`src/tensor_product.jl` for why). This is fixed in SciMLOperators' development version; once the
compat lower bound here includes it, the helper can be deleted and the constructor can call
`Base.promote_eltype(ops...)` directly.

---

## 7. Running the tests

```bash
julia --project=lib/QuantumToolboxOperators -e 'import Pkg; Pkg.test()'
```
