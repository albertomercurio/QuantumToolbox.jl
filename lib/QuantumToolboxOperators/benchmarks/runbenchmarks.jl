#!/usr/bin/env julia
#
# Benchmark driver for QuantumToolboxOperators.
#
#     julia --project=benchmarks --startup-file=no benchmarks/runbenchmarks.jl
#
# Environment:
#   QTO_BENCH_SIZE=smoke|full   problem sizes (default "full"; "smoke" runs in ~2 min)
#   QTO_BENCH_REACTANT=1        also measure the XLA backend (each `@compile` costs minutes)
#
# Degrades gracefully: no GPU or no working Reactant produces `n/a` cells, never a crash.

using Pkg
using LinearAlgebra
using SparseArrays
using Random
using Printf

using QuantumToolbox
using QuantumToolboxOperators
using SciMLOperators
using SciMLOperators: cache_operator, concretize
using CUDA

Random.seed!(20260802)
BLAS.set_num_threads(1)   # reproducibility: these are memory-bound kernels, not BLAS-bound

include("common.jl")

# ─── Reactant is optional and slow to load, so bring it in only on request ────

const REACTANT_OK = Ref(false)

if WANT_REACTANT
    try
        @eval using Reactant
        REACTANT_OK[] = true
        REACTANT_STATUS[] = "enabled"
    catch err
        @warn "Reactant requested but unavailable; XLA columns will read n/a" exception = err
        REACTANT_STATUS[] = "unavailable"
    end
else
    REACTANT_STATUS[] = "not requested (set QTO_BENCH_REACTANT=1)"
end

include("cases_single_mode.jl")
include("cases_tensor.jl")

# ─── Run ─────────────────────────────────────────────────────────────────────

PKG_VERSIONS[] = collect_pkg_versions()

const BACKENDS = [CPU_BACKEND, cuda_backend()]

@info "Running benchmarks" size = SIZE reactant = REACTANT_OK[] cuda = CUDA.functional()

@info "Single-mode operators…"
bench_single_mode!(BACKENDS)

@info "Multi-mode tensor products…"
bench_tensor!(BACKENDS)

# ─── Report ──────────────────────────────────────────────────────────────────

const OUTDIR = joinpath(@__DIR__, "results")
mkpath(OUTDIR)

open(joinpath(OUTDIR, "RESULTS.md"), "w") do io
    provenance_header(io)

    single = filter(r -> r.group == "single", RESULTS)
    tensor = filter(r -> r.group == "tensor", RESULTS)

    println(io, "## Single-mode operators\n")
    println(
        io, "Hilbert-space dimension ", SIZE == "smoke" ? "10⁴" : "10⁶", ", `ComplexF32`. ",
        "`Kerr H` is `Δ â†â + U (â†)²â² + F(â + â†)`; the rows above it are its individual terms.\n"
    )
    emit_table(io, single; columns = time_columns())

    println(io, "### Memory\n")
    println(
        io, "`operator` is before `cache_operator`, `cached` after. For single-mode operators ",
        "caching is a no-op, so the two agree.\n"
    )
    emit_table(
        io, filter(r -> r.backend == "CPU", single);
        columns = [
            "Case" => r -> r.case,
            "Variant" => r -> r.variant,
            "operator" => r -> fmt_bytes(r.bytes_op),
            "cached" => r -> fmt_bytes(r.bytes_cached),
        ],
    )

    println(io, "## Multi-mode tensor products\n")
    println(
        io, "`H = Σₙ Aₙ Bₙ₊₁` over a nearest-neighbor chain, compared against QuantumToolbox's ",
        "sparse `multisite_operator` and against `SciMLOperators.TensorProductOperator`.\n"
    )
    emit_table(io, tensor; columns = time_columns())

    println(io, "### Memory\n")
    emit_table(
        io, filter(r -> r.backend == "CPU", tensor);
        columns = [
            "Case" => r -> r.case,
            "Variant" => r -> r.variant,
            "operator" => r -> fmt_bytes(r.bytes_op),
            "cached" => r -> fmt_bytes(r.bytes_cached),
        ],
    )
end

# Raw rows, so STATUS.md figures can be re-derived without re-running.
open(joinpath(OUTDIR, "results.json"), "w") do io
    println(io, "[")
    for (i, r) in enumerate(RESULTS)
        fields = join(
            (
                "\"$k\": " * (
                        getfield(r, k) === nothing ? "null" :
                        getfield(r, k) isa AbstractString ? "\"$(getfield(r, k))\"" : string(getfield(r, k))
                    ) for k in fieldnames(Result)
            ), ", ",
        )
        println(io, "  {", fields, i == length(RESULTS) ? "}" : "},")
    end
    println(io, "]")
end

nfail = count(r -> r.status == "FAIL", RESULTS)
@info "Wrote $(joinpath(OUTDIR, "RESULTS.md"))" rows = length(RESULTS) failures = nfail
nfail == 0 || @error "$nfail correctness check(s) FAILED — do not publish these numbers"
