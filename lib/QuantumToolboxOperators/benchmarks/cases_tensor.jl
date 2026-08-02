# ──────────────────────────────────────────────────────────────────────────────
# Multi-mode Hamiltonians, three ways:
#
#   1. QuantumToolbox sparse (`multisite_operator`)      — the status quo
#   2. SciMLOperators `TensorProductOperator`            — the lazy competitor
#   3. `LocalTensorProductOperator`                      — this package
#
# Both chains include a term acting on the *last* subsystem, which is the contiguous fast path.
#
# NOTE: `LocalTensorProductOperator` does not adapt its sub-operators to the state's array type.
# `cache_operator` moves the work buffers to the device, but a `MatrixOperator` wrapping a host
# `Matrix` stays on the host and `mul!` then fails. So the chains are rebuilt per backend with
# device-resident sub-operator data. Bosonic operators are immune — they store only a dimension.
# ──────────────────────────────────────────────────────────────────────────────

"""
    share_cache(added, ψ)

Rebuild an `AddedOperator` of `LocalTensorProductOperator`s so all terms share one set of work
buffers.

This is a **manual workaround for a missing feature**: `cache_operator` on an `M`-term sum
allocates `2M` full state vectors, even though `AddedOperator` evaluates its terms strictly one at
a time so a single pair would do. It is safe for that reason, but it is *not* thread-safe and is
not something a user would discover. Both variants are benchmarked so the cost of the gap shows up
in the memory table. See STATUS.md.
"""
function share_cache(added, ψ)
    shared = cache_operator(added.ops[1], ψ).cache
    return SciMLOperators.AddedOperator(
        (LocalTensorProductOperator(op.dims, op.indices, op.ops, shared) for op in added.ops)...,
    )
end

# `site_ops(j, device)` returns the pair (left factor on site j, right factor on site j) as
# SciMLOperators; `site_qobjs(j)` returns the same pair as QuantumToolbox `QuantumObject`s.
function build_chain(T, dims, site_ops, site_qobjs, device)
    nsites = length(dims)

    H_sparse = sum(1:(nsites - 1)) do n
        l, _ = site_qobjs(n)
        _, r = site_qobjs(n + 1)
        SparseMatrixCSC{T}(multisite_operator(dims, n => l, n + 1 => r).data)
    end

    H_tensor = sum(1:(nsites - 1)) do n
        factors = ntuple(nsites) do j
            j == n ? site_ops(j, device)[1] :
                (j == n + 1 ? site_ops(j, device)[2] : SciMLOperators.IdentityOperator(dims[j]))
        end
        SciMLOperators.TensorProductOperator(factors...)
    end

    H_local = sum(1:(nsites - 1)) do n
        LocalTensorProductOperator(dims, n => site_ops(n, device)[1], n + 1 => site_ops(n + 1, device)[2])
    end

    return H_sparse, H_tensor, H_local
end

function bench_chain!(backends, label, T, dims, site_ops, site_qobjs)
    ψ_host = normalize(randn(Random.default_rng(), T, prod(dims)))

    for backend in backends
        backend.available || continue

        H_sparse, H_tensor, H_local = build_chain(T, dims, site_ops, site_qobjs, backend.to_device)
        ψ = backend.to_device(ψ_host)
        dψ = similar(ψ)

        sp = backend.to_sparse(H_sparse)
        ref = similar(ψ)
        mul!(ref, sp, ψ)

        variants = Any[("sparse", sp, Base.summarysize(H_sparse), nothing)]

        # TensorProductOperator's cache machinery does not follow the state onto the device; keep
        # it to the CPU rather than report a spurious failure.
        if backend.name == "CPU"
            Ht = cache_operator(H_tensor, ψ)
            push!(variants, ("TensorProductOperator", Ht, Base.summarysize(H_tensor), Base.summarysize(Ht)))
        end

        Hl = cache_operator(H_local, ψ)
        push!(variants, ("LocalTensorProduct", Hl, Base.summarysize(H_local), Base.summarysize(Hl)))

        Hs = share_cache(H_local, ψ)
        push!(variants, ("LocalTensorProduct (shared cache)", Hs, Base.summarysize(H_local), Base.summarysize(Hs)))

        for (vname, op, bytes_op, bytes_cached) in variants
            t = measure(() -> mul!(dψ, op, ψ), backend)
            status = "PASS"
            if vname != "sparse"
                try
                    mul!(dψ, op, ψ)
                    status = check(dψ, ref)
                catch
                    status = "n/a"
                end
            end
            record!(
                Result(
                    group = "tensor", case = label, backend = backend.name, variant = vname,
                    time_min = t === nothing ? nothing : t.min,
                    time_median = t === nothing ? nothing : t.median,
                    bytes_op = bytes_op, bytes_cached = bytes_cached, status = status,
                )
            )
        end
    end

    # ─── XLA ─────────────────────────────────────────────────────────────────
    # The workshop benchmarks left the TensorProductOperator `@compile` commented out, implying it
    # did not work. Try both here and record what actually happens.
    if REACTANT_OK[]
        H_sparse, H_tensor, H_local = build_chain(T, dims, site_ops, site_qobjs, identity)
        ψ_r = Reactant.to_rarray(ψ_host)
        dψ_r = similar(ψ_r)
        ref_host = H_sparse * ψ_host

        # Caching must happen against the traced array: buffers built from the host vector are
        # plain `Vector`s that `mul!` then tries to write traced values into. Both the caching and
        # the compile can fail, so the whole attempt sits inside the `try`.
        for (vname, build) in (
                ("LocalTensorProduct", () -> share_cache(H_local, ψ_r)),
                ("TensorProductOperator", () -> cache_operator(H_tensor, ψ_r)),
            )
            try
                op_host = build()
                compile_s = @elapsed compiled = Reactant.compile(mul!, (dψ_r, op_host, ψ_r))
                t = measure(() -> compiled(dψ_r, op_host, ψ_r), CPU_BACKEND)
                compiled(dψ_r, op_host, ψ_r)
                record!(
                    Result(
                        group = "tensor", case = label, backend = "XLA", variant = vname,
                        time_min = t === nothing ? nothing : t.min,
                        time_median = t === nothing ? nothing : t.median,
                        compile_s = compile_s, status = check(Array(dψ_r), ref_host),
                    )
                )
            catch err
                @warn "Reactant failed: $label / $vname" exception = (err, catch_backtrace())
                record!(
                    Result(
                        group = "tensor", case = label, backend = "XLA",
                        variant = vname, status = "unsupported: " * _short_reason(err),
                    )
                )
            end
        end
    end

    return nothing
end

function bench_tensor!(backends)
    T = ComplexF32

    # ─── Spin chain: H = Σₙ σˣₙ σˣₙ₊₁ ───────────────────────────────────────
    nspins = SIZE == "smoke" ? 8 : 18
    spin_dims = ntuple(_ -> 2, nspins)
    σx_data = T[0 1; 1 0]
    bench_chain!(
        backends, "$nspins spins (XX chain, dim $(2^nspins))", T, spin_dims,
        (j, device) -> begin
            op = SciMLOperators.MatrixOperator(device(σx_data))
            (op, op)
        end,
        _ -> (sigmax(), sigmax()),
    )

    # ─── Cavity chain: H = Σₙ â†ₙ âₙ₊₁ ───────────────────────────────────────
    ncav, d = SIZE == "smoke" ? (3, 10) : (4, 30)
    cav_dims = ntuple(_ -> d, ncav)
    bench_chain!(
        backends, "$ncav cavities (d=$d, dim $(d^ncav))", T, cav_dims,
        # Bosonic operators store only a dimension, so they need no device adaptation.
        (j, _) -> (DestroyOperator{T}(cav_dims[j])', DestroyOperator{T}(cav_dims[j])),
        j -> (create(T, cav_dims[j]), destroy(T, cav_dims[j])),
    )

    return nothing
end
