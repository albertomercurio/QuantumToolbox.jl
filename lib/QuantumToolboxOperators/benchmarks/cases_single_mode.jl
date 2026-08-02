# ──────────────────────────────────────────────────────────────────────────────
# Single-mode bosonic operators: matrix-free vs. sparse.
#
# The individual terms of the Kerr Hamiltonian are benchmarked alongside the whole thing. That is
# what turns "the composite operator is slow" into "this specific term is slow", which is the only
# form of that statement worth putting in a roadmap.
# ──────────────────────────────────────────────────────────────────────────────

function bench_single_mode!(backends)
    T = ComplexF32
    N = SIZE == "smoke" ? 10_000 : 1_000_000
    Δ, U, F = T(0.1), T(0.2), T(0.3)

    a = DestroyOperator{T}(N)
    A = concretize(a)                        # sparse â, the baseline representation

    cases = [
        ("â", a, A),
        ("â†", a', sparse(A')),
        ("n̂ = â†â", a' * a, A' * A),
        ("â²", a^2, A^2),
        ("(â†)²â²", a'^2 * a^2, A'^2 * A^2),
        (
            "Kerr H",
            Δ * a' * a + U * (a'^2 * a^2) + F * (a + a'),
            Δ * (A' * A) + U * (A'^2 * A^2) + F * (A + sparse(A')),
        ),
    ]

    ψ_host = normalize(randn(Random.default_rng(), T, N))

    for backend in backends
        backend.available || continue
        ψ = backend.to_device(ψ_host)
        dψ = similar(ψ)

        for (name, lazy_raw, sp_raw) in cases
            lazy = cache_operator(lazy_raw, ψ)
            sp = backend.to_sparse(sp_raw)

            # Reference: always the sparse product, computed on this backend.
            ref = similar(ψ)
            mul!(ref, sp, ψ)

            t_lazy = measure(() -> mul!(dψ, lazy, ψ), backend)
            mul!(dψ, lazy, ψ)
            status = check(dψ, ref)

            t_sparse = measure(() -> mul!(dψ, sp, ψ), backend)

            record!(
                Result(
                    group = "single", case = name, backend = backend.name, variant = "lazy",
                    time_min = t_lazy === nothing ? nothing : t_lazy.min,
                    time_median = t_lazy === nothing ? nothing : t_lazy.median,
                    bytes_op = Base.summarysize(lazy_raw), bytes_cached = Base.summarysize(lazy),
                    status = status,
                )
            )
            record!(
                Result(
                    group = "single", case = name, backend = backend.name, variant = "sparse",
                    time_min = t_sparse === nothing ? nothing : t_sparse.min,
                    time_median = t_sparse === nothing ? nothing : t_sparse.median,
                    bytes_op = Base.summarysize(sp_raw),
                    status = "PASS",
                )
            )
        end
    end

    # ─── XLA ─────────────────────────────────────────────────────────────────
    # Restricted to â and the Kerr H: each `@compile` costs minutes.
    if REACTANT_OK[]
        ψ_r = Reactant.to_rarray(ψ_host)
        dψ_r = similar(ψ_r)

        for (name, lazy_raw, sp_raw) in cases
            name in ("â", "Kerr H") || continue
            lazy = cache_operator(lazy_raw, ψ_host)

            try
                # `Reactant.compile` is the function form of the `@compile` macro; it takes the
                # argument tuple directly, which is what we need inside a loop.
                compile_s = @elapsed compiled = Reactant.compile(mul!, (dψ_r, lazy, ψ_r))
                t = measure(() -> compiled(dψ_r, lazy, ψ_r), CPU_BACKEND)
                compiled(dψ_r, lazy, ψ_r)
                status = check(Array(dψ_r), sp_raw * ψ_host)

                record!(
                    Result(
                        group = "single", case = name, backend = "XLA", variant = "lazy",
                        time_min = t === nothing ? nothing : t.min,
                        time_median = t === nothing ? nothing : t.median,
                        compile_s = compile_s, status = status,
                    )
                )
            catch err
                @warn "Reactant failed for case $name" exception = (err, catch_backtrace())
                record!(Result(group = "single", case = name, backend = "XLA", variant = "lazy", status = "n/a"))
            end
        end
    end

    return nothing
end
