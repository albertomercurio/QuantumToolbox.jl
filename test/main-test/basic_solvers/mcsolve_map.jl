#!PTR_MULTITHREAD
using Test
using QuantumToolbox
using LinearAlgebra
import Random: MersenneTwister, seed!, default_rng
import QuantumToolbox: SciMLBase, DiscreteCallback, ContinuousCallback, EnsembleSerial

@testset "mcsolve_map" begin
    states = [basis(2, 0), (basis(2, 0) + basis(2, 1)) / sqrt(2)]
    original_states = copy.(states)
    times = collect(range(0.0, 2.0, 21))
    H = QobjEvo(sigmaz() / 2, (p, t) -> p[1])
    c_ops = (QobjEvo(sigmam(), (p, t) -> sqrt(p[2])),)
    e_ops = (sigmap() * sigmam(), sigmax())
    params = ([0.2, 1.0], [0.3, 0.9])
    ntraj = 24
    options = (e_ops = e_ops, params = params, ntraj = ntraj, progress_bar = Val(false), keep_runs_results = Val(true))

    serial = mcsolve_map(H, states, times, c_ops; options..., rng = MersenneTwister(43), ensemblealg = EnsembleSerial())
    threaded = mcsolve_map(H, states, times, c_ops; options..., rng = MersenneTwister(43), batch_size = 13)
    averaged = mcsolve_map(
        H, states, times, c_ops;
        e_ops = e_ops, params = params, ntraj = ntraj, rng = MersenneTwister(43),
        progress_bar = Val(false), batch_size = 13,
    )
    @test size(serial) == size(threaded) == size(averaged) == (2, 2, 2)
    @test serial isa Array{<:TimeEvolutionMCSol}
    @test states == original_states
    for case in eachindex(serial)
        actual = threaded[case]
        expected = serial[case]
        @test actual.ntraj == ntraj
        @test actual.converged
        @test size(actual.states) == (ntraj, 1)
        @test size(actual.expect) == (2, ntraj, length(times))
        @test actual.states == expected.states
        @test actual.expect == expected.expect
        @test actual.col_times == expected.col_times
        @test actual.col_which == expected.col_which
        @test all(isapprox.(averaged[case].states, average_states(expected)))
        @test averaged[case].expect ≈ average_expect(expected)
        @test averaged[case].col_times == expected.col_times
        @test averaged[case].col_which == expected.col_which
    end

    # Compare with independent problems using the corresponding SciML streams.
    master = MersenneTwister(43)
    rand(master) # mcsolveProblem initializes its template jump callback
    seeds = [rand(master, UInt64) for _ in 1:(length(serial) * ntraj)]
    for case in eachindex(serial)
        indices = Tuple(CartesianIndices(serial)[case])
        p = (params[1][indices[2]], params[2][indices[3]])
        rng_func = ctx -> (seed!(seeds[case + length(serial) * (ctx.sim_id - 1)]); default_rng())
        reference_prob = mcsolveEnsembleProblem(
            H, states[indices[1]], times, c_ops;
            e_ops = e_ops, params = p, progress_bar = Val(false), rng = MersenneTwister(43),
        )
        reference = SciMLBase.solve(reference_prob.prob, QuantumToolbox.DP5(), EnsembleSerial(); trajectories = ntraj, rng_func = rng_func)
        @test serial[case].expect == stack([QuantumToolbox._get_expvals(sol, QuantumToolbox.SaveFuncMCSolve) for sol in reference.u], dims = 2)
        @test serial[case].col_times == [QuantumToolbox._mc_get_jump_callback(sol).affect!.col_times for sol in reference.u]
        for trajectory in 1:ntraj
            @test serial[case].states[trajectory, 1].data ≈ normalize(reference.u[trajectory].u[end])
        end
    end

    extended = mcsolve_map(H, states, times, c_ops; options..., ntraj = ntraj + 3, rng = MersenneTwister(43))
    for case in eachindex(serial)
        @test extended[case].expect[:, 1:ntraj, :] == serial[case].expect
        @test extended[case].col_times[1:ntraj] == serial[case].col_times
    end

    @testset "Single case and saving options" begin
        for keep in (Val(false), Val(true)), jump in (ContinuousLindbladJumpCallback(), DiscreteLindbladJumpCallback())
            opts = (ntraj = 12, keep_runs_results = keep, jump_callback = jump, progress_bar = Val(false), e_ops = e_ops)
            single = mcsolve_map(H, states[1], times, c_ops; opts..., params = ([0.2], [0.3]), rng = MersenneTwister(43), batch_size = 5)
            reference = mcsolve(H, states[1], times, c_ops; opts..., params = (0.2, 0.3), rng = MersenneTwister(43))
            @test size(single) == (1, 1, 1)
            @test size(single[1].states) == size(reference.states)
            @test all(isapprox.(single[1].states, reference.states))
            @test single[1].expect ≈ reference.expect
            @test single[1].col_times == reference.col_times
            @test single[1].col_which == reference.col_which
        end
        constant_H = 0.2sigmaz()
        constant_c = (sqrt(0.3) * sigmam(),)
        @inferred mcsolve_map(H, states, times, constant_c; e_ops = e_ops, params = params, ntraj = 3, progress_bar = Val(false))
        for keep in (Val(false), Val(true)), saveat in (times[2:3:20], Float64[])
            opts = (ntraj = 8, keep_runs_results = keep, saveat = saveat, progress_bar = Val(false))
            single = @inferred mcsolve_map(constant_H, states[1], times, constant_c; opts..., rng = MersenneTwister(43))
            reference = mcsolve(constant_H, states[1], times, constant_c; opts..., rng = MersenneTwister(43))
            @test size(single) == (1,)
            @test single[1].expect === nothing
            @test single[1].times_states == reference.times_states
            @test size(single[1].states) == size(reference.states)
            @test all(isapprox.(single[1].states, reference.states))
        end
        unsaved = mcsolve_map(constant_H, states, times, constant_c; ntraj = 3, e_ops = e_ops, save_end = false, progress_bar = Val(false))
        @test all(sol -> isempty(sol.states) && isempty(sol.times_states), unsaved)
        @test all(sol -> size(sol.expect) == (2, length(times)), unsaved)
        empty_ops = @inferred mcsolve_map(constant_H, states[1], times, constant_c; ntraj = 3, e_ops = (), progress_bar = Val(false))
        @test size(empty_ops[1].expect) == (0, length(times))
        @test length(empty_ops[1].states) == length(times)
        no_params_H = QobjEvo(sigmaz(), (p, t) -> (p isa QuantumToolbox.NullParameters ? 0.2 : error("Expected NullParameters")))
        @test size(mcsolve_map(no_params_H, states, times, constant_c; ntraj = 3, progress_bar = Val(false))) == (2,)
        @test size(mcsolve_map(constant_H, states[1], times, constant_c; params = (), ntraj = 3, progress_bar = Val(false))) == (1,)
        ints = Qobj([1, 0])
        @inferred mcsolve_map(constant_H, ints, times, constant_c; ntraj = 3, progress_bar = Val(true))
        raw = mcsolve_map(constant_H, states[1], times, constant_c; ntraj = 4, normalize_states = Val(false), keep_runs_results = Val(true), rng = MersenneTwister(43), progress_bar = Val(false))
        raw_ref = mcsolve(constant_H, states[1], times, constant_c; ntraj = 4, normalize_states = Val(false), keep_runs_results = Val(true), rng = MersenneTwister(43), progress_bar = Val(false))
        @test raw[1].states == raw_ref.states
        # Exercise both matrix-product blocks with more than 64 trajectories,
        # strided sweep grouping, and unnormalized states at multiple times.
        block_opts = (ntraj = 70, e_ops = e_ops, saveat = times[2:3:20], normalize_states = Val(false), batch_size = 140, progress_bar = Val(false))
        kept = mcsolve_map(constant_H, states, times, constant_c; block_opts..., keep_runs_results = Val(true), rng = MersenneTwister(43))
        means = mcsolve_map(constant_H, states, times, constant_c; block_opts..., rng = MersenneTwister(43))
        for case in eachindex(kept)
            for time in eachindex(means[case].states)
                @test means[case].states[time] ≈ sum(ket2dm, kept[case].states[:, time]) / 70
            end
            @test means[case].expect ≈ average_expect(kept[case])
        end
    end

    @testset "Custom callback isolation" begin
        for jump in (ContinuousLindbladJumpCallback(), DiscreteLindbladJumpCallback()), ops in (nothing, e_ops)
            counter = Ref(0)
            affect! = integrator -> (counter[] += 1; integrator.u .*= counter[] + 1; SciMLBase.derivative_discontinuity!(integrator, true); nothing)
            initialize = (cb, u, t, integrator) -> cb.affect!(integrator)
            callback = DiscreteCallback((u, t, integrator) -> false, affect!; initialize = initialize, save_positions = (false, false))
            sols = mcsolve_map(
                0.0sigmaz(), states[1], [0.0, 1.0], (0.0sigmam(),);
                callback = callback, jump_callback = jump, e_ops = ops, ntraj = 6,
                normalize_states = Val(false), keep_runs_results = Val(true), progress_bar = Val(false),
            )
            @test counter[] == 0
            @test all(ψ -> norm(ψ) ≈ 2, sols[1].states[:, end])
        end
        # A user continuous callback without e_ops must also survive reset.
        callback = ContinuousCallback((u, t, integrator) -> 1.0, integrator -> nothing)
        @test length(mcsolve_map(0.0sigmaz(), states[1], times, (sigmam(),); callback = callback, ntraj = 2, progress_bar = Val(false))) == 1
        # Constant composed collapse operators still own mutable multiplication
        # caches and must not share these between trajectories.
        composed_c = QobjEvo(sigmam()) * QobjEvo(sigmaz())
        prob = mcsolveProblem(0.2sigmaz(), states[1], times, (composed_c,))
        cb1 = QuantumToolbox._mcsolve_initialize_callbacks(prob.prob, times, MersenneTwister(1))
        cb2 = QuantumToolbox._mcsolve_initialize_callbacks(prob.prob, times, MersenneTwister(2))
        c1 = QuantumToolbox._mc_get_jump_callback(cb1).affect!.c_ops[1]
        c2 = QuantumToolbox._mc_get_jump_callback(cb2).affect!.c_ops[1]
        @test c1 !== c2
        @test c1.cache !== c2.cache
    end

    @testset "Invalid input and failed trajectories" begin
        args = (H, states, times, c_ops)
        @test_throws ArgumentError mcsolve_map(args...; params = params, ntraj = 0)
        @test_throws ArgumentError mcsolve_map(args...; params = params, batch_size = 0)
        @test_throws ArgumentError mcsolve_map(args...; params = params, pmap_batch_size = 0)
        @test_throws ArgumentError mcsolve_map(args...; params = (Float64[], [0.3]))
        @test_throws ArgumentError mcsolve_map(args...; params = (1.0, [0.3]))
        @test_throws ArgumentError mcsolve_map(H, states[1:0], times, c_ops; params = params)
        @test_throws DimensionMismatch mcsolve_map(H, [states[1], basis(3, 0)], times, c_ops; params = params)
        @test_throws ArgumentError mcsolve_map(args...; params = params, save_idxs = [1])
        @test_throws ErrorException mcsolve_map(sigmax(), states[1], [0.0, 10.0], (sigmam(),); ntraj = 1, maxiters = 1, ensemblealg = EnsembleSerial(), progress_bar = Val(false))
    end
end
