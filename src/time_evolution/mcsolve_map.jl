export mcsolve_map

_mcsolve_map_axes(::NullParameters) = ()
_mcsolve_map_axes(params::Tuple) = params
_mcsolve_map_params(::NullParameters, indices) = NullParameters()
_mcsolve_map_params(params::Tuple, indices) = ntuple(i -> params[i][indices[i + 1]], length(params))

struct MCSolveMapProbFunc{ST, PT, IT, TT}
    states::ST
    params::PT
    indices::IT
    times::TT
end

function (f::MCSolveMapProbFunc)(prob, ctx)
    indices = Tuple(f.indices[mod1(ctx.sim_id, length(f.indices))])
    return _mcsolve_prob_func(
        prob, ctx, f.times;
        u0 = f.states[indices[1]], p = _mcsolve_map_params(f.params, indices),
    )
end

struct MCSolveMapOutputFunc{DT, TT, NT}
    dimensions::DT
    times::TT
    normalize_states::NT
end

function (f::MCSolveMapOutputFunc)(sol, ctx)
    SciMLBase.successful_retcode(sol) ||
        error("Monte Carlo trajectory $(ctx.sim_id) failed with return code $(sol.retcode).")
    expvals = _get_expvals(sol, SaveFuncMCSolve)
    if !isnothing(expvals)
        _get_save_callback(sol, SaveFuncMCSolve).affect!.func.iter[] == length(f.times) + 1 ||
            throw(ArgumentError("Monte Carlo trajectory $(ctx.sim_id) terminated before all expectation values were saved."))
    end
    jump = _mc_get_jump_callback(sol).affect!
    count = jump.col_times_which_idx[] - 1
    states = [_normalize_state!(u, f.dimensions, f.normalize_states) for u in sol.u]
    # Copy only the recorded jumps, releasing the overallocated callback buffers.
    output = (
        states = states, expect = expvals, times_states = sol.t,
        col_times = jump.col_times[1:count], col_which = jump.col_which[1:count],
    )
    return output, false
end

mutable struct MCSolveMapAccumulator{ST, ET, TT}
    states::ST
    expect::ET
    times_states::TT
    col_times::Vector{Vector{Float64}}
    col_which::Vector{Vector{Int}}
    count::Int
end

function _mcsolve_map_states(ψ0, ntraj, ::Val{false})
    return Vector{Base.promote_op(ket2dm, typeof(ψ0))}()
end
_mcsolve_map_states(ψ0, ntraj, ::Val{true}) = Matrix{typeof(ψ0)}(undef, ntraj, 0)

_mcsolve_map_expect(::Nothing, times, ntraj, T, keep_runs_results) = nothing
_mcsolve_map_expect(e_ops::Union{AbstractVector, Tuple}, times, ntraj, T, ::Val{false}) = zeros(T, length(e_ops), length(times))
_mcsolve_map_expect(e_ops::Union{AbstractVector, Tuple}, times, ntraj, T, ::Val{true}) = zeros(T, length(e_ops), ntraj, length(times))

function _mcsolve_map_accumulator(prob, e_ops, ntraj, keep_runs_results)
    ψ0 = QuantumObject(prob.prob.u0, Ket(), prob.dimensions)
    return MCSolveMapAccumulator(
        _mcsolve_map_states(ψ0, ntraj, keep_runs_results),
        _mcsolve_map_expect(e_ops, prob.times, ntraj, eltype(ψ0), keep_runs_results),
        similar(prob.times, 0),
        Vector{Vector{Float64}}(undef, ntraj),
        Vector{Vector{Int}}(undef, ntraj),
        0,
    )
end

function _mcsolve_map_accumulator(output, ntraj, keep_runs_results)
    states = _mcsolve_map_empty_states(eltype(output.states), ntraj, keep_runs_results)
    expvals = _mcsolve_map_empty_expect(output.expect, ntraj, keep_runs_results)
    return MCSolveMapAccumulator(
        states, expvals, similar(output.times_states, 0),
        Vector{Vector{Float64}}(undef, ntraj), Vector{Vector{Int}}(undef, ntraj), 0,
    )
end
_mcsolve_map_empty_states(::Type{T}, ntraj, ::Val{false}) where {T} = Vector{Base.promote_op(ket2dm, T)}()
_mcsolve_map_empty_states(::Type{T}, ntraj, ::Val{true}) where {T} = Matrix{T}(undef, ntraj, 0)
_mcsolve_map_empty_expect(::Nothing, ntraj, keep_runs_results) = nothing
_mcsolve_map_empty_expect(expvals::AbstractMatrix, ntraj, ::Val{false}) = zero(expvals)
_mcsolve_map_empty_expect(expvals::AbstractMatrix, ntraj, ::Val{true}) = zeros(eltype(expvals), size(expvals, 1), ntraj, size(expvals, 2))

function _mcsolve_map_store!(acc, output, trajectory, ::Val{false})
    isnothing(acc.expect) || (acc.expect .+= output.expect)
    return nothing
end

function _mcsolve_map_store_states!(acc, outputs, ::Val{false})
    isempty(first(outputs).states) && return nothing
    if acc.count == 0
        append!(acc.states, ket2dm.(first(outputs).states))
        foreach(ρ -> fill!(ρ.data, zero(eltype(ρ))), acc.states)
    end
    ψ1 = first(first(outputs).states)
    Ψ = similar(ψ1.data, length(ψ1.data), min(64, length(outputs)))
    for time in eachindex(acc.states)
        for block in Iterators.partition(outputs, size(Ψ, 2))
            Ψb = view(Ψ, :, 1:length(block))
            for (column, output) in zip(eachcol(Ψb), block)
                column .= output.states[time].data
            end
            mul!(acc.states[time].data, Ψb, Ψb', true, true)
        end
    end
    return nothing
end
_mcsolve_map_store_states!(acc, outputs, ::Val{true}) = nothing

function _mcsolve_map_store!(acc, output, trajectory, ::Val{true})
    if acc.count == 0
        acc.states = Matrix{eltype(output.states)}(undef, length(acc.col_times), length(output.states))
    end
    acc.states[trajectory, :] .= output.states
    isnothing(acc.expect) || (acc.expect[:, trajectory, :] .= output.expect)
    return nothing
end

struct MCSolveMapReduction{KT, PT}
    keep_runs_results::KT
    progress::PT
    ncases::Int
    ntraj::Int
end

function (f::MCSolveMapReduction)(accumulators, batch, indices)
    # Allocate a new result vector rather than mutating u_init. The ensemble
    # template retains its empty u_init, so distributed workers never receive
    # the growing accumulators when another batch is dispatched.
    if isempty(accumulators)
        accumulators = [_mcsolve_map_accumulator(first(batch), f.ntraj, f.keep_runs_results) for _ in 1:f.ncases]
    end
    for offset in 1:min(f.ncases, length(batch))
        acc = accumulators[mod1(indices[offset], f.ncases)]
        outputs = view(batch, offset:f.ncases:length(batch))
        if acc.count == 0
            acc.times_states = first(outputs).times_states
        end
        if !all(output -> acc.times_states == output.times_states, outputs)
            throw(ArgumentError("Trajectories at the same sweep point must save states at the same times."))
        end
        _mcsolve_map_store_states!(acc, outputs, f.keep_runs_results)
        for (output, sim_id) in zip(outputs, view(indices, offset:f.ncases:length(indices)))
            trajectory = cld(sim_id, f.ncases)
            _mcsolve_map_store!(acc, output, trajectory, f.keep_runs_results)
            acc.col_times[trajectory] = output.col_times
            acc.col_which[trajectory] = output.col_which
            acc.count += 1
        end
    end
    isnothing(f.progress) || next!(f.progress; step = length(batch))
    return accumulators, false
end

function _mcsolve_map_average!(acc, ::Val{false})
    foreach(ρ -> (ρ.data ./= acc.count), acc.states)
    isnothing(acc.expect) || (acc.expect ./= acc.count)
    return nothing
end
_mcsolve_map_average!(acc, ::Val{true}) = nothing

# SplitThreads currently sends empty slices to workers when a tail batch is
# shorter than the worker count. Reduce the batch size or use distributed
# execution for batches that cannot keep every worker occupied.
_mcsolve_map_backend(backend, total, batch_size) = (backend, batch_size)
function _mcsolve_map_backend(backend::EnsembleSplitThreads, total, batch_size)
    nworkers = Distributed.nworkers()
    while batch_size >= nworkers && 0 < rem(total, batch_size) < nworkers
        batch_size -= 1
    end
    return batch_size < nworkers ? (EnsembleDistributed(), min(total, batch_size)) : (backend, batch_size)
end

@doc raw"""
    mcsolve_map(
        H::Union{AbstractQuantumObject{Operator},Tuple},
        ψ0::Union{QuantumObject{Ket},AbstractVector{<:QuantumObject{Ket}}},
        tlist::AbstractVector,
        c_ops::Union{Nothing,AbstractVector,Tuple} = nothing;
        params::Union{NullParameters,Tuple} = NullParameters(),
        ntraj::Int = 500,
        ensemblealg::EnsembleAlgorithm = EnsembleThreads(),
        batch_size::Int = 1024,
        kwargs...,
    )

Solve Monte Carlo wave-function evolution for every combination of initial states
and parameter values using a single `EnsembleProblem`. `ntraj` is the number of
trajectories **per combination**. The result is an array of [`TimeEvolutionMCSol`](@ref)
with size `(length(ψ0), length(params[1]), length(params[2]), ...)`. A single initial
state retains the leading dimension of length one; without a parameter sweep,
the result is a vector.

# Arguments

- `H`, `tlist`, `c_ops`: Hamiltonian, observation times, and collapse operators, as in [`mcsolve`](@ref).
- `ψ0`: A single initial [`Ket`](@ref) or a vector of initial kets, all with the same dimensions.
- `params`: A tuple of nonempty parameter vectors or ranges. Each trajectory receives a tuple containing one value from each sweep axis. Hamiltonian and collapse-operator coefficients can depend on these values through their parameter argument. Without a sweep, trajectories receive `NullParameters()`.
- `ntraj`: Positive number of trajectories per sweep point. Default is `500`.
- `alg`: ODE algorithm, default `DP5()`.
- `ensemblealg`: Ensemble scheduling algorithm, default `EnsembleThreads()`. Serial, distributed, and split-threaded execution are supported.
- `e_ops`, `rng`, `jump_callback`, `normalize_states`: As in [`mcsolve`](@ref).
- `keep_runs_results`: With `Val(false)` (default), accumulate density matrices and expectations between batches. With `Val(true)`, retain each trajectory's states and expectations. Jump histories are retained in both cases.
- `progress_bar`: Whether to display progress, updated after each completed batch. Default is `Val(true)`.
- `batch_size`: Maximum number of trajectory outputs retained before reduction, default `1024`. Smaller batches reduce working memory but increase scheduling overhead. The split-threaded backend may reduce this size or use distributed execution to avoid empty worker batches.
- `pmap_batch_size`: Optional distributed scheduling batch size, forwarded to the ensemble solve.
- `rng_func`: Per-trajectory RNG factory passed to SciML, default `SciMLBase.default_rng_func`. Jump callbacks use the resulting `ctx.rng`.
- `kwargs`: Keyword arguments for the underlying `ODEProblem`, including tolerances and `saveat`.

# Notes

The logical trajectory array has dimensions `(sweep_dimensions..., ntraj)`.
Increasing `ntraj` preserves existing trajectory random streams for a fixed grid
and identically initialized `rng`. A one-point sweep matches [`mcsolve`](@ref)
with the same parameters and RNG. All parameters must preserve the model's
dimensions and operator structure. A failed trajectory raises an error.

See [`sesolve_map`](@ref) and [`mesolve_map`](@ref) for deterministic sweeps.
"""
function mcsolve_map(
        H::Union{AbstractQuantumObject{Operator}, Tuple},
        ψ0::AbstractVector{<:QuantumObject{Ket}},
        tlist::AbstractVector,
        c_ops::Union{Nothing, AbstractVector, Tuple} = nothing;
        alg::AbstractODEAlgorithm = DP5(),
        e_ops::Union{Nothing, AbstractVector, Tuple} = nothing,
        params::Union{NullParameters, Tuple} = NullParameters(),
        rng::AbstractRNG = default_rng(),
        ntraj::Int = 500,
        ensemblealg::EnsembleAlgorithm = EnsembleThreads(),
        jump_callback::TJC = ContinuousLindbladJumpCallback(),
        progress_bar::Union{Val, Bool} = Val(true),
        keep_runs_results::Union{Val, Bool} = Val(false),
        normalize_states::Union{Val, Bool} = Val(true),
        batch_size::Int = 1024,
        pmap_batch_size::Union{Int, Nothing} = nothing,
        rng_func = SciMLBase.default_rng_func,
        kwargs...,
    ) where {TJC <: LindbladJumpCallbackType}
    isempty(ψ0) && throw(ArgumentError("The initial-state list must not be empty."))
    ntraj > 0 || throw(ArgumentError("ntraj must be positive."))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive."))
    isnothing(pmap_batch_size) || pmap_batch_size > 0 || throw(ArgumentError("pmap_batch_size must be positive."))
    axes = _mcsolve_map_axes(params)
    all(axis -> axis isa AbstractVector && !isempty(axis), axes) ||
        throw(ArgumentError("Each parameter sweep axis must be a nonempty vector or range."))
    Base.require_one_based_indexing(ψ0, axes...)
    all(state -> state.dimensions == first(ψ0).dimensions, ψ0) ||
        throw(DimensionMismatch("All initial states must have the same dimensions."))

    indices = CartesianIndices((length(ψ0), map(length, axes)...))
    total = Base.checked_mul(length(indices), ntraj)
    T = _complex_float_type(mapreduce(eltype, promote_type, ψ0))
    initial = QuantumObject(to_dense(T, first(ψ0).data), Ket(), first(ψ0).dimensions)
    prob = mcsolveProblem(
        H, initial, tlist, c_ops;
        e_ops = e_ops, params = _mcsolve_map_params(params, Tuple(first(indices))),
        rng = rng, jump_callback = jump_callback, kwargs...,
    )
    return _mcsolve_map_solve(
        prob, ψ0, params, indices, alg, e_ops, rng, ntraj, ensemblealg,
        makeVal(progress_bar), makeVal(keep_runs_results), makeVal(normalize_states),
        batch_size, pmap_batch_size, rng_func, haskey(kwargs, :callback),
    )
end

# Specialize setup and reduction on the concrete ODEProblem, including when
# constructing a composed time-dependent operator cannot infer its cache type.
function _mcsolve_map_solve(
        prob, ψ0, params, indices, alg, e_ops, rng, ntraj, ensemblealg,
        progress_bar, keep, normalize_states, batch_size, pmap_batch_size, rng_func, safetycopy,
    )
    states = [copy(to_dense(eltype(prob.prob.u0), state.data)) for state in ψ0]
    accumulators = [_mcsolve_map_accumulator(prob, e_ops, ntraj, keep) for _ in 1:0]
    total = Base.checked_mul(length(indices), ntraj)
    progress = getVal(progress_bar) ?
        Progress(total; desc = "[mcsolve_map] ", settings.ProgressMeterKWARGS...) : nothing
    ensemble = EnsembleProblem(
        prob.prob;
        prob_func = MCSolveMapProbFunc(states, params, indices, prob.times),
        output_func = MCSolveMapOutputFunc(prob.dimensions, prob.times, normalize_states),
        reduction = MCSolveMapReduction(keep, progress, length(indices), ntraj), u_init = accumulators,
        safetycopy = safetycopy,
    )
    backend, batch = _mcsolve_map_backend(ensemblealg, total, min(total, batch_size))
    pmap_batch = isnothing(pmap_batch_size) ? max(1, div(batch, 100)) : pmap_batch_size
    sol = solve(ensemble, alg, backend; trajectories = total, batch_size = batch, pmap_batch_size = pmap_batch, rng = rng, rng_func = rng_func)
    results = map(sol.u::typeof(accumulators)) do acc
        _mcsolve_map_average!(acc, keep)
        TimeEvolutionMCSol(
            acc.count, prob.times, acc.times_states, acc.states, acc.expect,
            acc.col_times, acc.col_which, true, alg, prob.prob.kwargs[:abstol], prob.prob.kwargs[:reltol],
        )
    end
    return reshape(results, size(indices))
end

mcsolve_map(
    H::Union{AbstractQuantumObject{Operator}, Tuple},
    ψ0::QuantumObject{Ket},
    tlist::AbstractVector,
    c_ops::Union{Nothing, AbstractVector, Tuple} = nothing;
    kwargs...,
) = mcsolve_map(H, [ψ0], tlist, c_ops; kwargs...)
