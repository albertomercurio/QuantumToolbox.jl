#!PTR_MULTITHREAD
using Test
using QuantumToolbox
import Random: MersenneTwister
import QuantumToolbox: Distributed, EnsembleDistributed, EnsembleSplitThreads
import Distributed: addprocs, rmprocs, @everywhere

worker_ids = addprocs(2; exeflags = `--project=$(dirname(Base.active_project())) --startup-file=no --threads=2`)
@everywhere module MCSolveMapDistributedModel
using QuantumToolbox
frequency(p, t) = p[1]
amplitude(p, t) = sqrt(p[2])
function model()
    return (
        QobjEvo(sigmaz() / 2, frequency),
        [basis(2, 0), (basis(2, 0) + basis(2, 1)) / sqrt(2)],
        collect(range(0.0, 2.0, 11)),
        (QobjEvo(sigmam(), amplitude),),
    )
end
end

try
    @testset "mcsolve_map distributed backends" begin
        args = MCSolveMapDistributedModel.model()
        for keep in (Val(false), Val(true))
            options = (params = ([0.2, 1.0], [0.3, 0.9]), e_ops = (sigmaz(),), ntraj = 8, keep_runs_results = keep)
            reference = mcsolve_map(args...; options..., rng = MersenneTwister(43), progress_bar = Val(false))
            for backend in (EnsembleDistributed(), EnsembleSplitThreads()), progress in (Val(false), Val(true))
                # 64 jobs and batches of 21 exercise the one-job tail that
                # SplitThreads cannot currently dispatch to two workers.
                actual = mcsolve_map(args...; options..., rng = MersenneTwister(43), ensemblealg = backend, batch_size = 21, pmap_batch_size = 2, progress_bar = progress)
                @test size(actual) == size(reference)
                for case in eachindex(reference)
                    @test size(actual[case].states) == size(reference[case].states)
                    @test all(isapprox.(actual[case].states, reference[case].states))
                    @test actual[case].expect == reference[case].expect
                    @test actual[case].col_times == reference[case].col_times
                    @test actual[case].col_which == reference[case].col_which
                end
            end
            small_batches = mcsolve_map(args...; options..., rng = MersenneTwister(43), ensemblealg = EnsembleSplitThreads(), batch_size = 1, progress_bar = Val(false))
            @test all(a.expect == b.expect for (a, b) in zip(small_batches, reference))
        end
        @test length(mcsolve_map(0.0sigmaz(), basis(2, 0), [0.0, 1.0], (sigmam(),); ntraj = 1, ensemblealg = EnsembleSplitThreads(), progress_bar = Val(false))) == 1
    end
finally
    rmprocs(worker_ids)
end
