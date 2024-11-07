#!/usr/bin/env -S julia --heap-size-hint=${task.memory.toGiga()}G
using Pkg
Pkg.activate(joinpath("$baseDir", "$julia_env")) 
include(joinpath("$baseDir", "$julia_env", "src", "sampling_functions.jl")) # loads dependencies too


# TODO: get rid of global variables (low priority)
const MINIMUM_REQUIRED_ESS = ${ESS_threshold}
const NUTS_ADAPT_STEPS = ${params.dryRun ? 10 : 1000}
const OPTIM_ITERATIONS = ${params.dryRun ? 4 : 1000}
const PRIOR_SIGMA = 10.0

function main()
	# collect global vars 
	explorer_type = "${arg.sampler}"
	d = ${arg.dim}
	max_d = ${dim_string[arg.data]}
	requested_size = ${arg.size}
	data_name = "${arg.data}"
	seed = ${arg.seed}
	n_samples = ${sample_string[arg.prior][arg.data]}
	model = "${arg.model}"
	prior = "${arg.prior}"

	if !isnothing(d)
		if d > max_d # set dimension to data size if requested_dim > data size
			d = max_d
		end
	end 

	df = if explorer_type == "NUTS_Stan"
		nuts_stan_run(seed,data_name,${params.dryRun ? "4" : "n_samples"},d,requested_size;
			n_samples = ${params.dryRun ? "4" : "1000"},
			model=model,prior=prior,synthetic=false)
	else 
		error()
	end
	
	CSV.write("logistic_${arg.prior}_${arg.data}.csv", df)
end



main()

