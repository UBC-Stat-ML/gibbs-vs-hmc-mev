#!/usr/bin/env -S julia --heap-size-hint=${task.memory.toGiga()}G
using Pkg
Pkg.activate(joinpath("$baseDir", "$julia_env")) 
include(joinpath("$baseDir", "$julia_env", "src", "sampling_functions.jl")) # loads dependencies too

const MINIMUM_REQUIRED_ESS = ${ESS_threshold}
const PRIOR_SIGMA = 10.0

function main()
	# collect global vars 
	explorer_type = "${arg.sampler}"
	d = ${arg.dim}
	max_d = ${dim_string[arg.data]}
	requested_size = ${arg.size}
	data_name = "${arg.data}"
	seed = ${arg.seed}
	n_samples = 1000
	model = "${arg.model}"
	prior = "${arg.prior}"

	df = if explorer_type == "CGGibbs" # use CGGibbs 
		cggibbs(seed,data_name,n_samples,d,requested_size;
			model=model,prior=prior,synthetic=false)
	elseif explorer_type == "NUTS_Stan"
		nuts_stansample(seed,data_name,n_samples,d,requested_size;
			model=model,prior=prior,synthetic=false)
	else # use Turing for NUTS
	    nuts(seed,data_name,n_samples,d,requested_size;
			model=model,prior=prior,synthetic=false)
	end
	
	isdir("csvs") || mkdir("csvs")
	CSV.write("csvs/summary.csv", df)
end

main()

