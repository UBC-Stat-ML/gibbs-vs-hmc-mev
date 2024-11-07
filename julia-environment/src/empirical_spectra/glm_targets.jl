using CSV 
using StanSample
using DataFrames

const PRIOR_SIGMA = 10.0

include("../data_utils.jl")
include("../utils.jl")
include("../Stan_functions.jl")


struct GLMTarget 
    dataset::String # e.g. colon, etc
    prior::String 
    samples_file::String
end
Base.show(io::IO, t::GLMTarget) = print(io, "GLMTarget_" * t.dataset * "_" * t.prior)

function reference_sample(t::GLMTarget, constrained::Bool)
   
    log_potential = log_pot(t)
    bridge_stan_names = param_names(log_potential.model)
    n_params = length(bridge_stan_names)
    result_array = Vector{Vector{Float64}}()

    raw = CSV.read(t.samples_file, DataFrame)

    for iter_index in 1:nrow(raw)
        current_constrained = Vector(raw[iter_index, 10:end])

        current_csv_lp = raw[iter_index, 3]
        unconstrained = param_unconstrain(log_potential.model, current_constrained)
        bridgestan_lp = log_density(log_potential.model, unconstrained)
        if iter_index < 10
            # check we get same lp. Unfortunately, Stan's log uses crazy aggressive rounding
            @assert isapprox(current_csv_lp, bridgestan_lp; atol = 0.1) (; csv = current_csv_lp, bs = bridgestan_lp)
        end
        @assert length(current_constrained) == n_params
        push!(result_array, constrained ? current_constrained : unconstrained)
    end

    return result_array
end

full_model(t) = "logistic_" * t.prior

function log_pot(t::GLMTarget)
    rng = SplittableRandom(1)
    model = "logistic" 
    rows_x, vec_y, data_size, d = load_and_preprocess_data(t.dataset;
      distribution = false, synthetic = false, rng, model)
    sm = SampleModel(full_model(t), model_string(full_model(t)))
    dist_params = nothing
    data = stan_data(full_model(t), rows_x, vec_y, d, dist_params)
    StanSample.update_json_files(sm, data, 1, "data")
    json_file = sm.data_file 

    return StanLogPotential(model_path(full_model(t)), json_file[1])
end



