using Pigeons

pigeons_path = dirname(dirname(pathof(Pigeons)))
include(pigeons_path * "/test/supporting/postdb.jl")

reference_posteriors_dir() = "$(post_db_dir())/reference_posteriors/"

struct PosteriorDBTarget
    json_file::String
end

log_pot(t::PosteriorDBTarget) = log_potential_from_posterior_db(t.json_file) 

Base.show(io::IO, t::PosteriorDBTarget) = print(io, "PosteriorDB_" * t.json_file)

function reference_sample(t::PosteriorDBTarget, constrained::Bool)
    posterior_json_file = t.json_file
    rf = reference_file(posterior_json_file) 
    raw = JSON.parsefile(rf)
    n_chains = length(raw)
    n_variables = length(raw[1])
    n_draws = length(first(raw[1])[2]) 

    log_potential = log_pot(t)
    result_array = Vector{Vector{Float64}}()

    # we want them to be sorted properly for consumption into BridgeStan
    variable_names = fix_name.(param_names(log_potential.model))
    @assert length(variable_names) == n_variables

    for c_index in 1:n_chains 
        current_json_dict = raw[c_index]
        for iter_index in 1:n_draws 
            current_constrained = zeros(n_variables)
            
            for v_index in 1:n_variables 
                v_name = variable_names[v_index]
                current_constrained[v_index] = current_json_dict[v_name][iter_index]
            end

            if constrained
                push!(result_array, current_constrained) 
            else
                unconstrained = param_unconstrain(log_potential.model, current_constrained)
                push!(result_array, unconstrained)
            end
        end
    end
    return result_array
end

function reference_file(posterior_json_file) 
    file = "$(reference_posteriors_dir())/draws/draws/$posterior_json_file"
    if !isfile(file)
        unzip!("$file.zip")
    end
    return file
end

function fix_name(str)
    spl = Base.split(str, ".") 
    result = spl[1]
    if length(spl) > 1 
        result = result * "[" * join(spl[2:end], ",")  * "]"
    end
    return result 
end