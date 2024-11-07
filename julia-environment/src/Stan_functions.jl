using StanSample

include("utils.jl")

function stan_cmd(sm::SampleModel, n_samples, n_warmup, stan_seed)
   cmd = `$(sm.output_base) num_threads=1 sample num_chains=1 num_samples=$n_samples`
   cmd = `$cmd num_warmup=$n_warmup save_warmup=1 adapt engaged=1`
   cmd = `$cmd algorithm=hmc engine=nuts max_depth=10`
   cmd = `$cmd id=1 data file=$(first(sm.data_file)) random seed=$stan_seed`
   cmd = `$cmd output file=$(sm.output_base)_chain_1.csv`
end

model_string(model) = read(model_path(model), String)
model_path(model) = 
   if startswith(model, "logistic_normal")
      return joinpath(base_dir(), "stan", "logistic_normal.stan")
   elseif startswith(model, "logistic_horseshoe")
      return joinpath(base_dir(), "stan", "logistic_horseshoe.stan")
   elseif startswith(model, "mvnormal")
      return joinpath(base_dir(), "stan", "mvnormal_.stan")
   else
      error("model_string: model $model unknown")
   end

# TODO: at some point, replace by calling JSON 
stan_data(model, rows_x, vec_y, d, dist_params) =
   if startswith(model, "logistic_normal")
      Dict("n" => length(vec_y), "d" => d, "x" => collect(hcat(rows_x...)'), # stan's bernoulli_logit_glm takes a matrix
      "y" => round.(Int,vec_y), "sigma" => PRIOR_SIGMA)
   elseif startswith(model, "logistic_horseshoe")
      Dict("n" => length(vec_y), "d" => d-1, "x" => (collect(hcat(rows_x...)')[:,2:end]), # remove intercept col since stan model has separate intercept
      "y" => round.(Int,vec_y))
   elseif startswith(model, "mvnormal")
      Dict("d" => d, "S" => dist_params)
   else
      error("stan_data: unknown model $model") 
   end 


# identify numbers in csv comment
function keep_num(c::AbstractChar)
   isdigit(c)|(c=='.')
end

# read stan output csv into chain, run time and adapt time
function read_stan_output(sm,model,n_samples)
   # read csv and remove the first 47 commented lines while keeping comments with timing info
   info = DataFrame(CSV.File(joinpath(sm.tmpdir, model * "_chain_1.csv"), header = 48)) 
   adapttime = parse(Float64, filter.(keep_num, info[end-3,1]))
   runtime = parse(Float64, filter.(keep_num, info[end-2,1]))
   info = DataFrame(CSV.File(joinpath(sm.tmpdir, model * "_chain_1.csv"), comment="#")) # remove comments
   chain = Chains(Array{Float64}(info),names(info)) # convert to mcmcchain
   return chain, adapttime, runtime
end
