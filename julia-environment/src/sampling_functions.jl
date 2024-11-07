using LogDensityProblemsAD, ReverseDiff

using InferenceReport

include("compute_graph_state.jl")
include("CG_SliceSampler.jl")

include("data_utils.jl")
include("CGGibbs_functions.jl")
include("Stan_functions.jl")

# main sampling function for CGGibbs
function cggibbs(seed, data_name, n_samples, requested_dim, requested_size,
      explorer = Pigeons.SliceSampler(n_passes=1);
      distribution = false,
      synthetic = true,
      model = "logistic",
      prior = "normal",
      dist_params = nothing,
      return_chain = false,
      true_params = nothing,
      design_mat = nothing,
      random_col = true,
      random_subset = false,
      sparse_standardize = true
   )

   # load data
   rng = SplittableRandom(seed)
   rows_x, vec_y, data_size, d = load_and_preprocess_data(data_name;
      distribution, synthetic, model, true_params, design_mat,
      requested_dim, requested_size, rng, random_col, random_subset, sparse_standardize)
   n_params = get_param_num(d,model,prior)
   
   if distribution
      @info "Running CGGibbs for a $d dimensional distribution"
      @assert !isnothing(dist_params)
   else
      @info "Running CGGibbs with a dataset of size ($data_size, $d)"
   end

   # generate CGGibbs inputs
   lp_fun = generate_log_potential(rows_x, vec_y; model=model, prior=prior)
   update_fun = generate_state_update(rows_x; model=model, prior=prior)

   # initialize state and replica
   state = init_state(d, rows_x, vec_y, rng; model, prior, dist_params)
   replica = Pigeons.Replica(state, 1, rng, (;), 1)

   # timed run
   @info "Running now..."
   n_samples = max(n_samples,ceil(Int,n_params/1000)*1000) # set to satisfy chain_len > d
   chain_len = copy(n_samples)
   res = @timed run!(explorer,replica,2*n_samples,lp_fun,update_fun,chain_len)
   @info "Finished in $(res.time) seconds."
   adapttime = res.time/2
   runtime = res.time/2
   chain = res.value   
   ess_value = compute_ess(res.value) 
   while (minimum(ess_value) < MINIMUM_REQUIRED_ESS) && (chain_len < 5000000)
      n_samples = ceil(Int, n_samples/1000*max(1,MINIMUM_REQUIRED_ESS*1.5/minimum(ess_value)-1))*1000
      @info "Drawing $n_samples samples more (had chain_len=$chain_len)"
      chain_len += n_samples
      if chain_len > 2*n_samples
         res_old = res.value[(end+2*n_samples-chain_len+1):end,:,:]
         println(range(res_old))
         res = @timed run!(explorer,replica,2*n_samples,lp_fun,update_fun,max(0,2*n_samples-chain_len))
         println(range(res.value))
         chain_new = setrange(res.value, range(res_old)[end] .+ range(res.value))
         chain = vcat(res_old, chain_new)
      else
         res = @timed run!(explorer,replica,2*n_samples,lp_fun,update_fun,max(0,2*n_samples-chain_len))
         chain = res.value
      end
      @info "Finished in $(res.time) seconds."
      adapttime += res.time/2
      runtime += res.time/2
      @info "Computing ESS..."
      ess_value = compute_ess(res.value) 
   end
   @info "Sampling completed!"

   if return_chain
      return chain
   else
      # return a DataFrame with the results
      chain_stats_df = record_chain_stats(res.value, "CGGibbs", n_params, chain_len)
      return hcat(
         DataFrame(
            run_time = runtime, adapt_time = adapttime, chain_length = chain_len,
            dimension = d, data_size = data_size, n_params = n_params
         ), 
         chain_stats_df
      )
   end
end
 
 
# Use cmdStan via low-level StanSample (and StanBase) utilities
# for more information on other available cmdline options, see
#     https://github.com/StanJulia/StanSample.jl/blob/master/src/stanrun/cmdline.jl
function nuts_stansample(seed, data_name, n_samples, requested_dim, requested_size;
      distribution = false,
      synthetic = true,
      model = "logistic",
      prior = "normal",
      dist_params = nothing,
      return_chain = false,
      true_params = nothing,
      design_mat = nothing,
      random_col = true,
      random_subset = false,
      sparse_standardize = true
   )

   # load data
   rng = SplittableRandom(seed)
   rows_x, vec_y, data_size, d = load_and_preprocess_data(data_name;
      distribution, synthetic, model, true_params, design_mat,
      requested_dim, requested_size, rng, random_col, random_subset, sparse_standardize)
   n_params = get_param_num(d,model,prior)
   
   # compile the model and save data in json file
   full_model = model * "_" * prior
   sm = SampleModel(full_model, model_string(full_model))
   data = stan_data(full_model, rows_x, vec_y, d, dist_params)
   StanSample.update_json_files(sm, data, 1, "data")
   
   if distribution
      @info "Running Stan NUTS for a $d dimensional distribution"
      @assert !isnothing(dist_params)
   else
      @info "Running Stan NUTS with a dataset of size ($data_size, $d)"
   end
   
   # create cmd string
   stan_seed = "$(rand(rng, UInt16))"
   cmd = stan_cmd(sm, n_samples, n_samples, stan_seed)

   # run
   run(cmd)

   # retrieve samples and additional info
   chain_len = copy(n_samples)
   chain, adapttime, runtime = read_stan_output(sm,full_model,chain_len)
   ess_value = compute_ess(chain[(n_samples+1):end,8:end,:]) # first 7 columns are additional infos

   while minimum(ess_value) < MINIMUM_REQUIRED_ESS && (chain_len < 5000000)
      n_samples = ceil(Int, n_samples/1000*max(1,MINIMUM_REQUIRED_ESS*1.5/minimum(ess_value)))*1000
      @info "Redrawing $n_samples samples (had chain_len=$chain_len)"
      
      # run
      cmd = stan_cmd(sm, n_samples, n_samples, stan_seed) # uses the same seed so that the first half is equal to the samples we already have
      run(cmd)

      # reread output
      chain_len = copy(n_samples)
      chain, adapttime, runtime = read_stan_output(sm,full_model,chain_len)
      @info "Finished in $(adapttime + runtime) seconds."                        
      @info "Computing ESS..."
      ess_value = compute_ess(chain[(n_samples+1):end,8:end,:]) # first 7 columns are additional infos
   end    
   @info "Sampling completed!"

   if return_chain
      return chain
   else
      # return a DataFrame with the results
      chain_stats_df = record_chain_stats(chain, "NUTS_Stan", n_params, chain_len)
      return hcat(
         DataFrame(
            run_time = runtime, adapt_time = adapttime, chain_length = chain_len,
            dimension = d, data_size = data_size, n_params=n_params
         ), 
         chain_stats_df
      )
   end
end


function nuts_stan_run(seed, data_name, n_warmup, requested_dim, requested_size;
      distribution = false,
      synthetic = true,
      model = "logistic",
      prior = "normal",
      dist_params = nothing,
      n_samples = 1000
   )

   # load data
   rng = SplittableRandom(seed)
   rows_x, vec_y, data_size, d = load_and_preprocess_data(data_name;
      distribution, synthetic, model,
      requested_dim, requested_size, rng)
   if (model=="logistic") & (length(unique(vec_y))>2)
      model = "categorical"
   end
   n_params = get_param_num(d,model,prior)

   # compile the model and save data in json file
   full_model = model * "_" * prior
   sm = SampleModel(full_model, model_string(full_model))
   data = stan_data(full_model, rows_x, vec_y, d, dist_params)
   StanSample.update_json_files(sm, data, 1, "data")

   if distribution
      @info "Running Stan NUTS for a $d dimensional distribution"
      @assert !isnothing(dist_params)
   else
      @info "Running Stan NUTS with a dataset of size ($data_size, $d)"
   end

   # create cmd string
   stan_seed = "$(rand(rng, UInt16))"
   cmd = stan_cmd(sm, n_samples, n_warmup, stan_seed)

   # run
   run(cmd)

   # retrieve samples and additional info
   chain, adapttime, runtime = read_stan_output(sm,full_model,n_warmup)

   @info "Sampling completed!"

   return chain[(n_warmup+1):end,:,:]

end


function trace_diagnostics(df, path)
   subset = select(df, Not([:iteration, :chain, :stepsize__, :accept_stat__, :treedepth__, :energy__, :n_leapfrog__, :divergent__]))
   report(Chains(Array(subset), names(subset)); view = false, exec_folder = path)

   # some hist on the tree depth 
   depths_plot = histogram(df[:, :treedepth__])
   savefig(depths_plot, "$path/tree_depths.png")

   # scatter of post means and variances
   just_vars = select(subset, Not([:lp__])) 
   mtx = Array(just_vars)
   m = mean(mtx; dims = 1)[1, :]
   s = std(mtx; dims = 1)[1, :]
   p = scatter(m, s; markeralpha = 0.1)
   savefig(p, "$path/post_mean_vars.png")
end

#df = CSV.read("../deliverables/pos_sample/logistic_normal_colon.csv", DataFrame)
#trace_diagnostics(df, Pigeons.next_exec_folder())