using MCMCChains
using DataFrames
using CSV

###############################################################################
# ESS and friends
###############################################################################

function compute_ess(chain) 
   ess_df = ess(chain)
   result = ess_df.nt.ess
   if minimum(result) < MINIMUM_REQUIRED_ESS 
       @warn "Low ESS: $(minimum(result))"
   end
   return result
end

###############################################################################
# recording stuff
###############################################################################

# get number of parameters for model+prior
function get_param_num(d, model="logistic", prior="normal")
   if (model == "logistic") || (model == "categorical") || (model == "linear")
      if prior == "horseshoe"
         return 2*d
      elseif prior == "normal"
         return d
      else
         error()
      end
   else
      error()
   end
end

function record_chain_stats(chain, method, d, chain_len) # d: number of parameters
   GC.gc() # free mem, need a lot to compute cond numbers

   @info "Computing chain statistics"

   ###############################################################################
   # Record the following:
   # total_lf_steps       : number of leapfrog steps w/ warm up
   # n_lf_steps           : number of leapfrog steps w/o warm up
   # lf_step_size         : leapfrog step size after warm up
   # total_cost_per_point : number of lp evals w/ warm up
   # n_cost_per_point     : number of lp evals w/o warm up
   # chain_mat            : input chain in Array format
   # cov_κ                : condition number of posterior covariance matrix
   # lp_ess, lp_ess2      : ESS of log potential chain
   # median_rhat(2)       : median rhat diagnostics
   # Other stuff: min, median, mean, std of the ESS vector for the params
   # ess2: second order ess (i.e. ess(chain.^2))
   ###############################################################################

   if method == "NUTS_Stan"
      total_lf_steps = sum(chain[:,5,:]) # leapfrog steps stored in 5th column
      n_lf_steps = sum(chain[(chain_len+1):end,5,:])
      lf_step_size = chain[:,3,:][end] # step sizes stored in 3nd column
      total_cost_per_point = d*(total_lf_steps+1)
      n_cost_per_point = d*(n_lf_steps+1)
      chain_mat = Array(chain.value[(chain_len+1):end,8:end]) # internals in the first 7 cols
      lp_ess = ess(chain[(chain_len+1):end,1,:])
      lp_ess2 = ess(chain[(chain_len+1):end,1,:].^2)
   else # CGGibbs
      total_lf_steps = 0
      n_lf_steps = 0
      lf_step_size = 0
      total_cost_per_point = chain[:lp_count][end]
      n_cost_per_point = chain[:lp_count][end] - chain[:lp_count][1]
      chain_mat = Array(chain.value[:,1:d]) # don't take internals
      lp_ess = ess(chain[:lp])
      lp_ess2 = ess(chain[:lp].^2)
   end
   @assert size(chain_mat)[2] == d # check for correct chain dim
   
   ## the following statistics are computed without the adaptation period
   # condition number of posterior covariance matrix
   @info "Computing condition number of sample posterior covariance"
   cov_κ = if size(chain_mat, 1) < size(chain_mat, 2)
      @info "Have less samples than dimensions, condition number is Inf"
      eltype(chain_mat)(Inf)
   else
      @info "Computing singular values of the samples matrix"
      lo, hi = extrema(abs2, svdvals!(chain_mat .- mean(chain_mat,dims=1)) ) # svdvals! overwrites the temp centered chain_mat, saves memory
      hi/lo # == cond(cov(chain_mat))
   end

   # meadian rhat diagnostics
   median_rhat = median(rhat(Chains(chain_mat)).nt.rhat)       # first order
   median_rhat2 = median(rhat(Chains(chain_mat.^2)).nt.rhat)   # second order

   # ESS
   ess_vec = ess(Chains(chain_mat)).nt.ess
   min_ess = minimum(ess_vec)
   median_ess = median(ess_vec)
   mean_ess = mean(ess_vec)
   std_ess = std(ess_vec)

   # second order ESS
   ess2_vec = ess(Chains(chain_mat.^2)).nt.ess
   min_ess2 = minimum(ess2_vec)
   median_ess2 = median(ess2_vec)
   mean_ess2 = mean(ess2_vec)
   std_ess2 = std(ess2_vec)


   # return a DataFrame
   DataFrame(
      min_ess = min_ess, median_ess = median_ess, mean_ess = mean_ess, std_ess = std_ess, 
      min_ess2 = min_ess2, median_ess2 = median_ess2, mean_ess2 = mean_ess2, std_ess2 = std_ess2, 
      median_rhat = median_rhat, median_rhat2 = median_rhat2, 
      lp_ess = lp_ess, lp_ess2 = lp_ess2, cov_κ = cov_κ,
      lf_step_size = lf_step_size, lf_steps = n_lf_steps, total_lf_steps = total_lf_steps,
      cost_per_point = n_cost_per_point, total_cost_per_point = total_cost_per_point
   )
end

###############################################################################
# loading files
###############################################################################

function base_dir()
   base_folder = dirname(dirname(Base.active_project()))
   return base_folder
end

function get_summary_df(experiment::String, folder = "aggregated")
   base_folder = base_dir()
   csv_path    = joinpath(base_folder, "deliverables", experiment, folder, "summary.csv")
   return DataFrame(CSV.File(csv_path))
end