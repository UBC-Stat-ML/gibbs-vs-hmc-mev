using Plots
using Plots.PlotMeasures: px
using StatsPlots
using LinearRegression
using LaTeXStrings

include("utils.jl")

###############################################################################
# plotting utilities
###############################################################################



# dim scale experiment plot
function dim_scale_res_plot(df::DataFrame,data_name;fn_end = ".png", folder = "dim_scale_ex")
   path   = joinpath(base_dir(), "deliverables", folder)

   idx_Gibbs = (df.sampler .== "CGGibbs") .& (df.data .== data_name)
   idx_NUTS = (df.sampler .== "Stan NUTS") .& (df.data .== data_name)
   dim_vec = log2.(df.dim)
   s = df.sampler
   e = log2.(df.run_time ./ df.median_ess2 )
   e_min = log2.(df.run_time ./ df.min_ess2 )
   sw = log2.(df.chain_length ./ df.median_ess2 )
   sw_min = log2.(df.chain_length ./ df.min_ess2 )
   tpsw = log2.(df.run_time ./ df.chain_length)

   pe = boxplot(dim_vec[idx_Gibbs], e[idx_Gibbs], group = s[idx_Gibbs], bar_width = 0.5, 
      legend=:bottomright, title="$data_name")
   boxplot!(dim_vec[idx_NUTS], e[idx_NUTS], group = s[idx_NUTS], bar_width = 0.5)
   xlabel!("log₂(Dimension)")
   ylabel!("log₂(Time (s) / ESS)")
   savefig(pe,joinpath(path, "dim_scale_medESS_$(data_name)" * fn_end))
   
   pe_min = boxplot(dim_vec[idx_Gibbs], e_min[idx_Gibbs], group = s[idx_Gibbs], bar_width = 0.5, 
      legend=:bottomright, title="$data_name")
   boxplot!(dim_vec[idx_NUTS], e_min[idx_NUTS], group = s[idx_NUTS], bar_width = 0.5)
   xlabel!("log₂(Dimension)")
   ylabel!("log₂(Time (s) / ESS)")
   savefig(pe_min,joinpath(path, "dim_scale_minESS_$(data_name)" * fn_end))

   psw = boxplot(dim_vec[idx_Gibbs], sw[idx_Gibbs], group = s[idx_Gibbs], bar_width = 0.5, 
      legend=:topleft, title="$data_name")
   boxplot!(dim_vec[idx_NUTS], sw[idx_NUTS], group = s[idx_NUTS], bar_width = 0.5)
   xlabel!("log₂(Dimension)")
   ylabel!("log₂(Sweeps / ESS)")
   vline!([log2(df.data_size[idx_Gibbs][1])],c=:black,label="d=n")
   savefig(psw,joinpath(path, "sweep_scale_medESS_$(data_name)" * fn_end))

   psw_min = boxplot(dim_vec[idx_Gibbs], sw_min[idx_Gibbs], group = s[idx_Gibbs], bar_width = 0.5, 
      legend=:topleft, title="$data_name")
   boxplot!(dim_vec[idx_NUTS], sw_min[idx_NUTS], group = s[idx_NUTS], bar_width = 0.5)
   xlabel!("log₂(Dimension)")
   ylabel!("log₂(Sweeps / ESS)")
   vline!([log2(df.data_size[idx_Gibbs][1])],c=:black,label="d=n")
   savefig(psw_min,joinpath(path, "sweep_scale_minESS_$(data_name)" * fn_end))

   ptpsw = boxplot(dim_vec[idx_Gibbs], tpsw[idx_Gibbs], group = s[idx_Gibbs], bar_width = 0.5, 
      legend=:bottomright, title="$data_name")
   boxplot!(dim_vec[idx_NUTS], tpsw[idx_NUTS], group = s[idx_NUTS], bar_width = 0.5)
   xlabel!("log₂(Dimension)")
   ylabel!("log₂(Time (s) / sweep)")
   savefig(ptpsw,joinpath(path, "sweep_time_scale_$(data_name)" * fn_end))

end

# real data reformat for plot
function real_data_reformat_for_plot(folder = "data_scale_normal")
   path   = joinpath(base_dir(), "deliverables", folder, "aggregated", "summary.csv")
   df     = CSV.read(path, DataFrame)

   df_cggibbs = df[df.sampler .== "CGGibbs"]
   df_nuts = df[df.sampler .== "NUTS_Stan"]

   compute_median(df_group) = median(df_group.run_time ./ df_group.median_ess2 .* 100)
   compute_min(df_group) = median(df_group.run_time ./ df_group.min_ess2 .* 100)

   df_nuts_grouped = combine(groupby(df_nuts[idx,:], :data)) do sdf
      DataFrame(data = sdf.data[1], nuts_median = compute_median(sdf), nuts_min = compute_min(sdf), 
      n_params = sdf.n_params[1], data_size = sdf.data_size[1])
   end

   df_cggibbs_grouped = combine(groupby(df_cggibbs, :data)) do sdf
      DataFrame(data = sdf.data[1], cggibbs_median = compute_median(sdf),cggibbs_min = compute_min(sdf))
   end

   df_combined = outerjoin(df_nuts_grouped, df_cggibbs_grouped, on = :data) # keep missing values (outer join)
   df_combined.nuts_median[ismissing.(df_combined.nuts_median)] .= (3*24*3600*1.2) # set to take more than 3 days to get 100 ESS
   df_combined.nuts_min[ismissing.(df_combined.nuts_min)] .= (3*24*3600*1.2)
   df_combined.cggibbs_median[ismissing.(df_combined.cggibbs_median)] .= (3*24*3600*1.2) # set to take more than 3 days to get 100 ESS
   df_combined.cggibbs_min[ismissing.(df_combined.cggibbs_min)] .= (3*24*3600*1.2)
   med_e_nuts = float.(df_combined.nuts_median)
   med_e_cggibbs = float.(df_combined.cggibbs_median)
   min_e_nuts = float.(df_combined.nuts_min)
   min_e_cggibbs = float.(df_combined.cggibbs_min)
   dbyn = df_combined.n_params ./ df_combined.data_size
   data_names = df_combined.data

   return med_e_nuts, med_e_cggibbs, min_e_nuts, min_e_cggibbs, dbyn, data_names
end

# real data plot
function real_data_plot(folder = "data_scale_normal", model = "normal")
   medn,medc,minn,minc,dbyn,z = real_data_reformat_for_plot(folder)

   x = medn
   y = medc
   p_med = plot(x, y, seriestype = :scatter,
        xlabel = "NUTS", ylabel = "CGGibbs", 
        aspect_ratio = 1,
        xaxis = :log, yaxis = :log)
   xlim = extrema(skipmissing(x))
   ylim = extrema(skipmissing(y))
   pad = 10^0.2
   lim_min = min(xlim[1], ylim[1]) / pad
   lim_max = max(xlim[2], ylim[2]) * pad
   plot!([10^-10, 10^10], [10^-10, 10^10], line=:solid, legend=false, c=:2) # add a 1:1 line for reference
   vline!([3*24*3600], line=:dash, legend=false, c=:red)
   hline!([3*24*3600], line=:dash, legend=false, c=:red)
   xlims!((lim_min, 3*24*3600 * pad)) # enforce 1:1 ratio of axis limits
   ylims!((lim_min, 3*24*3600 * pad))
   savefig(p_med,joinpath(base_dir(), "deliverables", "data_scale_normal", "Time_to_100medESS_real_data_$(model)" * ".png"))

   x = minn
   y = minc
   p_min = plot(x, y, seriestype = :scatter,
        xlabel = "NUTS", ylabel = "CGGibbs", 
        aspect_ratio = 1,
        xaxis = :log, yaxis = :log)
   xlim = extrema(skipmissing(x))
   ylim = extrema(skipmissing(y))
   pad = 10^0.2
   lim_min = min(xlim[1], ylim[1]) / pad
   lim_max = max(xlim[2], ylim[2]) * pad
   plot!([10^-10, 10^10], [10^-10, 10^10], line=:solid, legend=false, c=:2) # add a 1:1 line for reference
   vline!([3*24*3600], line=:dash, legend=false, c=:red)
   hline!([3*24*3600], line=:dash, legend=false, c=:red)
   xlims!((lim_min, 3*24*3600 * pad)) # enforce 1:1 ratio of axis limits
   ylims!((lim_min, 3*24*3600 * pad))
   savefig(p_min,joinpath(base_dir(), "deliverables", "data_scale_normal", "Time_to_100minESS_real_data_$(model)" * ".png"))
end