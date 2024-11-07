include("plotting_utils.jl")

###########################################################################################
# Full Data experiment
###########################################################################################

# Normal prior

real_data_plot("data_scale_normal","normal")


# Horseshoe prior

real_data_plot("data_scale_HSP","horseshoe")



###########################################################################################
# Dim scale experiment
###########################################################################################

dim_scale_res = get_summary_df("dim_scale_ex")
df = dim_scale_res
data_names = ["colon","leukemia","PCMAC","Prostate_GE","madelon","ALLAML","BASEHOCK","RELATHE"]
for data_name in data_names
    dim_scale_res_plot(df,data_name)
end


