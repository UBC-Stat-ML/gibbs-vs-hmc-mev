include("sampling_functions.jl")
include("plotting_utils.jl")

# dimension vs data size sweep scaling
const PRIOR_SIGMA = 10.0
const MINIMUM_REQUIRED_ESS = 100

res_cggibbs_normal = get_summary_df("dim_vs_size_scale")

function plot_size_vs_dim(data_name,model)
    if model == "normal"
        df = res_cggibbs_normal
    elseif model == "horseshoe"
        df = res_cggibbs_HSP
    else
        error() 
    end
    
    df = df[df.data .== data_name,:]

    compute_median(df_group) = median(df_group.chain_length ./ df_group.median_ess2)
    compute_min(df_group) = median(df_group.chain_length ./ df_group.min_ess2)

    p_med = plot()
    xlabel!("log₂(Dimension)")
    ylabel!("log₂(Sweeps / ESS)")
    for n in unique(df.data_size)
        idx1 = (df.data_size .== n) .& (df.sampler .== "CGGibbs")
        df1 = df[idx1,:]
        df_grouped1 = combine(groupby(df1, :n_params)) do sdf
            DataFrame(median = compute_median(sdf))
        end
        d = sort(log2.(df_grouped1.n_params))
        idxs = sortperm(log2.(df_grouped1.n_params))
        med_e = log2.(df_grouped1.median)[idxs]
        plot!(d,med_e,label="n=$n",markershape = :auto)
    end
    
    savefig(p_med,joinpath(base_dir(), "deliverables", "dim_vs_size_scale", "sweeps_to_med_ESS_$(data_name)_$(model)" * ".png"))

    p_min = plot()
    xlabel!("log₂(Dimension)")
    ylabel!("log₂(Sweeps / ESS)")
    for n in unique(df.data_size)
        idx1 = (df.data_size .== n) .& (df.sampler .== "CGGibbs")
        df1 = df[idx1,:]
        df_grouped1 = combine(groupby(df1, :n_params)) do sdf
            DataFrame(min = compute_min(sdf))
        end
        d = sort(log2.(df_grouped1.n_params))
        idxs = sortperm(log2.(df_grouped1.n_params))
        min_e = log2.(df_grouped1.min)[idxs]
        plot!(d,min_e,label="n=$n",markershape = :auto)
    end

    savefig(p_min,joinpath(base_dir(), "deliverables", "dim_vs_size_scale", "sweeps_to_min_ESS_$(data_name)_$(model)" * ".png"))
end

for i in eachindex(data_names)
    plot_size_vs_dim(data_names[i],"normal")
end




####################################################################
# condition number investigation 
####################################################################

function kappa_record(data_name,d,n)
    chain = cggibbs(1,data_name,10000,d,n;
        model="logistic",prior="normal",return_chain=true,synthetic=false)
    chain_mat = Array(chain[:,1:(d+1),:])
    cor_mat = cor(chain_mat)
    std_vec = std.(eachcol(chain_mat))
    cov_mat = Diagonal(std_vec)*cor_mat*Diagonal(std_vec)
    κ       = cond(cov_mat)
    κ_std   = cond(cor_mat)

    # estimate best diag cond by Optim
    function diag_cond(v)
        D = Diagonal(v)
        return cond(D*cov_mat*D)
    end
    init_v = ones(size(cov_mat)[1])
    D_estimate = Optim.optimize(diag_cond, init_v, LBFGS(), Optim.Options(iterations = 1000, g_tol = 1e-4))
    @info "Optim info: $D_estimate"
    best_κ = Optim.minimum(D_estimate)
    
    return DataFrame(data = data_name, n_params = d+1, data_size = n, 
        raw_κ = κ, std_κ = κ_std, best_κ = best_κ)
end

data_names = ["colon","leukemia","PCMAC","Prostate_GE","madelon"]
data_sizes = [62,72,1943,102,2600]
data_dims = [2000,7070,3289,5966,500]

res_bump = kappa_record("colon",1,1)
for (id,dat) in enumerate(data_names)
    for data_size in 3:5
        for data_dim in 1:6
            if (2^data_size <= data_sizes[id]) & (2^data_dim <= data_dims[id])
                res_temp     = kappa_record(dat,2^data_dim,2^data_size)
                res_bump     = vcat(res_bump,res_temp) 
            end
        end
    end
end
res_bump = res_bump[2:end,:]
CSV.write("deliverables/dim_vs_size_scale/cond_num_scale_normal.csv", res_bump)

res_bump = DataFrame(CSV.File(joinpath(base_dir(), 
"deliverables", "dim_vs_size_scale", "cond_num_scale_normal.csv")))

function plot_cond_num_dim(data_name,data_size)
    df = res_bump
    df = df[(df.data .== data_name).&(df.data_size .== data_size),:]
    d = log2.(df.n_params)

    raw_κ = log2.(df.raw_κ)
    std_κ = log2.(df.std_κ)
    best_κ = log2.(df.best_κ)
    
    p = plot(d,raw_κ,label="raw κ",title="$data_name",markershape = :circle)
    plot!(d,std_κ,label="cor κ",markershape = :x)
    plot!(d,best_κ,label="res κ",markershape = :+)
    vline!([log2(data_size)],c=:black,label="d=n")
    xlabel!("log₂(Dimension)")
    ylabel!("log₂(Condition number)")
    savefig(p,joinpath(base_dir(), "deliverables", "dim_vs_size_scale", "cond_num_compare_$(data_name)_$(data_size)_normal" * ".png"))
end

for (id,dat) in enumerate(data_names)
    for data_size in 3:5
        plot_cond_num_dim(dat,2^data_size)
    end
end

####################################################################
# bump investigation
####################################################################

function over_param_effect(true_β,x_mat)
    res_useless_β = cggibbs(1,"useless_β",1000,2^1,20;
        model="logistic", prior="normal", return_chain=false,
        synthetic=true, true_params=true_β, design_mat=x_mat, random_col=false, random_subset=false)
    for i in 2:9
        res_temp = cggibbs(1,"useless_β",1000,2^i,20;
            model="logistic",prior="normal",return_chain=false, 
            synthetic=true, true_params=true_β, design_mat=x_mat, random_col=false, random_subset=false)
        res_useless_β = vcat(res_temp,res_useless_β)
    end
    return res_useless_β
end


rng = SplittableRandom(1)
data_rng = SplittableRandoms.split(rng)

true_β = randn(rng,2^15+1).*3
true_β0 = copy(true_β)
true_β0[31:end] .= 0.0

goodx_mat = randn(data_rng,2^8,2^15)
badx_mat = copy(goodx_mat)
nullx_mat = copy(goodx_mat)
for i in 1:(2^8)
    badx_mat[i,30:end] .= badx_mat[i,1] #+ randn(data_rng)*0.01
    nullx_mat[i,30:end] .= 0.0
end
res_β0_badx = over_param_effect(true_β0,badx_mat)
res_β0_badx[!,:seed] .= 1
res_β0_goodx = over_param_effect(true_β0,goodx_mat)
res_β0_goodx[!,:seed] .= 1
res_β_badx = over_param_effect(true_β,badx_mat)
res_β_badx[!,:seed] .= 1
res_β_goodx = over_param_effect(true_β,goodx_mat)
res_β_goodx[!,:seed] .= 1
res_β_nullx = over_param_effect(true_β,nullx_mat)
res_β_nullx[!,:seed] .= 1

for seed in 2:10
    rng = SplittableRandom(seed)
    data_rng = SplittableRandoms.split(rng)

    true_β = randn(rng,2^15+1).*3
    true_β0 = copy(true_β)
    true_β0[31:end] .= 0.0

    goodx_mat = randn(data_rng,2^8,2^15)
    badx_mat = copy(goodx_mat)
    nullx_mat = copy(goodx_mat)
    for i in 1:(2^8)
        badx_mat[i,30:end] .= badx_mat[i,1] #+ randn(data_rng)*0.01
        nullx_mat[i,30:end] .= 0.0
    end
    res_temp = over_param_effect(true_β0,badx_mat)
    res_temp[!,:seed] .= seed
    res_β0_badx = vcat(res_β0_badx,res_temp)
    res_temp = over_param_effect(true_β0,goodx_mat)
    res_temp[!,:seed] .= seed
    res_β0_goodx = vcat(res_β0_goodx,res_temp)
    res_temp = over_param_effect(true_β,badx_mat)
    res_temp[!,:seed] .= seed
    res_β_badx = vcat(res_β_badx,res_temp)
    res_temp = over_param_effect(true_β,goodx_mat)
    res_temp[!,:seed] .= seed
    res_β_goodx = vcat(res_β_goodx,res_temp)
    res_temp = over_param_effect(true_β,nullx_mat)
    res_temp[!,:seed] .= seed
    res_β_nullx = vcat(res_β_nullx,res_temp)
end

res_β0_badx[!,:label] .= "β0_badx"
res_β0_goodx[!,:label] .= "β0_goodx"
res_β_badx[!,:label] .= "β_badx"
res_β_goodx[!,:label] .= "β_goodx"
res_β_nullx[!,:label] .= "β_nullx"

res_all_β = vcat(res_β0_badx,res_β0_goodx,res_β_badx,res_β_goodx,res_β_nullx)
CSV.write("deliverables/dim_vs_size_scale/over_param_effect.csv", res_all_β)

res_all_β = DataFrame(CSV.File(joinpath(base_dir(), 
"deliverables", "dim_vs_size_scale", "over_param_effect.csv")))


compute_median(df_group) = median(df_group.chain_length ./ df_group.median_ess2)
compute_min(df_group) = median(df_group.chain_length ./ df_group.min_ess2)

df = res_all_β[res_all_β.label .== "β0_goodx",:]
df_β_grouped2 = combine(groupby(df, :n_params)) do sdf
    DataFrame(label = "Scenario (1)", 
    median_ess2 = compute_median(sdf), min_ess2 = compute_min(sdf), n_params = sdf.n_params[1])
end

df = res_all_β[res_all_β.label .== "β0_badx",:]
df_β_grouped1 = combine(groupby(df, :n_params)) do sdf
    DataFrame(label = "Scenario (2)", 
    median_ess2 = compute_median(sdf), min_ess2 = compute_min(sdf), n_params = sdf.n_params[1])
end

df = res_all_β[res_all_β.label .== "β_nullx",:]
df_β_grouped3 = combine(groupby(df, :n_params)) do sdf
    DataFrame(label = "Scenario (3)", 
    median_ess2 = compute_median(sdf), min_ess2 = compute_min(sdf), n_params = sdf.n_params[1])
end

df_β_grouped = vcat(df_β_grouped2,df_β_grouped1,df_β_grouped3)

df = df_β_grouped
d = log2.(df.n_params)
med_e = log2.(df.median_ess2)
min_e = log2.(df.min_ess2)
lab = df.label
p_min = plot(d, min_e, group=lab,markershape = :auto)
vline!([log2(20)],label=false,line=:dash,linewidth=1.5)
vline!([log2(30)],label=false,line=:dashdot,linewidth=1.5)
xlabel!("log₂(Dimension)")
ylabel!("log₂(Sweeps / min ESS)")
savefig(p_min,joinpath(base_dir(), "deliverables", "dim_vs_size_scale", "over_param_effect_min" * ".png"))

p_med = plot(d, med_e, group=lab,markershape = :auto)
vline!([log2(20)],label=false,line=:dash,linewidth=1.5)
vline!([log2(30)],label=false,line=:dashdot,linewidth=1.5)
xlabel!("log₂(Dimension)")
ylabel!("log₂(Sweeps / median ESS)")
savefig(p_med,joinpath(base_dir(), "deliverables", "dim_vs_size_scale", "over_param_effect_med" * ".png"))

