using Turing
include("sampling_functions.jl")
include("plotting_utils.jl")

# dimension vs data size sweep scaling
const PRIOR_SIGMA = 10.0
const MINIMUM_REQUIRED_ESS = 100

function inv_diag_cond(v,cov_mat)
    D = Diagonal(v)
    return cond(D^-1*cov_mat*D^-1)
end


function nuts_diag_cond_record(seed, data_name, n_samples, requested_dim, requested_size, cov_mat, cor_mat,
        explorer = Turing.NUTS(3000, 0.65, adtype = AutoReverseDiff(false)); 
        distribution = false,
        synthetic = true,
        model = "logistic",
        prior = "normal",
        dist_params = nothing,
    )

    # load data
    rng = SplittableRandom(seed)
    rows_x, vec_y, data_size, d = load_and_preprocess_data(data_name;
        distribution = distribution, synthetic = synthetic, model = model,
        requested_dim = requested_dim, requested_size = requested_size, rng = rng)
    n_params = get_param_num(d,model,prior)

    if distribution
        @info "Running Turing NUTS for a $d dimensional distribution"
        @assert !isnothing(dist_params)
    else
        @info "Running Turing NUTS with a dataset of size ($data_size, $d)"
    end
        
    # Choose parameter dimensionality and initial parameter value
    initial_θ = randn(rng, n_params)
    if model == "logistic"
        m = logistic_regression(rows_x, vec_y, PRIOR_SIGMA)
    elseif model == "mvnormal"
        m = mv_normal(dist_params)
    end

    # timed run
    @info "Running now..."
    res = sample(rng, m, explorer, n_samples; progress=false, 
        discard_adapt=false, initial_params=initial_θ, save_state=true)
    last_θ = res.info.samplerstate
    @info "Sampling completed!"
    return inv_diag_cond(last_θ.hamiltonian.metric.sqrtM⁻¹,cov_mat)
end


data_name = "Prostate_GE"
n_samples = 3000
d = 2^4
n = 2^4
res_Turing = nuts(1,data_name,n_samples,d,n;synthetic=false,return_chain=true)
chain_mat = Array(res_Turing[:,1:(d+1),:])
cor_mat = cor(chain_mat)
std_vec = std.(eachcol(chain_mat))
cov_mat = Diagonal(std_vec)*cor_mat*Diagonal(std_vec)

function obj_fun(v)
    inv_diag_cond(v,cov_mat)
end
D_estimate = Optim.optimize(obj_fun, ones(d+1), LBFGS(), Optim.Options(iterations = 1000, g_tol = 1e-4))
best_κ = Optim.minimum(D_estimate)

diag_κ_vec = Vector{Float64}(undef,30)
for i in 1:30
    diag_κ_vec[i] = nuts_diag_cond_record(1,data_name,100*i,d,n,cov_mat,cor_mat;
        synthetic = false)
end

p = plot(0:100:3000,vcat(cond(cov_mat),diag_κ_vec),markershape=:circle,label="NUTS κ")
plot!([0,3000],[cond(cor_mat),cond(cor_mat)],label="cor κ",ls=:dash)
plot!([0,3000],[cond(cov_mat),cond(cov_mat)],label="raw κ",ls=:dashdot)
plot!([0,3000],[best_κ,best_κ],label="res κ",c=:black)
xlabel!("Iteration")
ylabel!("Condition number")
savefig(p,joinpath(base_dir(), "deliverables", "diag_cond_speed_NUTS", "diag_cond_speed_d16_n16_Prostate_GE_normal" * ".png"))
