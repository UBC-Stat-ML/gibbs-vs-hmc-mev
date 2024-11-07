include("sampling_functions.jl")
include("plotting_utils.jl")

###############################################################################
# CGGibbs dimensional scaling
###############################################################################

function CGGibbs_runtime_logistic(i)
    model = "logistic"
    prior = "normal"
    rng = SplittableRandom(1)
    rows_x, vec_y, data_size, d = load_and_preprocess_data("data_name";
        distribution = false, synthetic = true, model = model,
        requested_dim = 2^i-1, requested_size = 10, rng = rng)
    
    lp_fun = generate_log_potential(rows_x, vec_y; model=model, prior=prior)
    update_fun = generate_state_update(rows_x; model=model, prior=prior)


    # create other stuff
    n_samples = 1000
    state = init_state(d, rows_x, vec_y, rng; model=model, prior=prior)
    replica = Pigeons.Replica(state, 1, rng, (;), 1)
    expl = RWMH_sampler()

    return @elapsed run!(expl,replica,n_samples,lp_fun,update_fun)
end


const PRIOR_SIGMA = 10.0
max_dim = 12
exe_time_CGG = Vector{Float64}(undef, max_dim)

for i in 1:max_dim
    exe_time_CGG[i] = CGGibbs_runtime_logistic(i)
    println(exe_time_CGG[i])
end


# Values for CGGibbs are from code above
# Values for JAGS are from jags.rda using code in Jags_test.R
# Values for BUGS are from running files in BUGS folder
#
# exe_time_CGG = [0.0012328, 0.0019569, 0.0029658, 0.00534, 0.0102739, 0.0175357,
#     0.0331598, 0.0631501, 0.1354527, 0.2601604, 0.5348757, 1.0804568]
# exe_time_JAGS = [0.168, 0.216, 0.302, 0.619, 0.997, 1.975, 5.553,
#     18.921, 90.566,  332.239, 1275.424, 5028.433]
# exe_time_BUGS = [0.594, 0.609, 0.579, 0.625, 0.641, 0.906, 
#     1.968, 4.921, 13.734, 52.968, 201.25, 757.19]


# Gibbs comp scaling
lw_val = 2
max_dim = 12
plt = plot(2.0 .^ Vector(1:max_dim), exe_time_CGG, label="CG Gibbs", lw=lw_val, marker=:circle,
    xaxis=:log, yaxis=:log, legend=:topleft)
plot!(2.0 .^ Vector(1:max_dim), exe_time_BUGS,label="BUGS",lw=lw_val,ls=:dot,marker=:rect)
plot!(2.0 .^ Vector(1:max_dim), exe_time_JAGS,label="JAGS",lw=lw_val,ls=:dash,marker=:diamond)
xlabel!("Dimension")
ylabel!("Time (s) per 1000 sweeps")
plot!(2.0 .^ Vector(1:max_dim), 2.0 .^ (Vector(1:max_dim) .- 12),label="Linear rate",color=:black,ls=:dashdotdot)
plot!(2.0 .^ Vector(1:max_dim), 2.0 .^ ( 2 .* Vector(1:max_dim) .- 15),label="Quadratic rate",color=:black)
mkdir(joinpath(base_dir(), "deliverables", "dim_scale"))
savefig(plt,joinpath(base_dir(), "deliverables", "dim_scale","Gibbs_scales" * ".png"))

