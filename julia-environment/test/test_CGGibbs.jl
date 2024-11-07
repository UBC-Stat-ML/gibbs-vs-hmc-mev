include(joinpath(dirname(Base.active_project()), "src", "sampling_functions.jl"))
include(joinpath(dirname(Base.active_project()), "src", "plotting_utils.jl"))

const KNOWN_NOISE = 0.1
const PRIOR_SIGMA = 10.0
const PRIOR_OMEGA = 0.5
# logistic 
# normal prior ✓
# horseshoe prior ✓
@testset "CG_SliceSampler.jl" begin
    seed = 1
    rng = SplittableRandom(seed)
    rows_x, vec_y, data_size, d = load_and_preprocess_data("synthetic";
        requested_dim = 5, requested_size = 10, rng = rng)

    model = "logistic"
    prior = "normal"
    #prior = "horseshoe"
    
    println("dimensions $d")

    # Naive lp, ll and joint lp
    ll_fun = function (θ)
        return ll_naive(θ, rows_x, vec_y, d, model, prior)
    end

    lp_naive = function (θ, i)
        return ll_fun(θ)*(i<=d) + component_prior_naive(θ, d, i, prior)
        #return ll_fun(θ) + component_prior_naive(θ, d, i, prior)
    end

    joint_lp_naive = function (θ)
        return ll_fun(θ) + joint_prior_naive(θ, d, prior)
    end

    # initialize explorer
    expl = Pigeons.SliceSampler(n_passes=1)

    # generate CGGibbs functions
    lp_logistic = generate_log_potential(rows_x, vec_y; model=model,prior=prior)
    logistic_update = generate_state_update(rows_x; model=model,prior=prior)

    # test initialization
    @testset "init_state" begin
        state = init_state(d, rows_x, vec_y, rng; model=model,prior=prior)
        #@test state.other_vals ≈ [sum_βλ_ratio(state.cur_θ,d)]
        @test state.CG_vals ≈ [dot(state.cur_θ[1:d],x) for x in rows_x]
        #@test state.CG_vals ≈ [dot(vcat(1,state.cur_θ[(d+1):(2*d-1)]).*state.cur_θ[1:d],x) for x in rows_x]
        @test state.cached_lp ≈ lp_naive(state.cur_θ,state.cur_i)
        @test state.energy ≈ ll_fun(state.cur_θ)
        @test state.joint_lp ≈ joint_lp_naive(state.cur_θ)
    end

    # test each coordinate scan
    state = init_state(d, rows_x, vec_y, rng; model=model,prior=prior)
    replica = Pigeons.Replica(state, 1, rng, (;), 1)
    @testset "slice_sample_coord" begin
        state = replica.state
        CG_state = state.cur_θ
        for c in eachindex(CG_state)
            println("index")
            println(c)
            # pre sampling updates
            pointerr = Ref(CG_state, c)                   # pointer for state
            state.pre_θ_i = pointerr[]                    # store previous component
            cached_lp = copy(state.cached_lp)             # update cached_lp to match current prior
            
            @test cached_lp ≈ lp_naive(state.cur_θ,state.cur_i)

            if c>d
                println("cur_θ: $(state.cur_θ[state.cur_i])")
            end
            
            cached_lp = sample_coord!(expl, replica, pointerr, lp_logistic, cached_lp, typeof(pointerr[]))
            
            if c>d
                println("new cur_θ: $(state.cur_θ[state.cur_i])")
            end

            @test cached_lp ≈ lp_naive(state.cur_θ,state.cur_i)
            
            # post sampling updates
            logistic_update(state,cached_lp)

            #@test state.other_vals ≈ [sum_βλ_ratio(state.cur_θ,d)]
            @test state.CG_vals ≈ [dot(state.cur_θ[1:d],x) for x in rows_x]
            #@test state.CG_vals ≈ [dot(vcat(1,state.cur_θ[(d+1):(2*d-1)]).*state.cur_θ[1:d],x) for x in rows_x]
            println(state.cached_lp - lp_naive(state.cur_θ,state.cur_i))
            @test state.cached_lp ≈ lp_naive(state.cur_θ,state.cur_i)
            @test state.energy ≈ ll_fun(state.cur_θ)
            @test state.joint_lp ≈ joint_lp_naive(state.cur_θ)
        end
    end

    # test step function
    state = init_state(d, rows_x, vec_y, rng; model=model,prior=prior)
    replica = Pigeons.Replica(state, 1, rng, (;), 1)
    #expl = RWMH_sampler()
    @testset "SliceSampler step" begin
        cached_lp = step!(expl, replica, lp_logistic,logistic_update)
        state = replica.state
        #@test state.other_vals ≈ [sum_βλ_ratio(state.cur_θ,d)]
        @test state.CG_vals ≈ [dot(state.cur_θ[1:d],x) for x in rows_x]
        #@test state.CG_vals ≈ [dot(vcat(1,state.cur_θ[(d+1):(2*d-1)]).*state.cur_θ[1:d],x) for x in rows_x]
        @test state.cached_lp ≈ lp_naive(state.cur_θ,state.cur_i)
        @test state.energy ≈ ll_fun(state.cur_θ)
        @test state.joint_lp ≈ joint_lp_naive(state.cur_θ)
    end

    # run chain
    state = init_state(d, rows_x, vec_y, rng; model=model,prior=prior)
    replica = Pigeons.Replica(state, 1, rng, (;), 1)
    chn_CGGibbs = run!(expl,replica,1000,lp_logistic,logistic_update)
end

# check cggibbs ✓
# KNOWN_NOISE = 0.01
# PRIOR_SIGMA = 10
# PRIOR_OMEGA = 0.5
# MINIMUM_REQUIRED_ESS = 100
# test_chain = cggibbs(1, "synthetic", 100, 50, 100; 
#     model = "logistic", prior = "spikeslab", return_chain = false, distribution = false) #, dist_params = Diagonal([1,1,4]))

# # plot(test_chain[:lp][100:end])
# # plot(test_chain[100:end,11,:])
# a = ess(test_chain[:,1,:])


# check validity ✓
# seed = 1
# rng = SplittableRandom(seed)
# rows_x, vec_y, data_size, d = load_and_preprocess_data("synthetic";
#     requested_dim = 2, requested_size = 100, rng = rng, 
#     generated_dim = 2, generated_size = 100, true_params = [1.0, 3.0, -0.0])

# model = "logistic"
# prior = "spikeslab"
# lp_fun = generate_log_potential(rows_x, vec_y; model=model, prior=prior)
# update_fun = generate_state_update(rows_x; model=model, prior=prior)


# state = init_state(d, rows_x, vec_y, rng; model=model,prior=prior)
# replica = Pigeons.Replica(state, 1, rng, (;), 1)
# expl = Pigeons.SliceSampler(n_passes=1)
# chn_CGGibbs = run!(expl,replica,1000,lp_fun,update_fun)
