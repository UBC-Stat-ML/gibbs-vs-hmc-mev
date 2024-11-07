using Pigeons
using StatsBase

###############################################################################
# CG Gibbs for Slice sampler
###############################################################################

# step function
function step!(explorer::Pigeons.SliceSampler, replica, log_potential, state_update)
    cached_lp = replica.state.cached_lp
    for _ in 1:explorer.n_passes
        cached_lp = slice_sample!(explorer, replica.state, log_potential, state_update, cached_lp, replica)
    end
    #replica.state.cached_lp = cached_lp
end

function slice_sample!(explorer::Pigeons.SliceSampler, state, log_potential, state_update, cached_lp, replica)
    # iterate over coordinates
    CG_state = state.cur_θ
    for c in eachindex(CG_state)
        # pre sampling updates
        pointer = Ref(CG_state, c)                   # pointer for state
        state.pre_θ_i = pointer[]                    # store previous component
        cached_lp = copy(state.cached_lp)            # update cached_lp to match lp of current index

        cached_lp = sample_coord!(explorer, replica, pointer, log_potential, cached_lp, typeof(pointer[])) # note: when state is mixed, pointer is RefArray{generic common type} for all coordinates, so can't use it to dispatch 
        
        # check we still have a healthy state
        if !isfinite(cached_lp)
            error("""Got an invalid log density after updating state at index $c:
            - log density = $cached_lp
            - state[$c]   = $(pointer[])
            Dumping full replica state:
            $(replica.state)
            """)
        end

        # post sampling updates
        state_update(state,cached_lp)                # update other stuff beside cur_θ
    end
    return cached_lp
end

# use exact sampling if component is bounded discrete
function sample_coord!(explorer::Pigeons.SliceSampler, replica, pointer, log_potential, cached_lp, ::Type)
    ds = replica.state.discrete_sizes[replica.state.cur_i]
    if ds == 0
        return Pigeons.slice_sample_coord!(explorer, replica, pointer, log_potential, cached_lp, typeof(pointer[])) # note: when state is mixed, pointer is RefArray{generic common type} for all coordinates, so can't use it to dispatch 
    else # all bounded discrete var defaults value to 0:(ds-1)
        #println("discrete size: $ds")
        state = replica.state
        rng = replica.rng
        log_p_vec = Vector{Float64}(undef, ds)
        for i in 1:ds
            pointer[] = i-1
            log_p_vec[i] = log_potential(state)
        end
        pointer[] = sample(rng,0:(ds-1), Weights(exp.(log_p_vec .- maximum(log_p_vec))))
        return log_p_vec[Int(pointer[]+1)]
    end
end


# run MCMC chain
function run!(explorer::Pigeons.SliceSampler, replica, N::Int, log_potential, state_update, warmup = 0)
    d = length(replica.state.cur_θ)
    @assert N >= warmup
    chain = Array{Float64}(undef,N-warmup,d+3)
    if warmup == 0
        chain[1,:] = vcat(replica.state.cur_θ,replica.state.energy,replica.state.joint_lp,replica.state.lp_count)
        j = 2
    else
        j = 1
    end    

    for i in 2:N
        step!(explorer, replica, log_potential, state_update)
        if i > warmup
            chain[j,:] = vcat(replica.state.cur_θ,replica.state.energy,replica.state.joint_lp,replica.state.lp_count)
            j += 1
        end
    end

    param_names = ["Parameter $i" for i in 1:d]
    push!(param_names, "energy", "lp", "lp_count")

    return Chains(chain, param_names, (internals=["energy", "lp", "lp_count"],))
end

