using UnPack

include("compute_graph_state.jl")
include("GLM_utils.jl")


###############################################################################
# GLM with normal prior
# parameter vector: (β)
# CG_vals = (βx1, …, βxn)
# other_vals = ()
#
#
# GLM with horseshoe prior
# parameter vector: (β,λ,τ)
# CG_vals = (βx1, …, βxn)
# other_vals = (-∑ βi^2/(2λi^2))
###############################################################################




# Functions that generate the log potential and state update functions
# Note: by putting them outside of cggibbs(...), they act as function barriers
# helping with type stability of the closure.
function generate_log_potential(rows_x, vec_y; model="logistic",prior="normal")
    full_model = model * "_" * prior
    if full_model == "logistic_normal"
        function lp_logistic_normal(state::compute_graph_state)
            @unpack cur_θ, cur_i, pre_θ_i, other_vals, CG_vals = state
            d = length(first(rows_x))
            @inbounds begin
                res = 0.0
                # log likelihood
                δθ = cur_θ[cur_i] - pre_θ_i
                for (n,yn) in enumerate(vec_y)
                    xβ = CG_vals[n] + δθ*rows_x[n][cur_i]
                    if yn == 1
                        res -= log1pexp(-xβ)
                    else
                        res -= log1pexp(xβ)
                    end
                end

                # prior
                res -= cur_θ[cur_i]^2/(2*PRIOR_SIGMA^2)
                state.lp_count +=1
            end
            return res
        end
    elseif full_model == "logistic_horseshoe"
        function lp_logistic_horseshoe(state::compute_graph_state)
            @unpack cur_θ, cur_i, pre_θ_i, other_vals, CG_vals = state
            d = length(first(rows_x))
            @inbounds begin
                res = 0.0
                # log likelihood
                if cur_i <=d
                    δθ = cur_θ[cur_i] - pre_θ_i
                    for (n,yn) in enumerate(vec_y)
                        xβ = CG_vals[n] + δθ*rows_x[n][cur_i]
                        if yn == 1
                            res -= log1pexp(-xβ)
                        else
                            res -= log1pexp(xβ)
                        end
                    end    
                end

                # prior
                θ_val = cur_θ[cur_i]
                if cur_i == 1
                    res -= 2*log(1+θ_val^2/3)
                elseif cur_i <= d
                    res -= θ_val^2/(2*(cur_θ[cur_i+d-1]*cur_θ[end])^2)
                elseif cur_i == 2*d
                    if θ_val >= 0
                        res += other_vals[1]/θ_val^2 - (d-1)*log(θ_val^2)/2 - log(1+θ_val^2)
                    else 
                        res = -Inf
                    end                    
                else
                    if θ_val >= 0
                        res -= cur_θ[cur_i-d+1]^2/(2*(θ_val*cur_θ[end])^2) + log((θ_val*cur_θ[end])^2)/2 + log(1+θ_val^2)
                    else
                        res = -Inf 
                    end
                end
                state.lp_count +=1
            end
            return res
        end
    else
        error()
    end
end
 
function generate_state_update(rows_x; model="logistic", prior="normal")
    full_model = model * "_" * prior
    if full_model == "logistic_normal"
        function update_logistic_normal(state::compute_graph_state, cached_lp)
            @unpack cur_θ, cur_i, pre_θ_i, other_vals, CG_vals = state
            d = length(first(rows_x))
            @inbounds begin
                # CG_vals
                δθ = (cur_θ[cur_i] - pre_θ_i)
                if δθ != 0
                    for n in eachindex(state.CG_vals)
                        state.CG_vals[n] += δθ*rows_x[n][cur_i]
                    end 
                end

                # joint_lp and energy
                pre_pθ_i = -pre_θ_i^2/(2*PRIOR_SIGMA^2)
                pθ_i     = -cur_θ[cur_i]^2/(2*PRIOR_SIGMA^2)
                state.joint_lp += pθ_i - pre_pθ_i
                if cur_i <= d
                    new_energy      = cached_lp - pθ_i        
                    state.joint_lp += new_energy - state.energy  # update energy of joint_lp
                    state.energy    = new_energy
                end

                # change to next index then update its cached_lp
                state.cur_i     = 1 + (cur_i!=length(cur_θ))*cur_i
                state.cached_lp = state.energy - cur_θ[state.cur_i]^2/(2*PRIOR_SIGMA^2)
            end
        end
    elseif full_model == "logistic_horseshoe"
        function update_logistic_horseshoe(state::compute_graph_state, cached_lp)
            @unpack cur_θ, cur_i, pre_θ_i, other_vals, CG_vals = state
            d = length(first(rows_x))
            @inbounds begin
                # CG_vals
                if cur_i <= d
                    δθ = (cur_θ[cur_i] - pre_θ_i)
                    if δθ != 0
                        for n in eachindex(state.CG_vals)
                            state.CG_vals[n] += δθ*rows_x[n][cur_i]
                        end 
                    end
                end

                # other_vals
                if cur_i <= d
                    if cur_i > 1
                        state.other_vals[1] -= (cur_θ[cur_i]^2-pre_θ_i^2)/(2*cur_θ[cur_i+d-1]^2) 
                    end
                elseif (cur_i > d) & (cur_i < 2*d)
                    state.other_vals[1] -= cur_θ[cur_i-d+1]^2/2*(1/cur_θ[cur_i]^2-1/pre_θ_i^2)
                end

                # joint_lp and energy                
                if cur_i == 1
                    pre_pθ_i  = -2*log(1+pre_θ_i^2/3)
                    pθ_i      = -2*log(1+cur_θ[cur_i]^2/3)
                elseif cur_i <= d
                    pre_pθ_i  = -pre_θ_i^2/(2*(cur_θ[cur_i+d-1]*cur_θ[end])^2)
                    pθ_i      = -cur_θ[cur_i]^2/(2*(cur_θ[cur_i+d-1]*cur_θ[end])^2)
                elseif cur_i == 2*d
                    pre_pθ_i  = other_vals[1]/pre_θ_i^2 - (d-1)*log(pre_θ_i^2)/2 - log(1+pre_θ_i^2)
                    pθ_i      = other_vals[1]/cur_θ[cur_i]^2 - (d-1)*log(cur_θ[cur_i]^2)/2 - log(1+cur_θ[cur_i]^2)
                else
                    pre_pθ_i  = -cur_θ[cur_i-d+1]^2/(2*(pre_θ_i*cur_θ[end])^2) - log((pre_θ_i*cur_θ[end])^2)/2 - log(1+pre_θ_i^2)
                    pθ_i      = -cur_θ[cur_i-d+1]^2/(2*(cur_θ[cur_i]*cur_θ[end])^2) - log((cur_θ[cur_i]*cur_θ[end])^2)/2 - log(1+cur_θ[cur_i]^2)
                end
                state.joint_lp += pθ_i - pre_pθ_i
                if cur_i <= d
                    new_energy      = cached_lp - pθ_i        
                    state.joint_lp += new_energy - state.energy  # update energy of joint_lp
                    state.energy    = new_energy
                end

                # change to next index then update its cached_lp
                next_i = 1 + (cur_i!=length(cur_θ))*cur_i
                if next_i == 1
                    next_pθ_i = -2*log(1+cur_θ[next_i]^2/3)
                elseif next_i <= d
                    next_pθ_i = -cur_θ[next_i]^2/(2*(cur_θ[next_i+d-1]*cur_θ[end])^2)
                elseif next_i == 2*d
                    next_pθ_i = other_vals[1]/cur_θ[next_i]^2 - (d-1)*log(cur_θ[next_i]^2)/2 - log(1+cur_θ[next_i]^2)
                else
                    next_pθ_i = -cur_θ[next_i-d+1]^2/(2*(cur_θ[next_i]*cur_θ[end])^2) - log((cur_θ[next_i]*cur_θ[end])^2)/2 - log(1+cur_θ[next_i]^2)
                end
                state.cur_i     = next_i
                state.cached_lp = state.energy + next_pθ_i
                if next_i <= d
                    state.cached_lp = state.energy + next_pθ_i
                else
                    state.cached_lp = next_pθ_i
                end
            end
        end
    else
        error()
    end
end



###############################################################################
# state initializer for CGGibbs
###############################################################################

function init_state(d, rows_x, vec_y, rng; model="logistic", prior="normal", dist_params=nothing)
    full_model = model * "_" * prior
    if full_model == "logistic_normal"
        β  = randn(rng, d)
        βx = [dot(β,x) for x in rows_x] # n, vector of βx_i's
        θ  = copy(β)

        ll_val = ll_naive(θ, rows_x, vec_y, d, model, prior)
        other_vals = []
        cached_lp = ll_val + component_prior_naive(θ, d, 1, prior)
        joint_lp = ll_val + joint_prior_naive(θ, d, prior)
        CG = βx
        discrete_sizes = zeros(Int64,d)
    elseif full_model == "logistic_horseshoe"
        halfcauchy  = truncated(Cauchy(0, 1); lower=0)
        λ  = rand(rng,halfcauchy,d-1)
        τ  = rand(rng,halfcauchy,1)[1]
        β  = vcat(rand(rng,TDist(3),1),[rand(rng,Normal(0.0, abs(λ[i-1]*τ))) for i in 2:d])
        βx = [dot(β,x) for x in rows_x] # n, vector of βx_i's
        θ  = vcat(β,λ,τ)

        ll_val = ll_naive(θ, rows_x, vec_y, d, model, prior)
        other_vals = [sum_βλ_ratio(θ,d)]
        cached_lp = ll_val + component_prior_naive(θ, d, 1, prior)
        joint_lp = ll_val + joint_prior_naive(θ, d, prior)
        CG = βx
        discrete_sizes = zeros(Int64,2*d)
    else
        error()
    end
 
    state = compute_graph_state(θ,1,first(θ),other_vals,CG,cached_lp,ll_val,joint_lp,0,discrete_sizes)
    
    return state
end
