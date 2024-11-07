using Turing
using LinearAlgebra
using LogExpFunctions
using Distributions
using FillArrays


function ll_naive(θ, rows_x, vec_y, d, model="logistic", prior="normal")
   ll_val = 0.0
   if model == "logistic"
      if prior == "spikeslab"
         β = θ[1:d]
         Z = vcat(1,θ[(d+1):(2*d-1)])
         βx = [dot(Z.*β,x) for x in rows_x]
      else
         β = θ[1:d]
         βx = [dot(β,x) for x in rows_x]
      end
      ll_val += sum(zip(βx, vec_y)) do (βxi, yi)
         logpdf(BernoulliLogit(βxi), yi)
      end
   elseif model == "linear"
      if prior == "spikeslab"
         β = θ[1:d]
         Z = vcat(1,θ[(d+1):(2*d-1)])
         βx = [dot(Z.*β,x) for x in rows_x]
      else
         β = θ[1:d]
         βx = [dot(β,x) for x in rows_x]
      end
      ll_val += sum(zip(βx, vec_y)) do (βxi, yi)
         logpdf(Normal(βxi,KNOWN_NOISE), yi)
      end
   end

   return ll_val
end

function component_prior_naive(θ, d, i, prior="normal")
   if prior == "normal"
      return -θ[i]^2/(2*PRIOR_SIGMA^2)
   elseif prior == "horseshoe"
      β = θ[1:d]
      λ = θ[(d+1):(end-1)]
      τ = θ[end]
      if i == 1
         return - 2*log(1+β[i]^2/3)
      elseif i <= d
         return - β[i]^2/(2*(λ[i-1]*τ)^2)
      elseif i == 2*d
         return sum_βλ_ratio(θ,d)/τ^2 - (d-1)*log(τ^2)/2 - log(1+τ^2)
      else
         return - β[i-d+1]^2/(2*(λ[i-d]*τ)^2) - log((λ[i-d]*τ)^2)/2 - log(1+λ[i-d]^2)
      end
   elseif prior == "spikeslab"
      if i == 1
         return - 2*log(1+θ[i]^2/3)
      elseif i <= d
         return - θ[i]^2/(2*PRIOR_SIGMA^2)
      else
         return log(PRIOR_OMEGA)*θ[i] + log(1-PRIOR_OMEGA)*(1-θ[i])
      end
   end
end

function joint_prior_naive(θ, d, prior="normal")
   res = 0.0
   if prior == "normal"
      res += sum(θ) do θi
         -θi^2/(2*PRIOR_SIGMA^2)
      end
   elseif prior == "horseshoe"
      β = θ[1:d]
      λ = θ[(d+1):(end-1)]
      τ = θ[end]
      res += logpdf(TDist(3),β[1]) + logpdf(Cauchy(),τ)
      res += sum(zip(β[2:d],λ)) do (βi,λi)
         logpdf(Normal(0.0, abs(λi*τ)),βi) + logpdf(Cauchy(),λi)
      end
   elseif prior == "spikeslab"
      res += sum(1:length(θ)) do i
         component_prior_naive(θ,d,i,prior)
      end
   end

   return res
end

function sum_βλ_ratio(θ,d)
   β = θ[1:d]
   λ = θ[(d+1):(end-1)]
   return sum(zip(β[2:d],λ)) do (βi,λi)
      -βi^2/(2*λi^2)
   end
end

