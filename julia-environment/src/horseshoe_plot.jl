using Distributions
using DynamicPPL
using FillArrays
using LinearAlgebra
using CairoMakie
using PairPlots
using Random
using MCMCChains
using CSV
using DataFrames

######
# Posterior plots
######

## horseshoe
full_df = CSV.read("../deliverables/pos_sample/logistic_horseshoe_colon.csv", DataFrame)
lambda_means = map(mean, eachcol(full_df[:,[c for c in names(full_df) if startswith(c, "lambda")]]))
lambda_min = argmin(lambda_means)
lambda_max = argmax(lambda_means)
selected_cols = ["tau", "lambda.$(lambda_min)", "lambda.$(lambda_max)", "beta0", "beta.$(lambda_min)", "beta.$(lambda_max)"]
df = select(full_df, selected_cols)
new_col_names = ["τ","λ₁","λ₂","α","β₁","β₂"]
rename!(df, Dict(zip(selected_cols, new_col_names)))
p = pairplot(df)
save("horseshoe_posterior.png", p)

######
# Prior plots
######

@model function HorseShoePrior(J)
    halfcauchy  = truncated(Cauchy(0, 1); lower=0)
    τ ~ halfcauchy
    λ ~ product_distribution(Fill(halfcauchy, J))
    α ~ TDist(3) # Intercept
    β ~ MvNormal(Diagonal((λ .* τ).^2)) # Coefficients 
    [τ, λ..., α, β...]
end

m = HorseShoePrior(2)
n = 100_000
α = 0.05
rng = MersenneTwister(321)
samples = collect(hcat((m(rng) for _ in 1:n)...)')
filtered_samples = mapslices(samples; dims=1) do c
    qs = quantile(c, (α, 1-α))
    filter(x -> first(qs) ≤ x ≤ last(qs), c)
end
labs = Dict(
    Symbol(1) => "τ",
    Symbol(2) => "λ₁",
    Symbol(3) => "λ₂",
    Symbol(4) => "α",
    Symbol(5) => "β₁",
    Symbol(6) => "β₂",
)
p=pairplot(filtered_samples, labels=labs)
save("horseshoe.png", p)

## Same plot but for 6d N(0, 10^2 I) for comparison
d = 6
σ = 10
samples = randn(rng, (n,d))
samples .*= σ
labs = Dict(
    Symbol(1) => "β₁",
    Symbol(2) => "β₂",
    Symbol(3) => "β₃",
    Symbol(4) => "β₄",
    Symbol(5) => "β₅",
    Symbol(6) => "β₆",
)
p=pairplot(samples, labels=labs)
save("normal.png", p)
find