using BridgeStan
using JSON
using DataFrames 
using CairoMakie 
using AlgebraOfGraphics
using LinearAlgebra
using Statistics

include("posteriordb_targets.jl")
include("glm_targets.jl")

# function targets() 
#     result = []
    
#     for model_dir in ["pos_sample/logistic_normal"]
#         for file in readdir(model_dir)
#             if isfile("$model_dir/$file/sample.csv")
#                 @show file
#                 for prior in ["normal", "horseshoe"]
#                     push!(result, GLMTarget(file, prior))
#                 end
#             end
#         end
#     end

#     for target_id in posterior_db_list()
#         push!(result, PosteriorDBTarget(target_id))
#     end

#     return result
# end

"""
- First, compute the moments from all the samples (on the unconstrained space; everything is done in that space) 
- Form moment estimates: (1) marginal Std Dev. (2) sqrt of full covar matrix 
- Pick a subset of samples (color on the moduli plot)
    - On each, use Stan's AD to compute the Hessian matrix `hessian` at that point 
    - compute `sort(abs.(eigvals(matrix)))` on the following matrices:
        - `:original`: the original Hessian matrix 
        - `:axis_moments`: diagonal (axis-aligned) preconditioning from (1)
        - `:axis_autodiff`: diagonal (axis-aligned) with Diagonal(sqrt.(inv(-hessian)))
        - `:full_moments`: preconditioning from (2)
"""
function empirical_condition_numbers(target, samples = 1:100:1000)

    result = DataFrame(
        target = Symbol[],
        dim = Int[],
        iteration = Symbol[], 
        eigenindex = Int[], 
        axis_precond = Symbol[],
        modulus = Float64[]
    )

    unconstrained_samples = reference_sample(target, false)
    log_potential = log_pot(target) 

    # compute empirical axis-aligned preconditioner 
    scalings = std(unconstrained_samples)
    linear_map = Diagonal(scalings)

    # compute empirical covar matrix 
    full_covar = cov(unconstrained_samples)
    full_precond = sqrt(full_covar)

    for iteration in samples
        if iteration > length(unconstrained_samples)
            # when we do a dry run
            break
        end
        @show iteration
        sample = unconstrained_samples[iteration]
        hessian = log_density_hessian(log_potential.model, sample)[3]
        dim, _ = size(hessian)
        for axis_precond in [:original, :axis_moments, :axis_autodiff, :full_moments]
            try # no guarantee in general the inverse neg Hessian will have pos digonals...
                matrix = 
                    if axis_precond == :axis_moments 
                        linear_map * hessian * linear_map'
                    elseif axis_precond == :original 
                        hessian 
                    elseif axis_precond == :axis_autodiff 
                        approx_lin_map = ad_diag_precond(hessian)  
                        approx_lin_map * hessian * approx_lin_map'
                    elseif axis_precond == :full_moments
                        full_precond * hessian * full_precond'
                    else
                        error()
                    end

                moduli = sort(abs.(eigvals(matrix)))
                for eigenindex in eachindex(moduli)
                    push!(result, (; target = Symbol(target), dim, axis_precond, iteration = Symbol(iteration), eigenindex, modulus = moduli[eigenindex]))
                end
            catch e 
                @warn e
                # ... so skip if target highly non-Gaussian
            end
        end
    end

    return result
end

"""
Compute a diagonal preconditioner from a Hessian matrix computed 
via AD (Automatic Differentiation)
"""
function ad_diag_precond(hessian) 
    approx_covar = inv(-hessian)
    approx_std = sqrt.(diag(approx_covar))
    return Diagonal(approx_std)
end

function hessian_plot(target, sample_index = nothing) 
    unconstrained_samples = reference_sample(target, false)
    log_potential = log_pot(target) 
    sample = unconstrained_samples[sample_index === nothing ? end : sample_index] 
    hessian = log_density_hessian(log_potential.model, sample)[3]
    heat_plot(hessian, "$(target)_raw_hessian.png")

    # compute empirical axis-aligned preconditioner 
    scalings = std(unconstrained_samples) 
    linear_map = Diagonal(scalings)
    residual_hessian = linear_map * hessian * linear_map'

    heat_plot(residual_hessian, "$(target)_res_hessian.png")
end

function heat_plot(matrix, file)
    fig = Figure(size = (600, 400))
    Axis(fig[1, 1])
    hmap = CairoMakie.heatmap!(matrix; colormap = :Spectral_11)
    Colorbar(fig[1, 2], hmap; label = "values", width = 15, ticksize = 15, tickalign = 1)
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colgap!(fig.layout, 7)
    CairoMakie.save(file, fig)
end


function analyze(target)
    hessian_plot(target)

    result = empirical_condition_numbers(target)
    CSV.write("$(target)_moduli.csv", result)
    p = data(result) * visual(Lines, alpha = 0.5) *
        mapping(
            :eigenindex, :modulus, color = :iteration, row = :axis_precond
        ) 
    axis = (width = 225, height = 225, yscale = log10)
    fg = draw(p; axis)  
    CairoMakie.save("$target.png", fg)
end

# function analyze() 
#     for target in targets()
#         @show target
#         try
#             analyze(target)
#         catch e 
#             @warn "Skipping $target: $e"
#         end
#     end
# end


#analyze()
#t = PosteriorDBTarget("arK-arK.json")
#t = GLMTarget("colon", "horseshoe")
#analyze(t)

