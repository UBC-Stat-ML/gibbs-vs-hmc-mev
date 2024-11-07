using LinearAlgebra
using Random

## Mixing bounds from Roberts and Sahu, '97

# DUGS rate
function gibbs_mixing(precision::Matrix)
    Q = precision
    A = I - Diagonal(diag(Q).^(-1)) * Q
    L = tril(A)
    U = A - L 
    B = inv(I - L) * U
    return max_modulus_eig(B)
end

function sorted_spectrum(M)
    lambdas = eigvals(M)
    sort!(lambdas, by = abs)
    return lambdas
end

function max_modulus_eig(M)
    spectrum = sorted_spectrum(M)
    leading = last(spectrum)
    return abs(leading)
end

function condition_number(covariance::Matrix) 
    lambdas = sorted_spectrum(covariance) 
    return last(lambdas) / first(lambdas)
end

mixing_bound(covariance::Matrix) = exp(- 1.0 / condition_number(covariance))


## Some tests on the bounds

function random_covar(rng, dim)
    X = rand(rng, dim, dim)
    covariance = X' * X
    @assert rank(covariance) == dim
    return covariance
end

function stress_test(iters, dim, rng)
    for _ in 1:iters
        covariance = random_covar(rng, dim)   
        precision = inv(covariance) 

        gm = gibbs_mixing(precision)
        mb = mixing_bound(covariance)
        # same bound but based on the residual condition number
        rmb = mixing_bound(diag_preconditioned(covariance))
        
        @assert gm ≤ rmb
        @assert gm ≤ mb
        @show rmb, mb, rmb ≤ mb # Note: from Hird et al, not always 
    end
end

# idealized pre-conditioning
function diag_preconditioned(covariance)
    scalings = diag(covariance).^(-0.5)
    linear_map = Diagonal(scalings)
    return return linear_map * covariance * linear_map'
end

function sanity(iters, dim, rng)
    for _ in 1:iters
        covariance = random_covar(rng, dim)

        scalings = rand(rng, dim)
        linear_map = Diagonal(scalings)
        covariance2 = linear_map * covariance * linear_map' 

        @assert gibbs_mixing(covariance) ≈ gibbs_mixing(covariance2)
    end
end

rng = MersenneTwister(1)
stress_test(100, 4, rng)
sanity(100, 4, rng)



