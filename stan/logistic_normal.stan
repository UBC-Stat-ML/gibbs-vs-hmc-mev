data {
  int<lower=0> n;                  // Number of observations
  int<lower=0> d;                  // Number of predictors
  matrix[n,d] x;                   // design matrix
  array[n] int<lower=0,upper=1> y; // outputs
  real<lower=0> sigma;             // prior std dev for coefficients
}

parameters {
  vector[d] beta;                  // slopes
}

model {
  beta ~ normal(0, sigma); 
  y    ~ bernoulli_logit_glm(x, 0, beta);
}
