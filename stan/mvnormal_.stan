data {
  int<lower=1> d;                  // dimension
  matrix[d,d] S;                   // covariance matrix
}

parameters {
  vector[d] x;
}

model {
  vector[d] mu = rep_vector(0.0, d);
  x ~ multi_normal(mu, S); 
}
