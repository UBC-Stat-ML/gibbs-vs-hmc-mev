library(runjags)
library(boot)

# Model in the JAGS format
model <- "model {
    for (i in 1:N) {
      y[i] ~ dbern(p[i])
      logit(p[i]) <- sum(betax[i,])
      for (j in 1:d) {
        betax[i,j] <- beta[j] * x[i,j]
      }
    }
    for (j in 1:d) {
      beta[j] ~ dnorm(0.0,0.1)
    }
	}"

# Data and initial values in a named list format,
# with explicit control over the random number
# generator used for each chain (optional):

data_gen = function(N,d){
  beta <- rnorm(d+1)
  x <- rnorm(N*(d+1))
  x_mat <- matrix(x,N,d+1)
  x_mat[,1] <- 1
  y <- c()
  for (i in 1:N) {
    p_i <- inv.logit(sum(x_mat[i,]*beta))
    y[i] <- rbinom(1,1,p_i)
  }
  
  return(list(N = N, d = d, 
              x = structure(.Data = x, .Dim = c(N, d+1)),
              y = y))
}

data <- data_gen(10,5)

res <- run.jags(model=model, monitor=c("beta"), sample = 1000, burnin = 0, adapt = 0,
         data=data, n.chains=1, method="rjags")


set.seed(1)
max_dim = 12
exe_time = c()
for (i in 1:max_dim){
  data <- data_gen(10,2^i-1)
  exe_time[i] = system.time(results <- run.jags(model=model, monitor=c("beta"), sample = 1000, burnin = 0, adapt = 0,
                                                data=data, n.chains=1, method="rjags"))[3]
  print(exe_time[i])
}

save(exe_time, file='jags.rda')
