# Gibbs race MEV repo

## Prerequisite

- Install Julia 1.10
- Install Stan 2.35.0
- Set an environment variable `CMDSTAN` pointing to the directory where cmdstan is installed
- Install Java 11

Download the data using the following script:

```bash
cd data
./download.sh
```


## Replication of results

To replicate an experiment, use for example

```bash
./nextflow run data_scale_normal.nf -resume
```

The scripts correspond to the following figures:

- `data_scale_normal.nf` and `data_scale_HSP.nf` are used for Figures 6, 7, 8 and 18,
- `dim_scale_ex.nf` and `dim_scale_ex_HSP.nf` are for the Figures where an increasing number of features is subsampled,
- `dim_vs_size_scale.nf` is for Figure 5.


## Function usage

Some key functions in the Julia code:

- CGGibbs interface `cggibbs(...)`.
- Stan NUTS interface `nuts_stansample(...)`.

These two functions perform data preprocessing as well as sampling. 

For example, from the root of this repo:
```
using Pkg 
Pkg.activate("julia-environment")
Pkg.instantiate()
include("julia-environment/src/sampling_functions.jl")

const PRIOR_SIGMA = 10.0
const MINIMUM_REQUIRED_ESS = 100

seed = 1
data_name = "colon"
n_samples = 1000 # number of initial samples
requested_dim = 100
requested_size = 10


# CGGibbs sampling
a_cggibbs = cggibbs(seed,data_name,n_samples,requested_dim,requested_size;
		model="logistic",prior="normal",synthetic=false,return_chain=false)

# Stan NUTS sampling
a_nuts = nuts_stansample(seed,data_name,n_samples,requested_dim,requested_size;
		model="logistic",prior="normal",synthetic=false,return_chain=false)
```