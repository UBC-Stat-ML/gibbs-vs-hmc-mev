using MAT
using SplittableRandoms
using Random
using Statistics
using StatsBase
using Plots
include("utils.jl")

# description of data: https://jundongl.github.io/scikit-feature/datasets.html
function describe_datasets(directory = "data")
   h = nothing
   plot_folder = "../deliverables/design_matrices"
   @assert isdir(plot_folder)
   foreach_mat_file(directory) do file
      data = matread(file)
      x = data["X"] 
      y = data["Y"]
      n, p = size(x)
      n_zeros = sum(x -> x == 0 ? 1 : 0, x)
      sparsity = n_zeros/n/p
      println(
         """
         $file
            n, p: $((n, p))
            labels: $(sort(unique(Int.(y))))
            label counts: $(counts(Int.(y)))
            sparsity: $sparsity

         """
      )
      h = heatmap(x)
      savefig(h, "$plot_folder/$(basename(file)).png")
   end
   return h
end

is_one(x) = Int(x == 1 ? 1 : 0)
function load_and_preprocess_data(
      data_name;
      distribution = false, # true when sampling from a distribution
      synthetic = true,
      model = "logistic",
      generated_dim = 2^15,
      generated_size = 2^8,
      requested_dim = nothing, 
      requested_size = nothing, 
      rng = SplittableRandom(1), 
      add_intercept = true,
      binarize = is_one,
      standardize = true,
      true_params = nothing,
      design_mat = nothing,
      random_col = true,
      random_subset = false,
      sparse_standardize = true
   )

   if distribution
      return [nothing], [nothing], 0, requested_dim
   else
      if synthetic
         data_sim_rng = SplittableRandoms.split(rng)
         if !isnothing(true_params)
            if !isnothing(design_mat)
               data, true_params = create_data(data_sim_rng,generated_size,generated_dim,true_params,design_mat; model = model)
            else
               data, true_params = create_data(data_sim_rng,generated_size,generated_dim,true_params; model = model)
            end
         else # no need to control design_mat when param is not controlled
            data, true_params = create_data(data_sim_rng,generated_size,generated_dim; model = model)
         end
         
      else
         data = matread(joinpath(base_dir(), "data", data_name * ".mat"))
      end

      mat_x = data["X"]
      vec_y = data["Y"][:, begin]
      original_data_size, original_data_dim = size(mat_x)

      dim_rng = SplittableRandoms.split(rng)
      if !isnothing(requested_dim)
         @assert requested_dim ≤ original_data_dim
         
         if random_col
            d_idx = randperm(dim_rng, original_data_dim)[1:requested_dim]
         else
            if random_subset
               d_idx = randperm(dim_rng, requested_dim)
            else
               d_idx = 1:requested_dim
            end
         end
         mat_x = mat_x[:, d_idx]
      end

      instances_rng = SplittableRandoms.split(rng)
      if !isnothing(requested_size)
         @assert requested_size ≤ original_data_size
         n_idx = randperm(instances_rng, original_data_size)[1:requested_size]
         mat_x = mat_x[n_idx, :]
         vec_y = vec_y[n_idx]
      end
      
      data_size, data_dim = size(mat_x)
      if standardize
         n_zeros = sum(x -> x == 0 ? 1 : 0, mat_x) 
         sparsity = n_zeros/data_size/data_dim
         for c in eachcol(mat_x)
            s = std(c)
            if s > 0
               if (sparsity > 0.85) & sparse_standardize # when sparsity is high, only scale down the none 0 entries
                  idx = (c .!= 0)
                  c[idx] .= c[idx] ./ maximum(abs.(c[idx])) # use max abs scaling so features lie in [-1,1]
               else
                  c .= (c .- mean(c)) ./ s
               end
            else
               @warn "Trying to standardize data with zero standard deviation" maxlog=1
            end
         end
      end
      if add_intercept # after standardization!
         mat_x = hcat(ones(data_size), mat_x) 
      end

      if model == "linear"
         converted_y = vec_y
      else
         # all data with 3 classes or more default to the same format -> no need to change
         # data with 2 classes have different formats (-1,1 or 1,2, or 0,1) -> change to 0,1
         if length(unique(vec_y)) < 3 
            converted_y = isnothing(binarize) ? Int.(vec_y) : binarize.(vec_y)
         else
            converted_y = Int.(vec_y)
         end
      end
      rows_x   = collect(eachrow(mat_x)) # loglik only looks at rows, so this improves efficiency
      data_size, data_dim = size(mat_x)
      @assert length(rows_x) == data_size
      @assert length(first(rows_x)) == data_dim

      return rows_x, converted_y, data_size, data_dim
   end
end


function create_data(rng, n, d, β = randn(rng, d+1), x_mat = randn(rng, n, d); model = "logistic", noise = 0.01)
   @assert length(β) == d+1
   if model == "logistic"
      intercept = fill(β[1],n)
      p_vec = 1 ./ (1 .+ exp.( .- intercept .- x_mat*β[2:end]))
      y_vec = rand(rng, n) .< p_vec
   elseif model == "linear"
      x_mat = randn(rng, n, d)
      intercept = fill(β[1],n)
      y_vec = intercept .+ x_mat*β[2:end] .+ noise .* randn(rng, n) # noise = sd of obs noise
   end

   # conform with real dataset format:
   return Dict("X" => x_mat, "Y" => y_vec), β
end

foreach_mat_file(f, directory) = 
   for file in readdir(directory)
      if endswith(file, "mat")
         f(joinpath(directory, file))
      end
   end
