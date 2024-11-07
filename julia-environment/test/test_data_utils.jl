include(joinpath(dirname(Base.active_project()), "src", "data_utils.jl"))

# describe_datasets()

function test_read_data(name,syn)
   @show name

   # test subsetting
   mat_x, vec_y = load_and_preprocess_data(name; 
   requested_size = 3, requested_dim = 5, synthetic = syn, standardize = false)
   @assert size(mat_x) == (3,)
   @assert size(mat_x[1]) == (5+1,)
   sub_x, sub_y = load_and_preprocess_data(name; 
   requested_size = 2, requested_dim = 3, synthetic = syn, standardize = false)
   # only works if outputs are not standardized
   @assert collect(hcat(mat_x...)')[1:2, 1:4] == collect(hcat(sub_x...)') 

   load_and_preprocess_data(name; synthetic = syn)
end

test_read_data("synthetic",true)

test_read_data("colon",false)