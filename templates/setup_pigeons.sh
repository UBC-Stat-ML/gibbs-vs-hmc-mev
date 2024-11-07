#!/bin/bash


# need to do this here so that nextflow can fill in julia_env and baseDir
# also, cannot run the jl script in another process because then the container is reloaded
# so the ssh setup disappears
cat <<EOF > temp.jl
    using Pkg
    Pkg.activate("$julia_env")

    Pkg.update()
    Pkg.instantiate()
    Pkg.precompile()


EOF

julia temp.jl

