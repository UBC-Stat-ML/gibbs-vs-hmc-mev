params.dryRun = false

include { crossProduct; collectCSVs; setupPigeons; findDataDim; estimateMemoryMinESS; deliverables } from './utils.nf'

def variables = [
    dim: ["nothing"], // set to nothing for full data
    size: ["nothing"], // default to nothing for real data
    seed: [1],
    data: ["colon","ALLAML","leukemia","Prostate_GE","BASEHOCK","madelon","RELATHE","PCMAC","gisette"], 
    sampler: ["NUTS_Stan"],
    model: ["logistic"],
    prior: ["normal", "horseshoe"]
]

sample_string = [
    "normal": [ 
        colon: "1000",
        ALLAML: "1000",
        leukemia: "1000",
        Prostate_GE: "1000",
        BASEHOCK: "1000",
        madelon: "1000",
        RELATHE: "1000",
        PCMAC: "1000",
        //arcene: "1000",
        gisette: "1000"
        //GLI_85: "1000",
        //SMK_CAN_187: "1000"
    ],
    "horseshoe": [
        colon: "62000",
        ALLAML: "62000",
        leukemia: "62000",
        Prostate_GE: "62000",
        //BASEHOCK: "1000",
        madelon: "57000"
        // RELATHE: "1000",
        // PCMAC: "1000",
        // arcene: "1000",
        // gisette: "1000",
        // GLI_85: "1000",
        // SMK_CAN_187: "1000"
    ]
]

dim_string = [
    colon: "2000",
    ALLAML: "7129",
    leukemia: "7070",
    Prostate_GE: "5966",
    BASEHOCK: "4862",
    madelon: "500",
    RELATHE: "4322",
    PCMAC: "3289",
    arcene: "10000",
    gisette: "5000",
    GLI_85: "22283",
    SMK_CAN_187: "19993"
]

// global variables
ESS_threshold = 0

// local variables
def julia_env_dir = file("julia-environment")
def julia_depot_dir = file(".depot")
def deliv = deliverables(workflow)

workflow {
    args = crossProduct(variables, params.dryRun)
    julia_env = setupPigeons(julia_depot_dir, julia_env_dir)
    sim = runSimulation(julia_depot_dir, julia_env, args, dim_string, sample_string) 
    report(julia_depot_dir, julia_env, sim)
    cond_numbers = glm_cond_number(julia_depot_dir, julia_env, sim)
    summary(julia_depot_dir, julia_env, cond_numbers.toList())
}

process runSimulation {
    memory { params.dryRun ? '2GB' : estimateMemoryMinESS(dim_string[arg.data].toInteger()*1.5, ESS_threshold) * Math.pow(2, task.attempt-1) }
    time { 100.hour * Math.pow(2, task.attempt-1) } 
    cpus 1
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    errorStrategy 'ignore' //{ task.attempt < 3 ? 'retry' : 'ignore' } 
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
        val dim_string
        val sample_string
    output:
        path('*.csv')
  script:
    template 'pos_sample.jl'
}

process report { 
    memory { 40.GB * Math.pow(2, task.attempt-1) }
    time { 30.hour * Math.pow(2, task.attempt-1) }
    input: 
        env JULIA_DEPOT_PATH
        path julia_env
        path csv 
    output:
        path("${csv}.diagnostics")
    errorStrategy { task.attempt < 3 ? 'retry' : 'ignore' } 
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    """
    #!/usr/bin/env julia

    using Pkg
    Pkg.activate(joinpath("$baseDir", "$julia_env")) 
    include(joinpath("$baseDir", "$julia_env", "src", "sampling_functions.jl")) # loads dependencies too    

    id = "${csv}.diagnostics"
    mkdir(id)
    df = CSV.read("$csv", DataFrame)
    trace_diagnostics(df, id)

    """
}

process glm_cond_number {
    input: 
        env JULIA_DEPOT_PATH
        path julia_env
        path csv
    debug true
    output:
        path('*.csv')
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    """
    #!/usr/bin/env julia
    
    using Pkg
    Pkg.activate(joinpath("$baseDir", "$julia_env")) 
    include(joinpath("$baseDir", "$julia_env", "src/empirical_spectra/empirical_spectra.jl"))

    sample_file = "$csv"

    split = Base.split(sample_file, "_")
    dataset_name =  Base.split(split[3], ".")[1]
    prior = split[2]
    target = GLMTarget(dataset_name, prior, "$csv") 
    analyze(target) 
    
    """
}

process summary {
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        path csvs
    debug true
    """
    ls
    """
}