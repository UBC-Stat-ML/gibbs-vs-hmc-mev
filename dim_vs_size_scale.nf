include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; findDataDim; estimateMemoryMinESS } from './utils.nf'
params.dryRun = true

def variables = [
    dim: (1..10).collect{pow(2,it)},
    size: (3..10).collect{pow(2,it)},
    seed: (1..10),
    data: ["colon","leukemia","madelon","Prostate_GE","PCMAC"],
    sampler: ["CGGibbs","NUTS_Stan"],
    model: ["logistic"],
    prior: ["normal"]
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
ESS_threshold = params.dryRun ? 0 : 100

// local variables
def julia_env_dir = file("julia-environment")
def julia_depot_dir = file(".depot")
def deliv = deliverables(workflow)

workflow {
    args = crossProduct(variables, params.dryRun)
    julia_env = setupPigeons(julia_depot_dir, julia_env_dir)
    agg_path = runSimulation(julia_depot_dir, julia_env, args, dim_string) | collectCSVs
}

process runSimulation {
    memory { 15.GB }
    time { 60.hour } 
    cpus 1
    errorStrategy 'ignore'
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
        val dim_string
    output:
        tuple val(arg), path('csvs')
  script:
    template 'scale_main.jl'
}

