params.dryRun = true

include { crossProduct; collectCSVs; setupPigeons; findDataDim; estimateMemoryMinESS; deliverables } from './utils.nf'

def variables = [
    dim: ["nothing"], // set to nothing for full data
    size: ["nothing"], // default to nothing for real data
    seed: (1..30),
    data: ["colon","ALLAML","leukemia","Prostate_GE","BASEHOCK","madelon","RELATHE","PCMAC","arcene","GLI_85","SMK_CAN_187"],
    sampler: ["CGGibbs","NUTS_Stan"],
    model: ["logistic"],
    prior: ["horseshoe"]
]

sampler_string = [ 
    NUTS_Stan: "1",
    CGGibbs: "SliceSampler(n_passes=1)"
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
    memory { estimateMemoryMinESS(dim_string[arg.data].toInteger()*2, ESS_threshold)}
    time { 168.hour } 
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
