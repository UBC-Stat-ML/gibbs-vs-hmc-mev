""" State used in compute graph Gibbs. A compute_graph_state contains: """
mutable struct compute_graph_state{S, R, U, T}
    """ Current state. """
    cur_θ::S

    """ Current component index. """
    cur_i::Int64

    """ Previous component value. """
    pre_θ_i::R

    """ Other values (aiding in fetching log potentials). """
    other_vals::U

    """ Values aiding computation using previous state. """
    CG_vals::T

    """ Cached log potential of next component. """
    cached_lp::Float64

    """ Current energy (log likelihood). """
    energy::Float64

    """ Current joint log potential. """
    joint_lp::Float64

    """ Log potential evaluation count. """
    lp_count::Int64

    """ Vector of support sizes for discrete components (0 means the component is not discrete). """
    discrete_sizes::Vector{Int64}
end

