# LBasisPotential is not exported in InteratomicBasisPotentials.jl / basis_potentials.jl
# These functions should be removed once export issue is fixed.
mutable struct LBasisPotential
    basis
    β
    β0
end
function LBasisPotential(basis)
    return LBasisPotential(basis, zeros(length(basis)), 0.0)
end

# Learn with intercept
function learn!(lb::LBasisPotential, ds::DataSet; w_e = 1.0, w_f = 1.0, intercept = false)
    lp = PotentialLearning.LinearProblem(ds)

    @views B_train = reduce(hcat, lp.B)'
    @views dB_train = reduce(hcat, lp.dB)'
    @views e_train = lp.e
    @views f_train = reduce(vcat, lp.f)
    
    # Calculate A and b.
    if intercept
        int_col = ones(size(B_train, 1)+size(dB_train, 1))
        @views A = hcat(int_col, [B_train; dB_train])
    else
        @views A = [B_train; dB_train]
    end
    @views b = [e_train; f_train]

    # Calculate coefficients β.
    Q = Diagonal([w_e * ones(length(e_train));
                  w_f * ones(length(f_train))])
    βs = (A'*Q*A) \ (A'*Q*b)

    if intercept
        lb.β0 = βs[1]
        lb.β = βs[2:end]
    else
        lb.β =  βs
    end
end

# Auxiliary functions to compute all energies and forces as vectors (Zygote-friendly functions)

function PotentialLearning.get_all_energies(ds::DataSet, lb::LBasisPotential)
    Bs = sum.(get_values.(get_local_descriptors.(ds)))
    return lb.β0 .+ dot.(Bs, [lb.β])
end

function PotentialLearning.get_all_forces(ds::DataSet, lb::LBasisPotential)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    return vcat([lb.β0 .+  dB' * lb.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]...)
end


## Do not delete
# New learning function based on weigthed least squares ########################
#function learn!(lb::LBasisPotential, ds::DataSet; w_e = 1.0, w_f = 1.0)

#    lp = PotentialLearning.LinearProblem(ds)
#    
#    @views B_train = reduce(hcat, lp.B)'
#    @views dB_train = reduce(hcat, lp.dB)'
#    @views f_train = reduce(vcat, lp.f)
#    @views e_train = lp.e
#    
#    # Calculate A and b.
#    @views A = [B_train; dB_train]
#    @views b = [e_train; f_train]

#    # Calculate coefficients β.
#    Q = Diagonal([w_e * ones(length(e_train));
#    β = (A'*Q*A) \ (A'*Q*b)
#                  w_f * ones(length(f_train))])
#    #copyto!(lp.β, β)
#    #copyto!(lp.σe, w_e)
#    #copyto!(lp.σf, w_f)
#    copyto!(lb.β, β)
#end

#function get_all_energies(ds::DataSet)
#    return [get_values(get_energy(ds[c])) for c in 1:length(ds)]
#end

#function get_all_forces(ds::DataSet)
#    return reduce(vcat,reduce(vcat,[get_values(get_forces(ds[c]))
#                                    for c in 1:length(ds)]))
#end


