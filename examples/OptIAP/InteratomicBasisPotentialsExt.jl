# Add following function to IBS.jl
function InteratomicBasisPotentials.ACE(species, body_order, polynomial_degree,
                                        wL, csp, r0, rcutoff)
    return ACE(species = species, body_order = body_order,
               polynomial_degree = polynomial_degree, 
               wL = wL, csp = csp, r0 = r0, rcutoff = rcutoff)
end

