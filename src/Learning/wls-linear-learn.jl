"""
function learn!(
    lp::UnivariateLinearProblem,
    ws::Vector,
    int::Bool
)

Fit energies using weighted least squares.
"""
function learn!(
    lp::UnivariateLinearProblem,
    ws::Vector,
    int::Bool
)
    @views B_train = reduce(hcat, lp.iv_data)'
    @views e_train = lp.dv_data

    # Calculate A and b.
    if int
        int_col = ones(size(B_train, 1))
        @views A = hcat(int_col, B_train)
    else
        @views A = B_train
    end
    @views b = e_train

    # Calculate coefficients β.
    Q = Diagonal(ws[1] * ones(length(e_train)))
    βs = (A'*Q*A) \ (A'*Q*b)
    
    # Update lp.
    if int
        copyto!(lp.β0, [βs[1]])
        copyto!(lp.β, βs[2:end])
    else
        copyto!(lp.β, βs)
    end
    copyto!(lp.σ, ws.^(-1))
    AtA = sum(v * v' for v in lp.iv_data)
    α = 1e-8
    copyto!(lp.Σ, Symmetric(lp.σ[1]^2 * pinv(AtA, α)))
    
end

"""
function learn!(
    lp::CovariateLinearProblem,
    ws::Vector,
    int::Bool
)

Fit energies and forces using weighted least squares.
"""
function learn!(
    lp::CovariateLinearProblem,
    ws::Vector,
    int::Bool
)
    @views B_train = reduce(hcat, lp.B)'
    @views dB_train = reduce(hcat, lp.dB)'
    @views e_train = lp.e
    @views f_train = reduce(vcat, lp.f)
    
    # Calculate A and b.
    if int
        int_col = ones(size(B_train, 1) + size(dB_train, 1))
        @views A = hcat(int_col, [B_train; dB_train])
    else
        @views A = [B_train; dB_train]
    end
    @views b = [e_train; f_train]

    # Calculate coefficients βs.
    Q = Diagonal([ws[1] * ones(length(e_train));
                  ws[2] * ones(length(f_train))])
    βs = (A'*Q*A) \ (A'*Q*b)

    # Update lp.
    if int
        copyto!(lb.β0, [βs[1]])
        copyto!(lb.β, βs[2:end])
    else
        copyto!(lb.β, βs)
    end
    copyto!(lp.σ, ws.^(-1))
    AtA = sum(v * v' for v in lp.iv_data)
    α = 1e-8
    copyto!(lp.Σ, Symmetric(lp.σ[1]^2 * pinv(AtA, α)))

end


"""
function learn!(
    lp::LinearProblem
)

Default learning problem: weighted least squares.
"""
function learn!(
    lp::LinearProblem
)
    ws, int = ones(length(lp.σ)), false
    return learn!(lp, ws, int)
end


