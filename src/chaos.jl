#=
Functionality generally related to chaos theory and dynamical systems.
=#

@doc """
# Rescaled Range

Compute the rescaled range of a time series.

`rescaled_range(x::Vector{Float64})::Float64`
""" ->
function rescaled_range(X::Vector{Float64}; corrected::Bool=false)::Float64
    return (maximum(X) - minimum(X)) / std(X, corrected=false)
end


@doc """
# Hurst Exponent

Estimate the Hurst exponent of a time series.

## Arguments
* `x`: vector of time series data
* `t`: vector/range of values to use for `t` when estimating the exponent
* `k`: number of times to resample the time series for each partial time series length nᵢ
* `c`: constant number corresponding to \$C\$ in Equation 10 [here](http://www.bearcave.com/misl/misl_tech/wavelets/hurst/index.html#WaveletsAndRS)
* `intercept`: whether an x-intercept should be added when calculating the regression slope
* `seed`: random number generating seed to set before estimation routine, will skip seed setting if nonpositive value given (default)

## References
* Qian and Rasheed 2004, "Hurst Exponent and Financial Market Predictability", https://pdfs.semanticscholar.org/0816/a5a989c8d2431a6d20076d27c4295c00fb77.pdf
* Kaplan 2003, "Estimating the Hurst Exponent", http://www.bearcave.com/misl/misl_tech/wavelets/hurst/index.html#WaveletsAndRS
""" ->
function hurst_exponent(X::Vector{Float64};
                        t::AbstractVector{Int} = [2^i for i in 1:floor(Int, log(size(X,1)) / log(2))],
                        c::Float64 = 1.0,
                        intercept::Bool = true,
                        seed::Int = 0)
    if seed > 0
        srand(seed)
    end
    x = zeros(Int, length(t))  # the size of the periods
    y = zeros(length(t))*NaN  # rescaled ranges: the y values, or dependent variable
    m = mean(X)
    for (i, ti) in enumerate(t)
        Y = X[1:ti] - m
        Z = cumsum(Y)
        x[i] = ti
        y[i] = rescaled_range(Z)
    end
    # fit the poiiwer law to the data
    log_x = intercept ? [ones(length(x)) log2.(x)] : log2.(x)
    log_y = log2.(y)
    beta = log_x \ log_y
    if beta[end] < 0 || beta[end] > 1
        warn("Estimated Hurst exponent ∉ [0, 1].")
    end
    return beta, log_x, log_y
end


function hurst(X::Vector{Float64}; n::Int=10, cumulative::Bool=true, args...)
    N = size(X,1)
    out = zeros(N) * NaN
    if cumulative
        for i in n:N
            h = hurst_exponent(X[1:n]; args...)
            out[i] = h[1][end]
        end
    else
        for i in n:N
            h = hurst_exponent(X[i-n+1:i]; args...)
            out[i] = h[1][end]
        end
    end
    return out
end

function hurst(X::Matrix{Float64}; n::Int=10, cumulative::Bool=true, args...)
    N, K = size(X)
    out = zeros((N,K)) * NaN
    @inbounds for j in 1:K
        out[:,j] = hurst(X[:,j])
    end
    return out
end
