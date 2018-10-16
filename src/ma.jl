"""
    sma(x::Array{Float64}; n::Int64=10)::Array{Float64}

Simple moving average (SMA)
"""
function sma(x::Array{Float64}; n::Int64=10)::Array{Float64}
    return runmean(x, n=n, cumulative=false)
end

"""
    trima(x::Array{Float64}; n::Int64=10, ma::Function=sma, args...)::Array{Float64}


Triangular moving average (TRIMA)
"""
function trima(x::Array{Float64}; n::Int64=10, ma::Function=sma)::Array{Float64}
    return ma(ma(x, n=n), n=n)
end

"""
    wma(x::Array{Float64}; n::Int64=10, wts::Array{Float64}=collect(1:n)/sum(1:n))::Array{Float64}

Weighted moving average (WMA)
"""
function wma(x::Array{Float64}; n::Int64=10, wts::Array{Float64}=collect(1:n)/sum(1:n))
    @assert n<size(x,1) && n>0 "Argument n out of bounds"
    out = fill(NaN, size(x,1))
    @inbounds for i = n:size(x,1)
        out[i] = (wts' * x[i-n+1:i])[1]
    end
    return out
end

function first_valid(x::Array{Float64})::Int
    if !isnan(x[1])
        return 1
    else
        @inbounds for i in 2:length(x)
            if !isnan(x[i])
                return i
            end
        end
    end
    return 0
end

"""
    ema(x::Array{Float64}; n::Int64=10, alpha::Float64=2.0/(n+1.0), wilder::Bool=false)::Array{Float64}

Exponential moving average (EMA)
"""
function ema(x::Array{Float64}; n::Int64=10, alpha::Float64=2.0/(n+1), wilder::Bool=false)
    @assert n<size(x,1) && n>0 "Argument n out of bounds."
    if wilder
        alpha = 1.0/n
    end
    out = zeros(size(x))
    i = first_valid(x)
    out[1:n+i-2] .= NaN
    out[n+i-1] = mean(x[i:n+i-1])
    @inbounds for i = n+i:size(x,1)
        out[i] = alpha * (x[i] - out[i-1]) + out[i-1]
    end
    return out
end

"""
    mma(x::Array{Float64}; n::Int64=10)::Array{Float64}

Modified moving average (MMA)
"""
function mma(x::Array{Float64}; n::Int64=10)
    return ema(x, n=n, alpha=1.0/n)
end

"""
    dema(x::Array{Float64}; n::Int64=10, alpha=2.0/(n+1), wilder::Bool=false)::Array{Float64}

Double exponential moving average (DEMA)
"""
function dema(x::Array{Float64}; n::Int64=10, alpha=2.0/(n+1), wilder::Bool=false)
    return 2.0 * ema(x, n=n, alpha=alpha, wilder=wilder) -
        ema(ema(x, n=n, alpha=alpha, wilder=wilder),
            n=n, alpha=alpha, wilder=wilder)
end

"""
    tema(x::Array{Float64}; n::Int64=10, alpha=2.0/(n+1), wilder::Bool=false)::Array{Float64}

Triple exponential moving average (TEMA)
"""
function tema(x::Array{Float64}; n::Int64=10, alpha=2.0/(n+1), wilder::Bool=false)
    return 3.0 * ema(x, n=n, alpha=alpha, wilder=wilder) -
        3.0 * ema(ema(x, n=n, alpha=alpha, wilder=wilder),
                  n=n, alpha=alpha, wilder=wilder) +
        ema(ema(ema(x, n=n, alpha=alpha, wilder=wilder),
                n=n, alpha=alpha, wilder=wilder),
            n=n, alpha=alpha, wilder=wilder)
end

"""
    mama(x::Array{Float64}; fastlimit::Float64=0.5, slowlimit::Float64=0.05)::Matrix{Float64}

MESA adaptive moving average (MAMA)
"""
function mama(x::Array{Float64}; fastlimit::Float64=0.5, slowlimit::Float64=0.05)::Matrix{Float64}
    n = size(x,1)
    out = zeros(Float64, n, 2)
    #smooth = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    smooth_1 = 0.0
    smooth_2 = 0.0
    smooth_3 = 0.0
    smooth_4 = 0.0
    smooth_5 = 0.0
    smooth_6 = 0.0
    smooth_7 = 0.0
    #detrend = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    detrend_1 = 0.0
    detrend_2 = 0.0
    detrend_3 = 0.0
    detrend_4 = 0.0
    detrend_5 = 0.0
    detrend_6 = 0.0
    detrend_7 = 0.0
    #Q1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Q1_1 = 0.0
    Q1_2 = 0.0
    Q1_3 = 0.0
    Q1_4 = 0.0
    Q1_5 = 0.0
    Q1_6 = 0.0
    Q1_7 = 0.0
    #I1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    I1_1 = 0.0
    I1_2 = 0.0
    I1_3 = 0.0
    I1_4 = 0.0
    I1_5 = 0.0
    I1_6 = 0.0
    I1_7 = 0.0
    #I2 = [0.0, 0.0]
    I2_1 = 0.0
    I2_2 = 0.0
    #Q2 = [0.0, 0.0]
    Q2_1 = 0.0
    Q2_2 = 0.0
    #Re = [0.0, 0.0]
    Re_1 = 0.0
    Re_2 = 0.0
    #Im = [0.0, 0.0]
    Im_1 = 0.0
    Im_2 = 0.0
    #per = [0.0, 0.0]
    per_1 = 0.0
    per_2 = 0.0
    #sper = [0.0, 0.0]
    sper_1 = 0.0
    sper_2 = 0.0
    #phase = [0.0, 0.0]
    phase_1 = 0.0
    phase_2 = 0.0
    jI = 0.0
    jQ = 0.0
    dphase = 0.0
    alpha = 0.0
    a = 0.0962
    b = 0.5769
    c = 0.075
    d = 0.54
    @inbounds for i = 13:n
        # Smooth and detrend price movement ====================================
        smooth_7 = (4*x[i] + 3*x[i-1] + 2*x[i-2] + x[i-3]) * 0.1
        detrend_7 = (a*smooth_7+b*smooth_5-b*smooth_3-a*smooth_1) * (c*per_1+d)
        # Compute InPhase and Quandrature components ===========================
        Q1_7 = (a*detrend_7+b*detrend_5-b*detrend_3-a*detrend_1) * (c*per_1+d)
        I1_7 = detrend_4
        # Advance phase of I1 and Q1 by 90 degrees =============================
        jQ = (a*Q1_7+b*Q1_5-b*Q1_3-a*Q1_1) * (c*per_1+d)
        jI = (a*I1_7+b*I1_5-b*I1_3-a*I1_1) * (c*per_1+d)
        # Phasor addition for 3 bar averaging ==================================
        Q2_2 = Q1_7 + jI
        I2_2 = I1_7 - jQ
        # Smooth I & Q components before applying the discriminator ============
        Q2_2 = 0.2 * Q2_2 + 0.8 * Q2_1
        I2_2 = 0.2 * I2_2 + 0.8 * I2_1
        # Homodyne discriminator ===============================================
        Re_2 = I2_2 * I2_1 + Q2_2*Q2_1
        Im_2 = I2_2 * Q2_1 - Q2_2*I2_1
        Re_2 = 0.2 * Re_2 + 0.8*Re_1
        Im_2 = 0.2 * Im_2 + 0.8*Im_1
        if (Im_2 != 0.0) & (Re_2 != 0.0)
            per_2 = 360.0/atan(Im_2/Re_2)
        end
        if per_2 > 1.5 * per_1
            per_2 = 1.5*per_1
        elseif per_2 < 0.67 * per_1
            per_2 = 0.67 * per_1
        end
        if per_2 < 6.0
            per_2 = 6.0
        elseif per_2 > 50.0
            per_2 = 50.0
        end
        per_2 = 0.2*per_2 + 0.8*per_1
        sper_2 = 0.33*per_2 + 0.67*sper_1
        if I1_7 != 0.0
            phase_2 = atan(Q1_7/I1_7)
        end
        dphase = phase_1 - phase_2
        if dphase < 1.0
            dphase = 1.0
        end
        alpha = fastlimit / dphase
        if alpha < slowlimit
            alpha = slowlimit
        end
        out[i,1] = alpha*x[i] + (1.0-alpha)*out[i-1,1]
        out[i,2] = 0.5*alpha*out[i,1] + (1.0-0.5*alpha)*out[i-1,2]
        # Reset/increment array variables
        # smooth
        smooth_1 = smooth_2
        smooth_2 = smooth_3
        smooth_3 = smooth_4
        smooth_4 = smooth_5
        smooth_5 = smooth_6
        smooth_6 = smooth_7
        # detrend
        detrend_1 = detrend_2
        detrend_2 = detrend_3
        detrend_3 = detrend_4
        detrend_4 = detrend_5
        detrend_5 = detrend_6
        detrend_6 = detrend_7
        # Q1
        Q1_1 = Q1_2
        Q1_2 = Q1_3
        Q1_3 = Q1_4
        Q1_4 = Q1_5
        Q1_5 = Q1_6
        Q1_6 = Q1_7
        # I1
        I1_1 = I1_2
        I1_2 = I1_3
        I1_3 = I1_4
        I1_4 = I1_5
        I1_5 = I1_6
        I1_6 = I1_7
        # I2
        I2_1 = I2_2
        # Q2
        Q2_1 = Q2_2
        # Re
        Re_1 = Re_2
        # Im
        Im_1 = Im_2
        # per
        per_1 = per_2
        # sper
        sper_1 = sper_2
        # phase
        phase_1 = phase_2
    end
    out[1:32,:] .= NaN
    return out
end


"""
    hma(x::Array{Float64}; n::Int64=20)::Array{Float64}

Hull moving average (HMA)
"""
function hma(x::Array{Float64}; n::Int64=20)
    return wma(2 * wma(x, n=Int64(round(n/2.0))) - wma(x, n=n), n=Int64(trunc(sqrt(n))))
end

"""
Sine-weighted moving average
"""
function swma(x::Array{Float64}; n::Int64=10)
    @assert n<size(x,1) && n>0 "Argument n out of bounds."
    w = sin.(collect(1:n) * 180.0/6.0)  # numerator weights
    d = sum(w)  # denominator = sum(numerator weights)
    out = zeros(size(x))
    out[1:n-1] .= NaN
    @inbounds for i = n:size(x,1)
        out[i] = sum(w .* x[i-n+1:i]) / d
    end
    return out
end

"""
Kaufman adaptive moving average (KAMA)
"""
function kama(x::Array{Float64}; n::Int64=10, nfast::Float64=0.6667, nslow::Float64=0.0645)
    @assert n<size(x,1) && n>0 "Argument n out of bounds."
    @assert nfast>0.0 && nfast<1.0 "Argument nfast out of bounds."
    @assert nslow>0.0 && nslow<1.0 "Argument nslow out of bounds."
    dir = diffn(x, n=n)  # price direction := net change in price over past n periods
    vol = runsum(abs.(diffn(x,n=1)), n=n, cumulative=false)  # volatility/noise
    er = abs.(dir) ./ vol  # efficiency ratio
    ssc = er * (nfast-nslow) .+ nslow  # scaled smoothing constant
    sc = ssc .^ 2  # smoothing constant
    # initiliaze result variable
    out = zeros(size(x))
    i = ndims(x) > 1 ? findfirst(.!isnan.(x)).I[1] : findfirst(.!isnan.(x))
    out[1:n+i-2] .= NaN
    out[n+i-1] = mean(x[i:n+i-1])
    @inbounds for i = n+1:size(x,1)
        out[i] = out[i-1] + sc[i]*(x[i]-out[i-1])
    end
    return out
end

"""
    alma{Float64}(x::Array{Float64}; n::Int64=9, offset::Float64=0.85, sigma::Float64=6.0)::Array{Float64}

Arnaud-Legoux moving average (ALMA)
"""
function alma(x::Array{Float64}; n::Int64=9, offset::Float64=0.85, sigma::Float64=6.0)
    @assert n<size(x,1) && n>0 "Argument n out of bounds."
    @assert sigma>0.0 "Argument sigma must be greater than 0."
    @assert offset>=0.0 && offset<=1 "Argument offset must be in (0,1)."
    out = zeros(size(x))
    out[1:n-1] .= NaN
    m = floor(offset*(float(n)-1.0))
    s = float(n) / sigma
    w = exp.(-(((0.0:-1.0:-float(n)+1.0) .- m) .^ 2.0) / (2.0*s*s))
    wsum = sum(w)
    if wsum != 0.0
        w = w ./ wsum
    end
    @inbounds for i = n:length(x)
        out[i] = sum(x[i-n+1] .* w)
    end
    return out
end

function lagged(x::Array{Float64}, n::Int=1)::Array{Float64}
    if n > 0
        return [fill(NaN,n); x[1:end-n]]
    elseif n < 0
        return [x[(-n+1):end]; fill(NaN,-n)]
    else
        return x
    end
end

"""
    zlema(x::Array{Float64}; n::Int=10, ema_args...)::Array{Float64}

Zero-lag exponential moving average (ZLEMA)
"""
function zlema(x::Array{Float64}; n::Int=10, ema_args...)::Array{Float64}
    return ema(x+(x-lagged(x,round(Int, (n-1)/2.0))), n=n; ema_args...)
end
