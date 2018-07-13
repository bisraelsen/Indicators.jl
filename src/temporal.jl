# Methods for porting Indicators.jl functions to TS objects from Temporal.jl package
function close_fun(X::TS, f::Function, flds::Vector{Symbol}; args...)
    if size(X,2) == 1
        return ts(f(X.values; args...), X.index, flds)
    elseif size(X,2) > 1 && has_close(X)
        return ts(f(cl(X).values; args...), X.index, flds)
    else
        error("Must be univariate or contain Close/Settle/Last field.")
    end
end
function hlc_fun(X::TS, f::Function, flds::Vector{Symbol}; args...)
    if size(X,2) == 3
        return ts(f(X.values; args...), X.index, flds)
    elseif size(X,2) > 3 && has_high(X) && has_low(X) && has_close(X)
        return ts(f(hlc(X).values; args...), X.index, flds)
    else
        error("Argument must have 3 columns or have High, Low, and Close/Settle/Last fields.")
    end
end
function hl_fun(X::TS, f::Function, flds::Vector{Symbol}; args...)
    if size(X,2) == 2
        return ts(f(X.values; args...), X.index, flds)
    elseif size(X,2) > 2 && has_high(X) && has_low(X)
        return ts(f(hl(X).values; args...), X.index, flds)
    else
        error("Argument must have 2 columns or have High and Low fields.")
    end
end

###### run.jl ######
function runcov{V,T}(x::TS{V,T}, y::TS{V,T}; args...)
    @assert size(x,2) == 1 && size(y,2) == 1 "Arguments x and y must both be univariate (have only one column)."
    z = [x y].values
    return ts(runcov(z[:,1], z[:,2]; args...), x.index, :RunCov)
end
function runcor{V,T}(x::TS{V,T}, y::TS{V,T}; args...)
    @assert size(x,2) == 1 && size(y,2) == 1 "Arguments x and y must both be univariate (have only one column)."
    z = [x y].values
    ts(runcor(z[:,1], z[:,2]; args...), x.index, :RunCor)
end
mode{V,T}(X::TS{V,T}) = mode(X.values)
runmean{V,T}(X::TS{V,T}; args...) = close_fun(X, runmean, [:RunMean]; args...)
runsum{V,T}(X::TS{V,T}; args...) = close_fun(X, runsum, [:RunSum]; args...)
runmad{V,T}(X::TS{V,T}; args...) = close_fun(X, runmad, [:RunMAD]; args...)
runvar{V,T}(X::TS{V,T}; args...) = close_fun(X, runvar, [:RunVar]; args...)
runmax{V,T}(X::TS{V,T}; args...) = close_fun(X, runmax, [:RunMax]; args...)
runmin{V,T}(X::TS{V,T}; args...) = close_fun(X, runmin, [:RunMin]; args...)
runsd{V,T}(X::TS{V,T}; args...) = close_fun(X, runsd, [:RunSD]; args...)
runquantile{V,T}(X::TS{V,T}; args...) = close_fun(X, runquantile, [:RunQuantile]; args...)
wilder_sum{V,T}(X::TS{V,T}; args...) = close_fun(X, wilder_sum, [:WilderSum]; args...)
runacf{V,T}(X::TS{V,T}; n::Int=10, maxlag::Int=n-3, lags::AbstractArray{Int,1}=0:maxlag, cumulative::Bool=true) =
    close_fun(X, runacf, [Symbol(i) for i in lags]; n=n, maxlag=maxlag, lags=lags, cumulative=cumulative)

##### ma.jl ######
sma{V,T}(X::TS{V,T}; args...) = close_fun(X, sma, [:SMA]; args...)
hma{V,T}(X::TS{V,T}; args...) = close_fun(X, hma, [:HMA]; args...)
mma{V,T}(X::TS{V,T}; args...) = close_fun(X, mma, [:MMA]; args...)
swma{V,T}(X::TS{V,T}; args...) = close_fun(X, swma, [:SWMA]; args...)
kama{V,T}(X::TS{V,T}; args...) = close_fun(X, kama, [:KAMA]; args...)
alma{V,T}(X::TS{V,T}; args...) = close_fun(X, alma, [:ALMA]; args...)
trima{V,T}(X::TS{V,T}; args...) = close_fun(X, trima, [:TRIMA]; args...)
wma{V,T}(X::TS{V,T}; args...) = close_fun(X, wma, [:WMA]; args...)
ema{V,T}(X::TS{V,T}; args...) = close_fun(X, ema, [:EMA]; args...)
dema{V,T}(X::TS{V,T}; args...) = close_fun(X, dema, [:DEMA]; args...)
tema{V,T}(X::TS{V,T}; args...) = close_fun(X, tema, [:TEMA]; args...)
zlema{V,T}(X::TS{V,T}; args...) = close_fun(X, zlema, [:ZLEMA]; args...)
mama{V,T}(X::TS{V,T}; args...) = close_fun(X, mama, [:MAMA,:FAMA]; args...)

##### reg.jl ######
mlr_beta{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr_beta, [:Intercept,:Slope]; args...)
mlr_slope{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr_slope, [:Slope]; args...)
mlr_intercept{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr_intercept, [:Intercept]; args...)
mlr{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr, [:MLR]; args...)
mlr_se{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr_se, [:StdErr]; args...)
mlr_ub{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr_ub, [:MLRUB]; args...)
mlr_lb{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr_lb, [:MLRLB]; args...)
mlr_bands{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr_bands, [:MLRLB,:MLR,:MLRUB]; args...)
mlr_rsq{V,T}(X::TS{V,T}; args...) = close_fun(X, mlr_rsq, [:RSquared]; args...)

##### mom.jl ######
momentum{V,T}(X::TS{V,T}; args...) = close_fun(X, momentum, [:Momentum]; args...)
roc{V,T}(X::TS{V,T}; args...) = close_fun(X, roc, [:ROC]; args...)
macd{V,T}(X::TS{V,T}; args...) = close_fun(X, macd, [:MACD,:Signal,:Histogram]; args...)
rsi{V,T}(X::TS{V,T}; args...) = close_fun(X, rsi, [:RSI]; args...)
psar{V,T}(X::TS{V,T}; args...) = hl_fun(X, psar, [:PSAR]; args...)
kst{V,T}(X::TS{V,T}; args...) = close_fun(X, kst, [:KST]; args...)
wpr{V,T}(X::TS{V,T}; args...) = hlc_fun(X, wpr, [:WPR]; args...)
adx{V,T}(X::TS{V,T}; args...) = hlc_fun(X, adx, [:DiPlus,:DiMinus,:ADX]; args...)
cci{V,T}(X::TS{V,T}; args...) = hlc_fun(X, cci, [:CCI]; args...)
stoch{V,T}(X::TS{V,T}; args...) = hlc_fun(X, stoch, [:Stochastic,:Signal]; args...)
smi{V,T}(X::TS{V,T}; args...) = hlc_fun(X, smi, [:SMI,:Signal]; args...)
donch{V,T}(X::TS{V,T}; args...) = hl_fun(X, donch, [:Low,:Mid,:High]; args...)
aroon{V,T}(X::TS{V,T}; args...) = hl_fun(X, aroon, [:AroonUp,:AroonDn,:AroonOsc]; args...)

##### vol.jl ######
bbands{V,T}(X::TS{V,T}; args...) = close_fun(X, bbands, [:LB,:MA,:UB]; args...)
tr{V,T}(X::TS{V,T}; args...) = hlc_fun(X, tr, [:TR]; args...)
atr{V,T}(X::TS{V,T}; args...) = hlc_fun(X, atr, [:ATR]; args...)
keltner{V,T}(X::TS{V,T}; args...) = hlc_fun(X, keltner, [:KeltnerLower,:KeltnerMiddle,:KeltnerUpper]; args...)

##### trendy.jl #####
maxima{V,T}(X::TS{V,T}; args...) = close_fun(X, maxima, [:Maxima]; args...)
minima{V,T}(X::TS{V,T}; args...) = close_fun(X, minima, [:Minima]; args...)
support{V,T}(X::TS{V,T}; args...) = close_fun(X, support, [:Support]; args...)
resistance{V,T}(X::TS{V,T}; args...) = close_fun(X, resistance, [:Resistance]; args...)

#### utils.jl ####
crossover(x::TS, y::TS) = ts(crossover(x.values, y.values), x.index, [:CrossOver])
crossunder(x::TS, y::TS) = ts(crossunder(x.values, y.values), x.index, [:CrossUnder])

#### chaos.jl ####
hurst(X::TS; args...) = close_fun(X, hurst, [:Hurst]; args...)
