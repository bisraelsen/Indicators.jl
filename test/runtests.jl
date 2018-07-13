using Indicators
using Temporal
using Base.Test

const N = 1_000
const X0 = 50.0
const seed = 1

srand(seed)
x = cumsum(randn(N)) + X0
ohlc = cumsum(randn(N,4)) + X0
hlc = ohlc[:,2:4]
hl = ohlc[:,2:3]

count_nans(x) = sum(isnan.(x))

@testset "Utilities" begin
    y = x + randn(N)
    cxo = crossover(x, y)
    cxu = crossunder(x, y)
    @test any(cxo)
    @test any(cxu)
    @test !any(cxo .* cxu)  # ensure crossovers and crossunders never coincide
end

# trendy
@testset "Trendlines" begin
    tmp = resistance(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = support(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = minima(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = maxima(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
end

# moving regressions
@testset "Regressions" begin
    tmp = mlr_beta(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    @test count_nans(tmp) != N
    tmp = mlr_slope(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mlr_intercept(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mlr(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mlr_se(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mlr_ub(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mlr_lb(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mlr_bands(tmp)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    @test count_nans(tmp) != N
    tmp = mlr_rsq(x, adjusted=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mlr_rsq(x, adjusted=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
end

@testset "Running Calculations" begin
    tmp = diffn(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = diffn(ohlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == size(ohlc,2)
    @test count_nans(tmp) != N
    tmp = runmean(x, cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmean(x, cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runsum(x, cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = wilder_sum(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmad(x, cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmad(x, cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runvar(x, cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runvar(x, cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runsd(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runcov(x, x.*rand(N), cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runcov(x, x.*rand(N), cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runcor(x, x.*rand(N), cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runcor(x, x.*rand(N), cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmin(x, cumulative=true, inclusive=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmin(x, cumulative=true, inclusive=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmin(x, cumulative=false, inclusive=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmin(x, cumulative=false, inclusive=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmax(x, cumulative=true, inclusive=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmax(x, cumulative=true, inclusive=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmax(x, cumulative=false, inclusive=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = runmax(x, cumulative=false, inclusive=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mode(map(xi->round(xi), x))
    @test size(tmp, 1) == 1
    @test size(tmp, 2) == 1
    @test !isnan(tmp)
    tmp = runquantile(x, cumulative=true)
    @test !isnan(tmp[2]) && isnan(tmp[1])
    @test tmp[10] == quantile(x[1:10], 0.05)
    tmp = runquantile(x, cumulative=false)
    @test tmp[10] == quantile(x[1:10], 0.05)
    n = 20
    tmp = runacf(x, n=n, maxlag=15, cumulative=true)
    @test all(tmp[n:end,1] .== 1.0)
    tmp = runacf(x, n=n, maxlag=15, cumulative=false)
    @test all(tmp[n:end,1] .== 1.0)
end

# moving average functions
@testset "Moving Averages" begin
    tmp = sma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mama(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    @test count_nans(tmp) != N
    tmp = ema(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = wma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = hma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = trima(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = mma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = tema(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = dema(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = swma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = kama(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = alma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = zlema(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
end

# momentum function
@testset "Momentum" begin
    tmp = aroon(hl)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    @test count_nans(tmp) != N
    tmp = donch(hl)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    @test count_nans(tmp) != N
    tmp = momentum(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = roc(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = macd(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    @test count_nans(tmp) != N
    tmp = rsi(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = adx(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    @test count_nans(tmp) != N
    tmp = adx(hlc, wilder=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    @test count_nans(tmp) != N
    tmp = psar(hl)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = kst(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = wpr(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = cci(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = stoch(hlc, kind=:fast)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    @test count_nans(tmp) != N
    tmp = stoch(hlc, kind=:slow)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    @test count_nans(tmp) != N
    tmp = smi(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    @test count_nans(tmp) != N
end

# volatility functions
@testset "Volatility" begin
    tmp = bbands(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    @test count_nans(tmp) != N
    tmp = tr(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = atr(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = keltner(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    @test count_nans(tmp) != N
end

# chart patterns functions
@testset "Charting" begin
    tmp = renko(hlc, use_atr=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
    tmp = renko(hlc, use_atr=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    @test count_nans(tmp) != N
end

# chaos / signal processing / dynamical systems functionality
@testset "Chaos" begin
    X = randn(N, 5)
    H = hurst(X, cumulative = true)
    @test size(H) == size(X)
    @test all(isnan.(H[1:9,:]))
end

# ==== TEMPORAL INTERACTIONS ====
@testset "Temporal" begin
    x = TS(x)
    x.fields = [:Close]
    ohlc = TS(ohlc)
    ohlc.fields = [:Open, :High, :Low, :Close]
    hlc = ohlc[:,2:4]
    # moving regressions
    tmp = mlr_beta(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    tmp = mlr_slope(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mlr_intercept(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mlr(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mlr_se(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mlr_ub(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mlr_lb(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mlr_bands(tmp)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    tmp = mlr_rsq(x, adjusted=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mlr_rsq(x, adjusted=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    # running calculations
    tmp = runmean(x, cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmean(x, cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runsum(x, cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = wilder_sum(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmad(x, cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmad(x, cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runvar(x, cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runvar(x, cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runsd(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runcov(ohlc[:,1], ohlc[:,4], cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runcov(ohlc[:,1], ohlc[:,4], cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runcor(ohlc[:,1], ohlc[:,4], cumulative=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runcor(ohlc[:,1], ohlc[:,4], cumulative=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmin(x, cumulative=true, inclusive=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmin(x, cumulative=true, inclusive=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmin(x, cumulative=false, inclusive=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmin(x, cumulative=false, inclusive=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmax(x, cumulative=true, inclusive=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmax(x, cumulative=true, inclusive=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmax(x, cumulative=false, inclusive=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runmax(x, cumulative=false, inclusive=false)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = runquantile(x, cumulative=true)
    @test !isnan(tmp.values[2,1]) && isnan(tmp.values[1,1])
    @test tmp.values[10,1] == quantile(x.values[1:10,1], 0.05)
    tmp = runquantile(x, cumulative=false)
    @test tmp.values[10,1] == quantile(x.values[1:10,1], 0.05)
    n = 20
    tmp = runacf(x, n=n, maxlag=15, cumulative=true)
    @test all(tmp.values[n:end,1] .== 1.0)
    tmp = runacf(x, n=n, maxlag=15, cumulative=false)
    @test all(tmp.values[n:end,1] .== 1.0)
    # moving average functions
    tmp = sma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mama(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    tmp = ema(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = wma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = hma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = trima(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = mma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = tema(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = dema(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = swma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = kama(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = alma(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = zlema(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    # momentum function
    tmp = aroon(hl)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    tmp = donch(hl)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    tmp = momentum(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = roc(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = macd(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    tmp = rsi(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = adx(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    tmp = adx(hlc, wilder=true)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    tmp = psar(hl)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = kst(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = wpr(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = cci(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = stoch(hlc, kind=:fast)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    tmp = stoch(hlc, kind=:slow)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    tmp = smi(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 2
    # volatility functions
    tmp = bbands(x)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
    tmp = tr(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = atr(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 1
    tmp = keltner(hlc)
    @test size(tmp, 1) == N
    @test size(tmp, 2) == 3
end
