#using Revise
using Indicators, Distributions
using Plots
pyplot()

# define inputs
seed = 0
N = 1_000

# produce toy data
seed > 0 ? srand(seed) : nothing
r = rand(Normal(0, 1/252), N)
R = cumprod(1.0+r)
t = 2:N
t = [2^i for i in 1:floor(Int, log2(size(r,1)))]
beta, x, y = Indicators.hurst_exponent(r, t=t)
h = beta[end]

# generate visualizations
p1 = plot(R, xlabel="Time", ylabel="Return", color=:red, label="Cumulative Returns")
p2 = histogram(r, xlabel="Return", ylabel="Frequency", color=:orange, label="Returns Distribution")
if length(beta) == 1
    p3 = plot(x, y, xlabel="Sample Size", ylabel="Rescaled Range", color=:blue, label="Rescaled Range by Period Length")
    reg_x = [0; x[end]]
    reg_y = [0; beta[1]*x[end]]
else
    p3 = plot(x[:,2], y, xlabel="Sample Size", ylabel="Rescaled Range", color=:blue, label="Rescaled Range by Period Length")
    reg_x = [0; x[end]]
    reg_y = [beta[1], beta[2]*x[end]+beta[1]]
end
plot!(p3, reg_x, reg_y, color=:green, label="Hurst Exponent Estimation")

# aggregate the plots together
plot(p1, p2, p3, layout=(3,1))
