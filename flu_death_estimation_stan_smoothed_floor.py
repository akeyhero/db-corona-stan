import pystan
import numpy as np

model = pystan.StanModel(model_code="""
functions {
    // smoothed version of float in order to avoid zero gradients
    real sfloor(real a) {
        return max([0, round(a) + inv_logit(20 * (a - round(a))) - 1]);
    }
}
data {
    int S_00_04;
    int S_05_17;
    int S_18_49;
    int S_50_64;
    int S_65_xx;
    real p_flu_00_04;
    real p_flu_05_17;
    real p_flu_18_49;
    real p_flu_50_64;
    real p_flu_65_xx;
}
parameters {
    // upper = S_xx + 1 so that S_xx can be 0
    real<lower=0, upper=S_00_04+1> d_00_04;
    real<lower=0, upper=S_05_17+1> d_05_17;
    real<lower=0, upper=S_18_49+1> d_18_49;
    real<lower=0, upper=S_50_64+1> d_50_64;
    real<lower=0, upper=S_65_xx+1> d_65_xx;
}
transformed parameters {
    real d;
    d = sfloor(d_00_04) + sfloor(d_05_17) + sfloor(d_18_49) + sfloor(d_50_64) + sfloor(d_65_xx);
}
model {
    p_flu_00_04 ~ beta(sfloor(d_00_04) + 1, S_00_04 - sfloor(d_00_04) + 1);
    p_flu_05_17 ~ beta(sfloor(d_05_17) + 1, S_05_17 - sfloor(d_05_17) + 1);
    p_flu_18_49 ~ beta(sfloor(d_18_49) + 1, S_18_49 - sfloor(d_18_49) + 1);
    p_flu_50_64 ~ beta(sfloor(d_50_64) + 1, S_50_64 - sfloor(d_50_64) + 1);
    p_flu_65_xx ~ beta(sfloor(d_65_xx) + 1, S_65_xx - sfloor(d_65_xx) + 1);
}
""")

data = {
    'S_00_04':   0,
    'S_05_17':   3,
    'S_18_49': 131,
    'S_50_64': 121,
    'S_65_xx': 297,
    'p_flu_00_04':   266.0 /  3633104,
    'p_flu_05_17':   211.0 /  7663310,
    'p_flu_18_49':  2450.0 / 11913203,
    'p_flu_50_64':  5676.0 /  9238038,
    'p_flu_65_xx': 25555.0 /  3073227
}

fit = model.sampling(data=data, chains=4)
print(fit)

"""
Result

WARNING:pystan:2 of 4000 iterations ended with a divergence (0.05 %).
WARNING:pystan:Try running with adapt_delta larger than 0.8 to remove the divergences.
Inference for Stan model: anon_model_792fbccb666ecf04a5fe450812fa5d39.
4 chains, each with iter=2000; warmup=1000; thin=1;
post-warmup draws per chain=1000, total post-warmup draws=4000.

          mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
d_00_04   0.44  7.0e-3   0.26   0.02   0.21   0.44   0.67   0.88   1419    1.0
d_05_17   0.43  9.3e-3   0.25   0.02   0.21   0.43   0.64   0.88    749   1.02
d_18_49    0.5  9.3e-3   0.32   0.03   0.24   0.49   0.71   1.14   1175    1.0
d_50_64   0.55    0.01   0.37   0.02   0.27   0.53   0.77   1.56   1198    1.0
d_65_xx   3.07    0.04   1.55    0.5   1.95    2.9   3.98   6.52   1475    1.0
d         2.72    0.04   1.57   0.02   1.66   2.59   3.89   6.07   1479    1.0
lp__     10.17    0.08   2.06    5.4   9.01   10.5  11.72  13.16    643    1.0
"""
