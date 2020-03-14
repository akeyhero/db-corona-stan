import pystan
import numpy as np

model = pystan.StanModel(model_code="""
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
    d = d_00_04 + d_05_17 + d_18_49 + d_50_64 + d_65_xx;
}
model {
    p_flu_00_04 ~ beta(d_00_04 + 1, S_00_04 - d_00_04 + 1);
    p_flu_05_17 ~ beta(d_05_17 + 1, S_05_17 - d_05_17 + 1);
    p_flu_18_49 ~ beta(d_18_49 + 1, S_18_49 - d_18_49 + 1);
    p_flu_50_64 ~ beta(d_50_64 + 1, S_50_64 - d_50_64 + 1);
    p_flu_65_xx ~ beta(d_65_xx + 1, S_65_xx - d_65_xx + 1);
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

Inference for Stan model: anon_model_ddebea4cf51d23d751373eced357b19d.
4 chains, each with iter=2000; warmup=1000; thin=1;
post-warmup draws per chain=1000, total post-warmup draws=4000.

          mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
d_00_04    0.1  1.5e-3    0.1 2.2e-3   0.03   0.07   0.14   0.36   4352    1.0
d_05_17   0.11  1.5e-3   0.11 3.8e-3   0.03   0.08   0.16   0.42   5358    1.0
d_18_49   0.27  3.7e-3   0.26 7.7e-3   0.08    0.2   0.39   0.95   4965    1.0
d_50_64   0.38  4.9e-3   0.34   0.01   0.13   0.28   0.52   1.28   4818    1.0
d_65_xx   2.57    0.03   1.54   0.28   1.37   2.35   3.52   6.09   3539    1.0
d         3.43    0.03   1.61   0.98    2.2   3.22    4.4   7.04   3594    1.0
lp__      2.67    0.05   1.76  -1.74   1.72   2.99   3.96   5.09   1268    1.0
"""
