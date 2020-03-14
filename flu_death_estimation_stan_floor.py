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
    d = floor(d_00_04) + floor(d_05_17) + floor(d_18_49) + floor(d_50_64) + floor(d_65_xx);
}
model {
    p_flu_00_04 ~ beta(floor(d_00_04) + 1, S_00_04 - floor(d_00_04) + 1);
    p_flu_05_17 ~ beta(floor(d_05_17) + 1, S_05_17 - floor(d_05_17) + 1);
    p_flu_18_49 ~ beta(floor(d_18_49) + 1, S_18_49 - floor(d_18_49) + 1);
    p_flu_50_64 ~ beta(floor(d_50_64) + 1, S_50_64 - floor(d_50_64) + 1);
    p_flu_65_xx ~ beta(floor(d_65_xx) + 1, S_65_xx - floor(d_65_xx) + 1);
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

WARNING:pystan:Rhat above 1.1 or below 0.9 indicates that the chains very likely have not mixed
WARNING:pystan:4000 of 4000 iterations saturated the maximum tree depth of 10 (100 %)
WARNING:pystan:Run again with max_treedepth larger than 10 to avoid saturation
Inference for Stan model: anon_model_2740beb8d7649b0ff97e0b05381b7406.
4 chains, each with iter=2000; warmup=1000; thin=1;
post-warmup draws per chain=1000, total post-warmup draws=4000.

          mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
d_00_04    0.4    0.06   0.26 1.7e-3   0.22    0.4   0.59   0.93     19   1.25
d_05_17   0.36    0.06    0.3 9.9e-3    0.1   0.27   0.62   0.95     26   1.08
d_18_49   0.54    0.04    0.3   0.02   0.31   0.55   0.78   0.99     48   1.08
d_50_64   0.59    0.05   0.37   0.05   0.33   0.57    0.8   1.62     61   1.05
d_65_xx   3.13    0.14   1.52   0.85   2.02   2.91   4.06   6.67    117   1.02
d         2.71    0.14   1.54    0.0    2.0    3.0    4.0    6.0    116   1.02
lp__      10.1    0.65   2.62   3.72   8.48  10.68  12.12  13.64     16   1.34
"""
