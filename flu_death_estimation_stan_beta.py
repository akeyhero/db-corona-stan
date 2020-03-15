import pystan
import matplotlib.pyplot as plt
from data import data

model = pystan.StanModel(model_code="""
data {
    int S_0x;
    int S_1x;
    int S_2x;
    int S_3x;
    int S_4x;
    int S_5x;
    int S_6x;
    int S_7x;
    int S_8x;
    int S_9x;
    real p_flu_0x;
    real p_flu_1x;
    real p_flu_2x;
    real p_flu_3x;
    real p_flu_4x;
    real p_flu_5x;
    real p_flu_6x;
    real p_flu_7x;
    real p_flu_8x;
    real p_flu_9x;
}
parameters {
    // upper = S_xx + 1 so that S_xx can be 0
    real<lower=0, upper=S_0x+1> d_0x;
    real<lower=0, upper=S_1x+1> d_1x;
    real<lower=0, upper=S_2x+1> d_2x;
    real<lower=0, upper=S_3x+1> d_3x;
    real<lower=0, upper=S_4x+1> d_4x;
    real<lower=0, upper=S_5x+1> d_5x;
    real<lower=0, upper=S_6x+1> d_6x;
    real<lower=0, upper=S_7x+1> d_7x;
    real<lower=0, upper=S_8x+1> d_8x;
    real<lower=0, upper=S_9x+1> d_9x;
}
transformed parameters {
    real d;
    d = d_0x + d_1x + d_2x + d_3x + d_4x + d_5x + d_6x + d_7x + d_8x + d_9x;
}
model {
    p_flu_0x ~ beta(d_0x + 1, S_0x - d_0x + 1);
    p_flu_1x ~ beta(d_1x + 1, S_1x - d_1x + 1);
    p_flu_2x ~ beta(d_2x + 1, S_2x - d_2x + 1);
    p_flu_3x ~ beta(d_3x + 1, S_3x - d_3x + 1);
    p_flu_4x ~ beta(d_4x + 1, S_4x - d_4x + 1);
    p_flu_5x ~ beta(d_5x + 1, S_5x - d_5x + 1);
    p_flu_6x ~ beta(d_6x + 1, S_6x - d_6x + 1);
    p_flu_7x ~ beta(d_7x + 1, S_7x - d_7x + 1);
    p_flu_8x ~ beta(d_8x + 1, S_8x - d_8x + 1);
    p_flu_9x ~ beta(d_9x + 1, S_9x - d_9x + 1);
}
""")

fit = model.sampling(data=data, chains=4)
print(fit)

fit.plot()
plt.show()

"""
Result

       mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
d_0x    0.1  1.4e-3   0.09 2.3e-3   0.03   0.07   0.14   0.35   4361    1.0
d_1x   0.12  1.7e-3   0.12 2.5e-3   0.03   0.08   0.16   0.43   4474    1.0
d_2x   0.19  2.6e-3   0.18 5.0e-3   0.06   0.14   0.27   0.67   4976    1.0
d_3x    0.2  3.1e-3   0.19 4.4e-3   0.06   0.14   0.27   0.71   3641    1.0
d_4x   0.19  2.5e-3   0.19 3.5e-3   0.05   0.13   0.26   0.69   5315    1.0
d_5x   0.25  3.2e-3   0.23 8.0e-3   0.08   0.18   0.34   0.85   5100    1.0
d_6x   0.93    0.01   0.73   0.04   0.36   0.77   1.33    2.8   4387    1.0
d_7x   1.06    0.01    0.8   0.04   0.43    0.9   1.51   3.01   4028    1.0
d_8x   0.54  7.0e-3   0.48   0.01   0.18   0.42   0.76   1.82   4729    1.0
d_9x   0.17  2.4e-3   0.16 4.3e-3   0.05   0.12   0.24   0.58   4580    1.0
d      3.73    0.02   1.25   1.66   2.84   3.59   4.51   6.57   4367    1.0
lp__  -1.64    0.08   2.58  -7.78  -3.13  -1.28   0.24   2.37   1048    1.0
"""
