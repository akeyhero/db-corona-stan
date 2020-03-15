import pystan
import matplotlib.pyplot as plt
from data import data

model = pystan.StanModel(model_code="""
functions {
    // smoothed version of float in order to avoid zero gradients
    real sfloor(real a) {
        return max([0, round(a) + inv_logit(20 * (a - round(a))) - 1]);
    }
}
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
    d = sfloor(d_0x) + sfloor(d_1x) + sfloor(d_2x) + sfloor(d_3x) + sfloor(d_4x) + sfloor(d_5x) + sfloor(d_6x) + sfloor(d_7x) + sfloor(d_8x) + sfloor(d_9x);
}
model {
    p_flu_0x ~ beta(sfloor(d_0x) + 1, S_0x - sfloor(d_0x) + 1);
    p_flu_1x ~ beta(sfloor(d_1x) + 1, S_1x - sfloor(d_1x) + 1);
    p_flu_2x ~ beta(sfloor(d_2x) + 1, S_2x - sfloor(d_2x) + 1);
    p_flu_3x ~ beta(sfloor(d_3x) + 1, S_3x - sfloor(d_3x) + 1);
    p_flu_4x ~ beta(sfloor(d_4x) + 1, S_4x - sfloor(d_4x) + 1);
    p_flu_5x ~ beta(sfloor(d_5x) + 1, S_5x - sfloor(d_5x) + 1);
    p_flu_6x ~ beta(sfloor(d_6x) + 1, S_6x - sfloor(d_6x) + 1);
    p_flu_7x ~ beta(sfloor(d_7x) + 1, S_7x - sfloor(d_7x) + 1);
    p_flu_8x ~ beta(sfloor(d_8x) + 1, S_8x - sfloor(d_8x) + 1);
    p_flu_9x ~ beta(sfloor(d_9x) + 1, S_9x - sfloor(d_9x) + 1);
}
""")

fit = model.sampling(data=data, chains=4)
print(fit)

fit.plot()
plt.show()

"""
Result

       mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
d_0x   0.43  6.2e-3   0.25   0.02   0.21   0.43   0.64   0.87   1667    1.0
d_1x   0.43  7.9e-3   0.25   0.02   0.21   0.42   0.64   0.88   1031   1.01
d_2x   0.47  8.7e-3   0.28   0.03   0.23   0.47   0.69   0.93   1057    1.0
d_3x   0.46    0.01   0.27   0.02   0.23   0.46   0.69   0.93    730    1.0
d_4x   0.46  9.1e-3   0.27   0.03   0.23   0.47   0.68   0.93    906    1.0
d_5x   0.47  8.3e-3    0.3   0.02   0.22   0.46    0.7   0.98   1317    1.0
d_6x   1.19    0.02   0.83   0.07   0.55    1.0    1.7   3.12   1596    1.0
d_7x   1.23    0.03    0.9   0.04   0.52   1.09   1.75   3.34   1136   1.01
d_8x   0.72    0.01   0.54   0.04   0.32   0.64   0.93   1.98   1404    1.0
d_9x   0.46  8.0e-3   0.26   0.02   0.24   0.46   0.68   0.91   1076    1.0
d      1.82    0.03   1.29   0.01   1.01   1.69    2.5   4.93   1710    1.0
lp__  14.02    0.12   2.91   7.24   12.3  14.37  16.15  18.61    607   1.01
"""

