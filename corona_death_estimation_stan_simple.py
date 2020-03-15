import pystan
import matplotlib.pyplot as plt

model = pystan.StanModel(model_code='''
data {
    int N;
    int D;
}
parameters {
    real<lower=0, upper=1> p;
}
model {
    D ~ binomial(N, p);
}
''')

data = {
    'N': 696,
    'D':   7
}

fit = model.sampling(data=data, chains=4)
print(fit)

fit.plot()
plt.show()

"""
Result

       mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
p      0.01  1.1e-4 3.9e-3 4.7e-3 8.4e-3   0.01   0.01   0.02   1329    1.0
lp__ -44.22    0.02   0.75 -46.41 -44.35 -43.93 -43.75  -43.7   1597    1.0
"""
