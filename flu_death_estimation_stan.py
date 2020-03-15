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
    int<lower=0, upper=S_0x+1> d_0x;
    int<lower=0, upper=S_1x+1> d_1x;
    int<lower=0, upper=S_2x+1> d_2x;
    int<lower=0, upper=S_3x+1> d_3x;
    int<lower=0, upper=S_4x+1> d_4x;
    int<lower=0, upper=S_5x+1> d_5x;
    int<lower=0, upper=S_6x+1> d_6x;
    int<lower=0, upper=S_7x+1> d_7x;
    int<lower=0, upper=S_8x+1> d_8x;
    int<lower=0, upper=S_9x+1> d_9x;
}
transformed parameters {
    int d;
    d = d_0x + d_1x + d_2x + d_3x + d_4x + d_5x + d_6x + d_7x + d_8x + d_9x;
}
model {
    d_0x ~ binomial(S_0x, p_flu_0x);
    d_1x ~ binomial(S_1x, p_flu_1x);
    d_2x ~ binomial(S_2x, p_flu_2x);
    d_3x ~ binomial(S_3x, p_flu_3x);
    d_4x ~ binomial(S_4x, p_flu_4x);
    d_5x ~ binomial(S_5x, p_flu_5x);
    d_6x ~ binomial(S_6x, p_flu_6x);
    d_7x ~ binomial(S_7x, p_flu_7x);
    d_8x ~ binomial(S_8x, p_flu_8x);
    d_9x ~ binomial(S_9x, p_flu_9x);
}
""")

fit = model.sampling(data=data, chains=4)
print(fit)

fit.plot()
plt.show()

"""
ValueError: Failed to parse Stan model 'anon_model_fecb1e77228fe372ef4eb9bc4bcc8086'. Error message:
SYNTAX ERROR, MESSAGE(S) FROM PARSER:
Parameters or transformed parameters cannot be integer or integer array;  found int variable declaration, name=d_0x
 error in 'unknown file name' at line 26, column 35
  -------------------------------------------------
    24: parameters {
    25:     // upper = S_xx + 1 so that S_xx can be 0
    26:     int<lower=0, upper=S_0x+1> d_0x;
                                          ^
    27:     int<lower=0, upper=S_1x+1> d_1x;
  -------------------------------------------------
"""
