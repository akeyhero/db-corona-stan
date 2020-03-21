import pystan

model = pystan.StanModel(model_code='''
data {
    // # of total deaths in Diamond Princes
    int D_dp;

    // # of comfirmed cases by age in Diamond Princess
    int N_dp_0x;
    int N_dp_1x;
    int N_dp_2x;
    int N_dp_3x;
    int N_dp_4x;
    int N_dp_5x;
    int N_dp_6x;
    int N_dp_7x;
    int N_dp_8x;
    int N_dp_9x;

    // # of deaths by age in China
    int D_ch_0x;
    int D_ch_1x;
    int D_ch_2x;
    int D_ch_3x;
    int D_ch_4x;
    int D_ch_5x;
    int D_ch_6x;
    int D_ch_7x;
    int D_ch_80_up;

    // # of comfirmed cases by age in China
    int N_ch_0x;
    int N_ch_1x;
    int N_ch_2x;
    int N_ch_3x;
    int N_ch_4x;
    int N_ch_5x;
    int N_ch_6x;
    int N_ch_7x;
    int N_ch_80_up;
}
transformed data {
    // # of total comfirmed cases in Diamond Princess
    int N_dp;

    // # of total comfirmed cases in China
    int N_ch;

    N_dp = N_dp_0x + N_dp_1x + N_dp_2x + N_dp_3x + N_dp_4x + N_dp_5x + N_dp_6x + N_dp_7x + N_dp_8x + N_dp_9x;
    N_ch = N_ch_0x + N_ch_1x + N_ch_2x + N_ch_3x + N_ch_4x + N_ch_5x + N_ch_6x + N_ch_7x + N_ch_80_up;
}
parameters {
    // coverage of comfirmation in China
    real<lower=0, upper=1> c;

    // mortality rate by age in China
    real<lower=0, upper=1> p_ch_0x;
    real<lower=0, upper=1> p_ch_1x;
    real<lower=0, upper=1> p_ch_2x;
    real<lower=0, upper=1> p_ch_3x;
    real<lower=0, upper=1> p_ch_4x;
    real<lower=0, upper=1> p_ch_5x;
    real<lower=0, upper=1> p_ch_6x;
    real<lower=0, upper=1> p_ch_7x;
    real<lower=0, upper=1> p_ch_80_up;
}
transformed parameters {
    // expected mortality rate by age in China
    real<lower=0, upper=1> p_0x;
    real<lower=0, upper=1> p_1x;
    real<lower=0, upper=1> p_2x;
    real<lower=0, upper=1> p_3x;
    real<lower=0, upper=1> p_4x;
    real<lower=0, upper=1> p_5x;
    real<lower=0, upper=1> p_6x;
    real<lower=0, upper=1> p_7x;
    real<lower=0, upper=1> p_80_up;

    // expected mortality rate in China
    real<lower=0, upper=1> p;

    // mortality rate in Diamond Princess
    real<lower=0, upper=1> p_dp;

    p_0x    = c * p_ch_0x;
    p_1x    = c * p_ch_1x;
    p_2x    = c * p_ch_2x;
    p_3x    = c * p_ch_3x;
    p_4x    = c * p_ch_4x;
    p_5x    = c * p_ch_5x;
    p_6x    = c * p_ch_6x;
    p_7x    = c * p_ch_7x;
    p_80_up = c * p_ch_80_up;

    p = (p_0x    * N_ch_0x +
         p_1x    * N_ch_1x +
         p_2x    * N_ch_2x +
         p_3x    * N_ch_3x +
         p_4x    * N_ch_4x +
         p_5x    * N_ch_5x +
         p_6x    * N_ch_6x +
         p_7x    * N_ch_7x +
         p_80_up * N_ch_80_up
         ) / N_ch;

    p_dp = (p_0x    * N_dp_0x +
            p_1x    * N_dp_1x +
            p_2x    * N_dp_2x +
            p_3x    * N_dp_3x +
            p_4x    * N_dp_4x +
            p_5x    * N_dp_5x +
            p_6x    * N_dp_6x +
            p_7x    * N_dp_7x +
            p_80_up * (N_dp_8x + N_dp_9x)
            ) / N_dp;
}
model {
    c ~ uniform(0, 1);

    D_ch_0x    ~ binomial(N_ch_0x,    p_ch_0x);
    D_ch_1x    ~ binomial(N_ch_1x,    p_ch_1x);
    D_ch_2x    ~ binomial(N_ch_2x,    p_ch_2x);
    D_ch_3x    ~ binomial(N_ch_3x,    p_ch_3x);
    D_ch_4x    ~ binomial(N_ch_4x,    p_ch_4x);
    D_ch_5x    ~ binomial(N_ch_5x,    p_ch_5x);
    D_ch_6x    ~ binomial(N_ch_6x,    p_ch_6x);
    D_ch_7x    ~ binomial(N_ch_7x,    p_ch_7x);
    D_ch_80_up ~ binomial(N_ch_80_up, p_ch_80_up);

    D_dp ~ binomial(N_dp, p_dp);
}
''')

data = {
    # number of total deaths in Diamond Princes
    'D_dp': 6,

    # number of comfirmed cases by age in Diamond Princess *1
    'N_dp_0x':   1,
    'N_dp_1x':   5,
    'N_dp_2x':  28,
    'N_dp_3x':  34,
    'N_dp_4x':  27,
    'N_dp_5x':  59,
    'N_dp_6x': 177,
    'N_dp_7x': 234,
    'N_dp_8x':  52,
    'N_dp_9x':   2,

    # number of deaths by age in China *2
    'D_ch_0x':      0,
    'D_ch_1x':      1,
    'D_ch_2x':      7,
    'D_ch_3x':     18,
    'D_ch_4x':     38,
    'D_ch_5x':    130,
    'D_ch_6x':    309,
    'D_ch_7x':    312,
    'D_ch_80_up': 208,

    # number of comfirmed cases by age in China *2
    'N_ch_0x':     416,
    'N_ch_1x':     549,
    'N_ch_2x':    3619,
    'N_ch_3x':    7600,
    'N_ch_4x':    8571,
    'N_ch_5x':   10008,
    'N_ch_6x':    8583,
    'N_ch_7x':    3918,
    'N_ch_80_up': 1408,
}
# *1 https://www.niid.go.jp/niid/ja/diseases/ka/corona-virus/2019-ncov/2484-idsc/9422-covid-dp-2.html
# *2 http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51

fit = model.sampling(data=data, chains=4)
print(fit)

"""
Result

             mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
c            0.21  8.7e-4   0.08   0.08   0.15    0.2   0.25   0.38   7792    1.0
p_ch_0x    2.4e-3  3.0e-5 2.3e-3 7.1e-5 7.1e-4 1.7e-3 3.2e-3 8.3e-3   5850    1.0
p_ch_1x    3.6e-3  3.3e-5 2.6e-3 4.3e-4 1.7e-3 3.0e-3 4.9e-3   0.01   6364    1.0
p_ch_2x    2.2e-3  9.7e-6 8.0e-4 9.5e-4 1.6e-3 2.1e-3 2.7e-3 4.0e-3   6894    1.0
p_ch_3x    2.5e-3  6.3e-6 5.7e-4 1.5e-3 2.1e-3 2.5e-3 2.9e-3 3.7e-3   8234    1.0
p_ch_4x    4.6e-3  8.7e-6 7.4e-4 3.2e-3 4.1e-3 4.5e-3 5.0e-3 6.1e-3   7259    1.0
p_ch_5x      0.01  1.3e-5 1.1e-3   0.01   0.01   0.01   0.01   0.02   7562    1.0
p_ch_6x      0.04  2.4e-5 2.0e-3   0.03   0.03   0.04   0.04   0.04   7096    1.0
p_ch_7x      0.08  5.2e-5 4.3e-3   0.07   0.08   0.08   0.08   0.09   6650    1.0
p_ch_80_up   0.15  1.1e-4 9.2e-3   0.13   0.14   0.15   0.15   0.17   7552    1.0
p_0x       4.8e-4  7.1e-6 5.3e-4 1.2e-5 1.3e-4 3.1e-4 6.5e-4 1.9e-3   5530    1.0
p_1x       7.4e-4  8.5e-6 6.3e-4 7.0e-5 3.0e-4 5.8e-4 9.8e-4 2.4e-3   5462    1.0
p_2x       4.5e-4  3.2e-6 2.4e-4 1.3e-4 2.8e-4 4.0e-4 5.7e-4 1.1e-3   5637    1.0
p_3x       5.2e-4  2.7e-6 2.3e-4 1.8e-4 3.4e-4 4.8e-4 6.4e-4 1.1e-3   7336    1.0
p_4x       9.4e-4  4.6e-6 3.9e-4 3.6e-4 6.5e-4 8.8e-4 1.2e-3 1.9e-3   7248    1.0
p_5x       2.7e-3  1.2e-5 1.0e-3 1.1e-3 1.9e-3 2.5e-3 3.3e-3 5.0e-3   7484    1.0
p_6x       7.4e-3  3.1e-5 2.8e-3 3.0e-3 5.4e-3 7.0e-3 9.0e-3   0.01   7796    1.0
p_7x         0.02  6.9e-5 6.1e-3 6.6e-3   0.01   0.02   0.02   0.03   7770    1.0
p_80_up      0.03  1.3e-4   0.01   0.01   0.02   0.03   0.04   0.06   7937    1.0
p          4.7e-3  2.0e-5 1.8e-3 1.9e-3 3.5e-3 4.5e-3 5.8e-3 8.6e-3   7853    1.0
p_dp         0.01  4.7e-5 4.2e-3 4.5e-3 8.3e-3   0.01   0.01   0.02   7865    1.0
lp__        -4215    0.06   2.35  -4220  -4216  -4214  -4213  -4211   1641    1.0
"""
