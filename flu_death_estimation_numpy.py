import numpy as np

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
N = 10000
MIN_DEATH = 7

sample = (np.random.binomial(data['S_00_04'], data['p_flu_00_04'], N) +
          np.random.binomial(data['S_05_17'], data['p_flu_05_17'], N) +
          np.random.binomial(data['S_18_49'], data['p_flu_18_49'], N) +
          np.random.binomial(data['S_50_64'], data['p_flu_50_64'], N) +
          np.random.binomial(data['S_65_xx'], data['p_flu_65_xx'], N))
probability = float(sum(sample >= MIN_DEATH)) / N

print('Average # of deaths: %.2f' % (sample.mean()))
print('# of deaths >= %d: %.2f%%' % (MIN_DEATH, probability * 100))


"""
Result

Average # of deaths: 2.57
# of deaths >= 7: 1.72%
"""
