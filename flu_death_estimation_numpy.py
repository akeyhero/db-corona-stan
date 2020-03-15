import numpy as np
from data import data

N = 10000
MIN_DEATH = 7

sample = (np.random.binomial(data['S_0x'], data['p_flu_0x'], N) +
          np.random.binomial(data['S_1x'], data['p_flu_1x'], N) +
          np.random.binomial(data['S_2x'], data['p_flu_2x'], N) +
          np.random.binomial(data['S_3x'], data['p_flu_3x'], N) +
          np.random.binomial(data['S_4x'], data['p_flu_4x'], N) +
          np.random.binomial(data['S_5x'], data['p_flu_5x'], N) +
          np.random.binomial(data['S_6x'], data['p_flu_6x'], N) +
          np.random.binomial(data['S_7x'], data['p_flu_7x'], N) +
          np.random.binomial(data['S_8x'], data['p_flu_8x'], N) +
          np.random.binomial(data['S_9x'], data['p_flu_9x'], N))
probability = float(sum(sample >= MIN_DEATH)) / N

print('Average # of deaths: %.2f' % (sample.mean()))
print('# of deaths >= %d: %.2f%%' % (MIN_DEATH, probability * 100))


"""
Result

Average # of deaths: 1.67
# of deaths >= 7: 0.21%
"""
