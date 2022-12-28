################################################################################
# Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm


################################################################################
# Games running data
np.set_printoptions(threshold=np.inf)

total_states = 8
total_actions = 3
iterations = 15000
alpha = 0.2   #Learning Rate
df = 0.8  #discount Factor
d_win = 0.0025
d_lose = 0.01
end = total_actions - 1
