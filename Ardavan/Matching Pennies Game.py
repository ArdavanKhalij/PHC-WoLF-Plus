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


################################################################################
# Helper function
def actions_present(state):
	available_actions=[]
	for i in range(0,total_actions):
		if ((state[0,i]>=0)):
			available_actions.append(i)
	return available_actions

def Qmax(Q,next_state):
	return np.max(Q[next_state])

def updateQ(state,action,Q,R):
	next_state=action
	q = ((1-alpha)*Q[state,action]) + alpha*(R[state,action] + (df * Qmax(Q,next_state)))
	return q

def actions_select(state,Policy):
	p1=Policy[state,:]
	if (np.sum(p1)==1.0):
		return np.random.choice(3,1,p=p1)
	else:
		p1 /= p1.sum()
		return np.random.choice(3,1,p=p1)

def delta(state,Q,Policy,MeanPolicy,d_win,d_lose):
	sumPolicy=0.0
	sumMeanPolicy=0.0
	for i in range(0,total_actions):
		sumPolicy=sumPolicy+(Policy[state,i]*Q[state,i])
		sumMeanPolicy=sumMeanPolicy+(MeanPolicy[state,i]*Q[state,i])
	if (sumPolicy>sumMeanPolicy):
		return d_win
	else:
		return d_lose

def update_pi(state,Policy,MeanPolicy,Q,d_win,d_lose):
	maxQValueIndex = np.argmax(Q[state])
	for i in range(0,total_actions):
		d_plus = delta(state,Q,Policy,MeanPolicy,d_win,d_lose)
		d_minus = ((-1.0)*d_plus)/((total_actions) - 1.0)
		if (i==maxQValueIndex):
			Policy[state,i] = min(1.0,Policy[state,i] + d_plus)
		else:
			Policy[state,i] = max(0.0,Policy[state,i] + d_minus)
	return Policy

def update_meanpi(state,C,MeanPolicy,Policy):
	for i in range(0,total_actions):
		MeanPolicy[state,i] = MeanPolicy[state,i] + ((1.0/C[state]) * (Policy[state,i]-MeanPolicy[state,i]))
	return	MeanPolicy
