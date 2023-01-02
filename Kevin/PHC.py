############################################################################
# Libraries
import numpy as np
import matplotlib.pyplot as plt
############################################################################


############################################################################
# Data of the problem
alpha = 0.0001
delta = 0.000001
NumberOfStates = 3
NumberOfActions = 3
Iterations = 1000000
gamma = 0.99
RUN = 1

AVERAGEEVERY = 100
############################################################################


############################################################################
# Variables for plotting
p_of_head_1 = []
p_of_head_2 = []
p = []
p2 = []
############################################################################


############################################################################
# Creating Q table
Q1 = []
for i in range(0, NumberOfStates):
    Q1.append([])
    for j in range(0, NumberOfActions):
        Q1[i].append(0)
Q2 = []
for i in range(0, NumberOfStates):
    Q2.append([])
    for j in range(0, NumberOfActions):
        Q2[i].append(0)
############################################################################


############################################################################
# Creating policy table
Policy1 = []
for i in range(0, NumberOfStates):
    Policy1.append([])
    for j in range(0, NumberOfActions):
        Policy1[i].append(1 / NumberOfActions)
Policy2 = []
for i in range(0, NumberOfStates):
    Policy2.append([])
    for j in range(0, NumberOfActions):
        Policy2[i].append(1 / NumberOfActions)
############################################################################


############################################################################
# Rewards or Payoff table
Rewards = [[0, -1, 1],
           [1, 0, -1],
           [-1, 1, 0]]
############################################################################

############################################################################
# Function to average the probabilities of playing a certain action


def avg_prob(action):
    all_probs = []

    for i in range(NumberOfStates):
        all_probs.append(Policy1[i][action])

    return sum(all_probs)/len(all_probs)
############################################################################


############################################################################
# Repeating part
for run in range(0, RUN):
    for i in range(0, Iterations):
        # print(Q1)
        if i % 1000 == 0:
            print(i)
        #
        p.append(Policy1[0][1])  # Probability of playing Rock
        p2.append(Policy1[0][0])  # Probability of playing Paper

        for j in range(0, NumberOfStates):
            # Choose Action
            action1 = np.random.choice(range(0, NumberOfActions), p=Policy1[j])
            action2 = np.random.choice(range(0, NumberOfActions), p=Policy2[j])
            # Get rewards
            reward1 = Rewards[action1][action2]
            reward2 = -1 * reward1
            # Update Q tables and Policy table
            QPrim1 = []
            QPrim2 = []
            QPrim1 = Q1[j]
            QPrim2 = Q2[j]
            if reward1 == max(Rewards[0]):
                Q1[j][action1] = ((1 - alpha) * Q1[j][action1]) + \
                    (alpha * (reward1 + gamma * max(Q1[j])))
            else:
                Q1[j][action1] = ((1 - alpha) * Q1[j][action1]) + \
                    (alpha * (reward1 + gamma *
                              max(Q1[(j + 1) % NumberOfStates])))
            if reward2 == max(Rewards[0]):
                Q2[j][action2] = ((1 - alpha) * Q2[j][action2]) + \
                    (alpha * (reward2 + gamma * max(Q2[j])))
            else:
                Q2[j][action2] = ((1 - alpha) * Q2[j][action2]) + \
                    (alpha * (reward2 + gamma *
                              max(Q2[(j + 1) % NumberOfStates])))
            if action1 == QPrim1.index(max(QPrim1)):
                if Policy1[j][action1] <= 1 - delta:
                    if min(Policy2[j]) >= (delta / (NumberOfStates - 1)):
                        Policy1[j][action1] = Policy1[j][action1] + delta
                        for k in range(0, NumberOfActions):
                            if k != action1:
                                Policy1[j][k] = Policy1[j][k] - \
                                    (delta/(NumberOfActions - 1))
            else:
                if Policy1[j][action1] >= delta:
                    if max(Policy1[j]) <= 1 - (delta / (NumberOfStates - 1)):
                        Policy1[j][action1] = Policy1[j][action1] - \
                            (delta / (NumberOfActions - 1))
                        for k in range(0, NumberOfActions):
                            if k != action1:
                                Policy1[j][k] = Policy1[j][k] + \
                                    ((delta / (NumberOfActions - 1)) /
                                     (NumberOfActions - 1))
            if action2 == QPrim2.index(max(QPrim2)):
                if Policy2[j][action2] <= 1 - delta:
                    if min(Policy2[j]) >= (delta / (NumberOfStates - 1)):
                        Policy2[j][action2] = Policy2[j][action2] + delta
                        for k in range(0, NumberOfActions):
                            if k != action2:
                                Policy2[j][k] = Policy2[j][k] - \
                                    (delta/(NumberOfActions - 1))
            else:
                if Policy2[j][action2] >= delta:
                    if max(Policy2[j]) <= 1 - (delta / (NumberOfStates - 1)):
                        Policy2[j][action2] = Policy2[j][action2] - \
                            (delta/(NumberOfActions - 1))
                        for k in range(0, NumberOfActions):
                            if k != action2:
                                Policy2[j][k] = Policy2[j][k] + \
                                    ((delta / (NumberOfActions - 1)) /
                                     (NumberOfActions - 1))
    p_of_head_1.append(p)
    p_of_head_2.append(p2)
    p = []
    p2 = []
    Policy1 = []
    for i in range(0, NumberOfStates):
        Policy1.append([])
        for j in range(0, NumberOfActions):
            Policy1[i].append(1 / NumberOfActions)
    Policy2 = []
    for i in range(0, NumberOfStates):
        Policy2.append([])
        for j in range(0, NumberOfActions):
            Policy2[i].append(1 / NumberOfActions)
    Q1 = []
    for i in range(0, NumberOfStates):
        Q1.append([])
        for j in range(0, NumberOfActions):
            Q1[i].append(0)
    Q2 = []
    for i in range(0, NumberOfStates):
        Q2.append([])
        for j in range(0, NumberOfActions):
            Q2[i].append(0)
############################################################################


############################################################################
# Plotting
plottingx = []
plottingy = []
# print(len(p_of_head_1[0]))
for i in range(0, len(p_of_head_1[0]), AVERAGEEVERY):
    x = []
    for j in range(0, len(p_of_head_1)):
        avgOf300 = sum(p_of_head_1[j][i:i + AVERAGEEVERY]) / \
            len(p_of_head_1[j][i:i + AVERAGEEVERY])
        x.append(avgOf300)
    plottingx.append(sum(x) / len(x))


for i in range(0, len(p_of_head_2[0]), AVERAGEEVERY):
    x = []
    for j in range(0, len(p_of_head_2)):
        avgOf300 = sum(p_of_head_2[j][i:i + AVERAGEEVERY]) / \
            len(p_of_head_2[j][i:i + AVERAGEEVERY])
        x.append(avgOf300)
    plottingy.append(sum(x) / len(x))
#plt.subplot(1, 2, 1)
# plt.plot(plottingx)

#plt.subplot(1, 2, 2)
# plt.plot(plottingy)

#plt.subplot(2, 1, 3)
#


# plt.plot(p_of_head_1[0])
#plt.plot(p_of_head_2, color="red")


# plt.plot(p_of_head_1[0])
#plt.plot(p_of_head_2, color="red")

plt.plot(plottingx[:200], plottingy[:200], color='red')
plt.plot(plottingx[200:], plottingy[200:], color='green')

plt.xlabel("Pr(Rock)")
plt.ylabel("Pr(Paper)")
plt.show()
############################################################################
