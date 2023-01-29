############################################################################
# Libraries
import numpy as np
import matplotlib.pyplot as plt
############################################################################


############################################################################
# Data of the problem
alpha = 0.1
# delta = 0.00001
delta_w = 0.0001
delta_l = 0.0002
NumberOfStates = 3
NumberOfActions = 3
Iterations = 300000
gamma = 0.99
RUN = 1

AVERAGEEVERY = 1
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
"""Policy1 = []
for i in range(0, NumberOfStates):
    Policy1.append([])
    for j in range(0, NumberOfActions):
        Policy1[i].append(1 / NumberOfActions)
Policy2 = []
for i in range(0, NumberOfStates):
    Policy2.append([])
    for j in range(0, NumberOfActions):
        Policy2[i].append(1 / NumberOfActions)"""

Policy1 = [
    [0.25, 0.25, 0.50],
    [0.25, 0.25, 0.50],
    [0.25, 0.25, 0.50]
]

Policy2 = [
    [0.50, 0.25, 0.25],
    [0.50, 0.25, 0.25],
    [0.50, 0.25, 0.25]
]
############################################################################


############################################################################
# Creating average policy table
Average_Policy1 = []
for i in range(0, NumberOfStates):
    Average_Policy1.append([])
    for j in range(0, NumberOfActions):
        Average_Policy1[i].append(1 / NumberOfActions)
Average_Policy2 = []
for i in range(0, NumberOfStates):
    Average_Policy2.append([])
    for j in range(0, NumberOfActions):
        Average_Policy2[i].append(1 / NumberOfActions)
############################################################################


############################################################################
# Initialize C
C1 = []
for i in range(0, NumberOfStates):
    C1.append(0)
C2 = []
for i in range(0, NumberOfStates):
    C2.append(0)
############################################################################


############################################################################
# Rewards or Payoff table
Rewards = [[0, -1, 1],
           [1, 0, -1],
           [-1, 1, 0]]
############################################################################


############################################################################
# Function for returning delta
def get_delta(s):
    sum11 = 0
    sum12 = 0
    sum21 = 0
    sum22 = 0
    return_delta1 = 0
    return_delta2 = 0
    for i in range(0, NumberOfActions):
        sum11 = sum11 + Policy1[s][i] * Q1[s][i]
        sum12 = sum12 + Average_Policy1[s][i] * Q1[s][i]
        sum21 = sum21 + Policy2[s][i] * Q2[s][i]
        sum22 = sum22 + Average_Policy2[s][i] * Q2[s][i]
    if sum11 > sum12:
        return_delta1 = delta_w
    else:
        return_delta1 = delta_l
    if sum21 > sum22:
        return_delta2 = delta_w
    else:
        return_delta2 = delta_l
    return return_delta1, return_delta2
############################################################################


############################################################################
# Repeating part
for run in range(0, RUN):
    for i in range(0, Iterations):
        # print(Q1)
        if i % 1000 == 0:
            print(i)
        p.append(Policy1[0][1])
        p2.append(Policy1[0][0])
        for j in range(0, NumberOfStates):
            # Choose Action
            action1 = np.random.choice(range(0, NumberOfActions), p=Policy1[j])
            action2 = np.random.choice(range(0, NumberOfActions), p=Policy2[j])
            # Get rewards
            reward1 = Rewards[action1][action2]
            reward2 = -1 * reward1
            # Update Q table
            QPrim1 = []
            QPrim2 = []
            QPrim1 = Q1[j]
            QPrim2 = Q2[j]
            if reward1 == max(Rewards[0] or reward1 != min(Rewards[0])):
                Q1[j][action1] = ((1 - alpha) * Q1[j][action1]) + \
                    (alpha * (reward1 + gamma * max(Q1[j])))
            else:
                Q1[j][action1] = ((1 - alpha) * Q1[j][action1]) + \
                    (alpha * (reward1 + gamma *
                              max(Q1[(j + 1) % NumberOfStates])))
            if reward2 == max(Rewards[0] or reward2 != min(Rewards[0])):
                Q2[j][action2] = ((1 - alpha) * Q2[j][action2]) + \
                    (alpha * (reward2 + gamma * max(Q2[j])))
            else:
                Q2[j][action2] = ((1 - alpha) * Q2[j][action2]) + \
                    (alpha * (reward2 + gamma *
                              max(Q2[(j + 1) % NumberOfStates])))
            # Update average policy
            C1[j] = C1[j] + 1
            C2[j] = C2[j] + 1
            for k in range(0, NumberOfActions):
                Average_Policy1[j][k] = Average_Policy1[j][k] + \
                    (1 / C1[j]) * (Policy1[j][k] - Average_Policy1[j][k])
                Average_Policy2[j][k] = Average_Policy2[j][k] + \
                    (1 / C2[j]) * (Policy2[j][k] - Average_Policy2[j][k])
            # Update policy
            delta1, delta2 = get_delta(j)
            if action1 == QPrim1.index(max(QPrim1)):
                if Policy1[j][action1] <= 1 - delta1:
                    if min(Policy2[j]) >= (delta1 / (NumberOfStates - 1)):
                        Policy1[j][action1] = Policy1[j][action1] + delta1
                        for k in range(0, NumberOfActions):
                            if k != action1:
                                Policy1[j][k] = Policy1[j][k] - \
                                    (delta1/(NumberOfActions - 1))
            else:
                if Policy1[j][action1] >= delta1:
                    if max(Policy1[j]) <= 1 - (delta1 / (NumberOfStates - 1)):
                        Policy1[j][action1] = Policy1[j][action1] - \
                            (delta1 / (NumberOfActions - 1))
                        for k in range(0, NumberOfActions):
                            if k != action1:
                                Policy1[j][k] = Policy1[j][k] + \
                                    ((delta1 / (NumberOfActions - 1)) /
                                     (NumberOfActions - 1))
            if action2 == QPrim2.index(max(QPrim2)):
                if Policy2[j][action2] <= 1 - delta2:
                    if min(Policy2[j]) >= (delta2 / (NumberOfStates - 1)):
                        Policy2[j][action2] = Policy2[j][action2] + delta2
                        for k in range(0, NumberOfActions):
                            if k != action2:
                                Policy2[j][k] = Policy2[j][k] - \
                                    (delta2/(NumberOfActions - 1))
            else:
                if Policy2[j][action2] >= delta2:
                    if max(Policy2[j]) <= 1 - (delta2/(NumberOfStates - 1)):
                        Policy2[j][action2] = Policy2[j][action2] - \
                            (delta2/(NumberOfActions - 1))
                        for k in range(0, NumberOfActions):
                            if k != action2:
                                Policy2[j][k] = Policy2[j][k] + \
                                    ((delta2 / (NumberOfActions - 1)) /
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
            len(p_of_head_1[j][i:i + 1])
        x.append(avgOf300)
    plottingx.append(sum(x) / len(x))


for i in range(0, len(p_of_head_2[0]), AVERAGEEVERY):
    x = []
    for j in range(0, len(p_of_head_1)):
        avgOf300 = sum(p_of_head_2[j][i:i + AVERAGEEVERY]) / \
            len(p_of_head_2[j][i:i + AVERAGEEVERY])
        x.append(avgOf300)
    plottingy.append(sum(x) / len(x))


plt.plot(plottingx[:200], plottingy[:200], color='red')
# plt.plot(p_of_head_1[0])
#plt.plot(p_of_head_2, color="red")

plt.plot(plottingx[200:], plottingy[200:], color='green')

# plt.plot(plottingx)

plt.xlabel("Pr(Rock)")
plt.ylabel("Pr(Paper)")
plt.show()
############################################################################
