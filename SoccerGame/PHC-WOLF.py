############################################################################
# Libraries
import matplotlib.pyplot as plt
import numpy as np
import SoccerGame as sg
import statistics

############################################################################


############################################################################
# Data of the problem
alpha = 1
alpha_decay = 0.0000004
delta_w = 0.000001
delta_l = 0.000004
NumberOfStates = 24
NumberOfActions = 5
Iterations = 2000000
LearnAfter = 1000000
gamma = 0.9
RUN = 1

AVERAGEEVERY = 50000
############################################################################


############################################################################
# Variables for Result
wins_agent1 = []
wins_agent12 = []
SD_agent1 = []
SD_agent12 = []
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
Rewards = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
############################################################################


############################################################################
# Function for returning delta
def get_delta(s1, s2):
    sum11 = 0
    sum12 = 0
    sum21 = 0
    sum22 = 0
    return_delta1 = 0
    return_delta2 = 0
    for i in range(0, NumberOfActions):
        sum11 = sum11 + Policy1[s1][i] * Q1[s1][i]
        sum12 = sum12 + Average_Policy1[s1][i] * Q1[s1][i]
        sum21 = sum21 + Policy2[s2][i] * Q2[s2][i]
        sum22 = sum22 + Average_Policy2[s2][i] * Q2[s2][i]
    if sum11 > sum12:
        return_delta1 = delta_w
    else:
        return_delta1 = delta_l
    if sum21 > sum22:
        return_delta2 = delta_w
    else:
        return_delta2 = delta_l
    return return_delta1, return_delta2


# Converts x y coordinates to a state number
def convert_to_state(x, y):

    if x == 0 and y == 0:
        return 0
    elif x == 0 and y == 1:
        return 1
    elif x == 0 and y == 2:
        return 2
    elif x == 0 and y == 3:
        return 3
    elif x == 1 and y == 0:
        return 4
    elif x == 1 and y == 1:
        return 5
    elif x == 1 and y == 2:
        return 6
    elif x == 1 and y == 3:
        return 7
    elif x == 2 and y == 0:
        return 8
    elif x == 2 and y == 1:
        return 9
    elif x == 2 and y == 2:
        return 10
    elif x == 2 and y == 3:
        return 11
    elif x == 3 and y == 0:
        return 12
    elif x == 3 and y == 1:
        return 13
    elif x == 3 and y == 2:
        return 14
    elif x == 3 and y == 3:
        return 15
    elif x == 4 and y == 0:
        return 16
    elif x == 4 and y == 1:
        return 17
    elif x == 4 and y == 2:
        return 18
    elif x == 4 and y == 3:
        return 19
    ##Goal states
    elif x == -1 and y == 1:
        return 20
    elif x == -1 and y == 2:
        return 21
    elif x == 5 and y == 1:
        return 22
    elif x == 5 and y == 2:
        return 23


# Calc average probability to take a certain action


def calc_avg_prob(action):
    probas = []

    for state in NumberOfStates:
        proba = state[action]
        probas.append(proba)

    return sum(probas) / len(probas)


############################################################################


############################################################################

env = sg.BabyRobotEnv_v2()

# Repeating part
for run in range(0, RUN):
    for i in range(0, Iterations):
        if i < LearnAfter:
            if i % 1000 == 0:
                print(i)
            env.reset()
            # Reset cumulative reward
            env.agents[0].cum_rew = 0
            env.agents[1].cum_rew = 0
            done = False
            # Find the initial state of the agents
            state1 = convert_to_state(env.agents[0].x, env.agents[0].y)
            state2 = convert_to_state(env.agents[1].x, env.agents[1].y)
            while not done:
                # Choose Action
                if min(Policy1[state1]) < 0:
                    for k in range(0, NumberOfActions):
                        Policy1[state1][k] = Policy1[state1][k] - min(Policy1[state1])
                    sum1 = sum(Policy1[state1])
                    for k in range(0, NumberOfActions):
                        Policy1[state1][k] = Policy1[state1][k] / sum1
                if min(Policy2[state2]) < 0:
                    for k in range(0, NumberOfActions):
                        Policy2[state2][k] = Policy2[state2][k] - min(Policy2[state2])
                    sum2 = sum(Policy2[state2])
                    for k in range(0, NumberOfActions):
                        Policy2[state2][k] = Policy2[state2][k] / sum2
                action1 = np.random.choice(range(0, NumberOfActions), p=Policy1[state1])
                action2 = np.random.choice(range(0, NumberOfActions), p=Policy2[state2])
                # Get rewards and new state
                new_state1, new_state2, reward1, reward2, done, info = env.step(
                    action1=action1, action2=action2
                )
                new_state1 = convert_to_state(new_state1[0], new_state1[1])
                new_state2 = convert_to_state(new_state2[0], new_state2[1])

                # Update Q table
                Q1[state1][action1] = ((1 - alpha) * Q1[state1][action1]) + (
                    alpha * (reward1 + gamma * max(Q1[new_state1]))
                )
                Q2[state2][action2] = ((1 - alpha) * Q2[state2][action2]) + (
                    alpha * (reward2 + gamma * max(Q2[new_state2]))
                )

                QPrim1 = Q1[state1]
                QPrim2 = Q2[state2]

                # Update average policy
                C1[state1] = C1[state1] + 1
                C2[state2] = C2[state2] + 1
                for k in range(0, NumberOfActions):
                    Average_Policy1[state1][k] = Average_Policy1[state1][k] + (
                        1 / C1[state1]
                    ) * (Policy1[state1][k] - Average_Policy1[state1][k])
                    Average_Policy2[state2][k] = Average_Policy2[state2][k] + (
                        1 / C2[state2]
                    ) * (Policy2[state2][k] - Average_Policy2[state2][k])
                # Update policy
                delta1, delta2 = get_delta(s1=state1, s2=state2)
                if action1 == QPrim1.index(max(QPrim1)):
                    if Policy1[state1][action1] <= 1 - delta1:
                        if min(Policy1[state1]) >= (delta1 / (NumberOfStates - 1)):
                            Policy1[state1][action1] = Policy1[state1][action1] + delta1
                            for k in range(0, NumberOfActions):
                                if k != action1:
                                    Policy1[state1][k] = Policy1[state1][k] - (
                                        delta1 / (NumberOfActions - 1)
                                    )
                else:
                    if Policy1[state1][action1] >= delta1:
                        if max(Policy1[state1]) <= 1 - (delta1 / (NumberOfStates - 1)):
                            Policy1[state1][action1] = Policy1[state1][action1] - (
                                delta1 / (NumberOfActions - 1)
                            )
                            for k in range(0, NumberOfActions):
                                if k != action1:
                                    Policy1[state1][k] = Policy1[state1][k] + (
                                        (delta1 / (NumberOfActions - 1))
                                        / (NumberOfActions - 1)
                                    )
                if action2 == QPrim2.index(max(QPrim2)):
                    if Policy2[state2][action2] <= 1 - delta2:
                        if min(Policy2[state2]) >= (delta2 / (NumberOfStates - 1)):
                            Policy2[state2][action2] = Policy2[state2][action2] + delta2
                            for k in range(0, NumberOfActions):
                                if k != action2:
                                    Policy2[state2][k] = Policy2[state2][k] - (
                                        delta2 / (NumberOfActions - 1)
                                    )
                else:
                    if Policy2[state2][action2] >= delta2:
                        if max(Policy2[state2]) <= 1 - (delta2 / (NumberOfStates - 1)):
                            Policy2[state2][action2] = Policy2[state2][action2] - (
                                delta2 / (NumberOfActions - 1)
                            )
                            for k in range(0, NumberOfActions):
                                if k != action2:
                                    Policy2[state2][k] = Policy2[state2][k] + (
                                        (delta2 / (NumberOfActions - 1))
                                        / (NumberOfActions - 1)
                                    )

                state1 = new_state1
                state2 = new_state2
                env.agents[0].cum_rew += reward1
                env.agents[1].cum_rew += reward2

            alpha -= alpha_decay
        else:
            if i % 1000 == 0:
                print(i)
            env.reset()
            # Reset cumulative reward
            env.agents[0].cum_rew = 0
            env.agents[1].cum_rew = 0
            done = False
            # Find the initial state of the agents
            state1 = convert_to_state(env.agents[0].x, env.agents[0].y)
            state2 = convert_to_state(env.agents[1].x, env.agents[1].y)
            while not done:
                # Choose Action
                if min(Policy1[state1]) < 0:
                    for k in range(0, NumberOfActions):
                        Policy1[state1][k] = Policy1[state1][k] - min(Policy1[state1])
                    sum1 = sum(Policy1[state1])
                    for k in range(0, NumberOfActions):
                        Policy1[state1][k] = Policy1[state1][k] / sum1
                if min(Policy2[state2]) < 0:
                    for k in range(0, NumberOfActions):
                        Policy2[state2][k] = Policy2[state2][k] - min(Policy2[state2])
                    sum2 = sum(Policy2[state2])
                    for k in range(0, NumberOfActions):
                        Policy2[state2][k] = Policy2[state2][k] / sum2
                action1 = np.random.choice(range(0, NumberOfActions), p=Policy1[state1])
                action2 = np.random.choice(range(0, NumberOfActions), p=Policy2[state2])
                # Get rewards and new state
                new_state1, new_state2, reward1, reward2, done, info = env.step(
                    action1=action1, action2=action2
                )
                new_state1 = convert_to_state(new_state1[0], new_state1[1])
                new_state2 = convert_to_state(new_state2[0], new_state2[1])

                # Update Q table
                Q1[state1][action1] = ((1 - alpha) * Q1[state1][action1]) + (
                    alpha * (reward1 + gamma * max(Q1[new_state1]))
                )
                Q2[state2][action2] = ((1 - alpha) * Q2[state2][action2]) + (
                    alpha * (reward2 + gamma * max(Q2[new_state2]))
                )

                QPrim1 = Q1[state1]
                QPrim2 = Q2[state2]

                # Update average policy
                C1[state1] = C1[state1] + 1
                for k in range(0, NumberOfActions):
                    Average_Policy1[state1][k] = Average_Policy1[state1][k] + (
                        1 / C1[state1]
                    ) * (Policy1[state1][k] - Average_Policy1[state1][k])
                # Update policy
                delta1, delta2 = get_delta(s1=state1, s2=state2)
                if action1 == QPrim1.index(max(QPrim1)):
                    if Policy1[state1][action1] <= 1 - delta1:
                        if min(Policy1[state1]) >= (delta1 / (NumberOfStates - 1)):
                            Policy1[state1][action1] = Policy1[state1][action1] + delta1
                            for k in range(0, NumberOfActions):
                                if k != action1:
                                    Policy1[state1][k] = Policy1[state1][k] - (
                                        delta1 / (NumberOfActions - 1)
                                    )
                else:
                    if Policy1[state1][action1] >= delta1:
                        if max(Policy1[state1]) <= 1 - (delta1 / (NumberOfStates - 1)):
                            Policy1[state1][action1] = Policy1[state1][action1] - (
                                delta1 / (NumberOfActions - 1)
                            )
                            for k in range(0, NumberOfActions):
                                if k != action1:
                                    Policy1[state1][k] = Policy1[state1][k] + (
                                        (delta1 / (NumberOfActions - 1))
                                        / (NumberOfActions - 1)
                                    )

                state1 = new_state1
                state2 = new_state2
                env.agents[0].cum_rew += reward1
                env.agents[1].cum_rew += reward2
            if env.winner == 1:
                wins_agent1.append(1)
            else:
                wins_agent1.append(2)
            if i % AVERAGEEVERY == 0:
                wins_agent12.append(wins_agent1)
                wins_agent1 = []
            alpha -= alpha_decay
############################################################################


############################################################################
# Result
ListOfResults = []
for i in range(0, len(wins_agent12)):

    # print(len(wins_agent12[i]))
    numberOf1 = wins_agent12[i].count(1)
    numberOf2 = wins_agent12[i].count(2)

    # print(f"Win of agent 1: {numberOf1}")
    # print(f"Win of agent 2: {numberOf2}")
    if i == 0:
        ListOfResults.append(0.5)
    else:
        ListOfResults.append(numberOf1 / (numberOf1 + numberOf2))

plt.plot(ListOfResults)
plt.ylabel("Probability of winning")
plt.xlabel("Iterations (averaged every 50 000)")
print(f"Probability of winning for agent1 : {sum(ListOfResults)/len(ListOfResults)}")
print(f"Standard Deviation of agent 1 : {statistics.stdev(ListOfResults)}")
############################################################################
