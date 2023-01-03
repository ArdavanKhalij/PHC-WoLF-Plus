import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from enum import IntEnum
from random import uniform
from copy import copy
from stable_baselines3.common.env_checker import check_env

""" simple helper class to enumerate actions in the grid levels """


class Actions(IntEnum):
    Stay = 0
    North = 1
    East = 2
    South = 3
    West = 4


class Agent:
    def __init__(self, startx, starty, env):

        self.cum_rew = 0
        self.x = startx
        self.y = starty

        # define the maximum x and y values
        self.max_x = env.width - 1
        self.max_y = env.height - 1

    def next_pos(self, action):
        x = copy(self.x)
        y = copy(self.y)

        # move in the direction of the specified action
        if action == Actions.North:
            if self.x == 0 and self.y == 2:
                rand = uniform(0, 1)
                if rand < 0.5:
                    y -= 1
            elif self.x == 2 and self.y == 2:
                rand = uniform(0, 1)
                if rand < 0.5:
                    y -= 1
            else:
                y -= 1
        elif action == Actions.South:
            y += 1
        elif action == Actions.West:
            x -= 1
        elif action == Actions.East:
            x += 1

        # make sure the move stays on the grid
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.max_x:
            x = self.max_x
        if y > self.max_y:
            y = self.max_y

        return [x, y]

    def take_action(self, action):
        """apply the supplied action"""

        # move in the direction of the specified action
        if action == Actions.North:
            if self.x == 0 and self.y == 2:
                rand = uniform(0, 1)
                if rand < 0.5:
                    self.y -= 1
            elif self.x == 2 and self.y == 2:
                rand = uniform(0, 1)
                if rand < 0.5:
                    self.y -= 1
            else:
                self.y -= 1
        elif action == Actions.South:
            self.y += 1
        elif action == Actions.West:
            self.x -= 1
        elif action == Actions.East:
            self.x += 1

        # make sure the move stays on the grid
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x > self.max_x:
            self.x = self.max_x
        if self.y > self.max_y:
            self.y = self.max_y

    # get the enum name without the class
    def __str__(self):
        return self.name


class BabyRobotEnv_v1(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()

        # dimensions of the grid
        self.width = kwargs.get("width", 3)
        self.height = kwargs.get("height", 3)

        # define the maximum x and y values
        self.max_x = self.width - 1
        self.max_y = self.height - 1

        # there are 5 possible actions: move N,E,S,W or stay in same state
        self.action_space = Discrete(5)

        # the observation will be the coordinates of Baby Robot
        self.observation_space = MultiDiscrete([self.width, self.height])

        # Baby Robot's position in the grid

        # agent1 position
        self.x1 = 0
        self.y1 = self.max_y

        # agent2 position
        self.x2 = self.max_x
        self.y2 = self.max_y

    def step(self, action):
        obs = np.array([self.x, self.y])
        reward = -1
        done = True
        info = {}
        return obs, reward, done, info

    def reset(self):
        # reset Baby Robot's position in the grid
        self.x1 = 0
        self.y1 = self.max_y

        # reset Baby Robot's position in the grid
        self.x2 = self.max_x
        self.y2 = self.max_y

        return np.array([self.x1, self.y1, self.x2, self.y2])

    def render(self, mode):
        pass


class BabyRobotEnv_v2(BabyRobotEnv_v1):

    metadata = {"render_modes": ["human"]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # the start and end positions in the grid
        # - by default these are the top-left and bottom-right respectively
        self.start1 = kwargs.get("start", [0, self.max_y])
        self.start2 = kwargs.get("start", [self.max_x, self.max_y])

        self.end = kwargs.get("end", [1, 0])

        # Baby Robot's initial position
        # - by default this is the grid start
        self.initial_pos1 = kwargs.get("initial_pos", self.start1)
        self.initial_pos2 = kwargs.get("initial_pos", self.start2)

        # Baby Robot's position in the grid
        self.x1 = self.initial_pos1[0]
        self.y1 = self.initial_pos1[1]

        self.x2 = self.initial_pos2[0]
        self.y2 = self.initial_pos2[1]

        # Initialize agents
        self.agents = []
        self.agents.append(Agent(startx=self.x1, starty=self.y1, env=self))
        self.agents.append(Agent(startx=self.x2, starty=self.y2, env=self))

    def step(self, action1, action2):

        next_positions = []
        """for agent in self.agents:
            # take the action and update the position
            nxt_pos = agent.next_pos(action)
            next_positions.append(nxt_pos)
"""
        nxt_pos1 = self.agents[0].next_pos(action1)
        nxt_pos2 = self.agents[1].next_pos(action2)

        next_positions.append(nxt_pos1)
        next_positions.append(nxt_pos2)

        if next_positions[0] == next_positions[1]:
            reward1 = -1
            reward2 = -1
        else:
            self.agents[0].take_action(action1)
            self.agents[1].take_action(action2)
        # set the 'done' flag if we've reached the exit
        done1 = self.agents[0].x == self.end[0] and self.agents[0].y == self.end[1]
        done2 = self.agents[1].x == self.end[0] and self.agents[1].y == self.end[1]

        # get -1 reward for each step
        # - except at the terminal state which has zero reward
        reward1 = 0 if done1 else -1
        reward2 = 0 if done2 else -1

        obs1 = [self.agents[0].x, self.agents[0].y]
        obs2 = [self.agents[0].x, self.agents[0].y]

        info = {}

        done = done1 or done2
        return obs1, obs2, reward1, reward2, done, info

    def render(self, mode="human", action1=0, action2=0, reward1=0, reward2=0):
        if mode == "human":
            print(
                f"{Actions(action1): <5}: ({self.agents[0].x},{self.agents[0].y}) reward = {reward1} for agent 1"
            )
            print(
                f"{Actions(action2): <5}: ({self.agents[1].x},{self.agents[1].y}) reward = {reward2} for agent 2"
            )
        else:
            super().render(mode=mode)  # just raise an exception


# create an instance of our custom environment
env = BabyRobotEnv_v2()

# initialize the environment
env.reset()

done = False
while not done:

    # choose a random action
    action1 = env.action_space.sample()
    action2 = env.action_space.sample()

    # take the action and get the information from the environment
    new_state1, new_state2, reward1, reward2, done, info = env.step(
        action1=action1, action2=action2
    )

    env.agents[0].cum_rew += reward1
    env.agents[1].cum_rew += reward2

    # show the current position and reward
    env.render(action1=action1, action2=action2, reward1=reward1, reward2=reward2)
print(f"Cumulative reward of agent1 = {env.agents[0].cum_rew}")
print(f"Cumulative reward of agent2 = {env.agents[1].cum_rew}")
