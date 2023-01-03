import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from enum import IntEnum
from random import uniform

""" simple helper class to enumerate actions in the grid levels """


class Actions(IntEnum):
    Stay = 0
    North = 1
    East = 2
    South = 3
    West = 4

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
        self.x = 0
        self.y = 0

    def step(self, action):
        obs = np.array([self.x, self.y])
        reward = -1
        done = True
        info = {}
        return obs, reward, done, info

    def reset(self):
        # reset Baby Robot's position in the grid
        self.x = 0
        self.y = self.max_y
        return np.array([self.x, self.y])

    def render(self, mode):
        pass


class BabyRobotEnv_v2(BabyRobotEnv_v1):

    metadata = {"render_modes": ["human"]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # the start and end positions in the grid
        # - by default these are the top-left and bottom-right respectively
        self.start = kwargs.get("start", [0, 2])
        self.start2 = kwargs.get("start", [self.max_x, self.max_y])
        self.end = kwargs.get("end", [1, 0])

        # Baby Robot's initial position
        # - by default this is the grid start
        self.initial_pos = kwargs.get("initial_pos", self.start)

        # Baby Robot's position in the grid
        self.x = self.initial_pos[0]
        self.y = self.initial_pos[1]

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

    def step(self, action):

        # take the action and update the position
        self.take_action(action)
        obs = np.array([self.x, self.y])

        # set the 'done' flag if we've reached the exit
        done = (self.x == self.end[0]) and (self.y == self.end[1])

        # get -1 reward for each step
        # - except at the terminal state which has zero reward
        reward = 0 if done else -1

        info = {}
        return obs, reward, done, info

    def render(self, mode="human", action=0, reward=0):
        if mode == "human":
            print(f"{Actions(action): <5}: ({self.x},{self.y}) reward = {reward}")
        else:
            super().render(mode=mode)  # just raise an exception


from stable_baselines3.common.env_checker import check_env

# create an instance of our custom environment
env = BabyRobotEnv_v2()

# initialize the environment
env.reset()

print()

done = False
while not done:

    # choose a random action
    action = env.action_space.sample()

    # take the action and get the information from the environment
    new_state, reward, done, info = env.step(action)

    # show the current position and reward
    env.render(action=action, reward=reward)
