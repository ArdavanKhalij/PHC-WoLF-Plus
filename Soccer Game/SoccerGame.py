import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from enum import IntEnum
from random import uniform, randint
from copy import copy
from stable_baselines3.common.env_checker import check_env


class Actions(IntEnum):
    North = 0
    East = 1
    South = 2
    West = 3
    Stay = 4

    # get the enum name without the class
    def __str__(self):
        return self.name


class Agent:
    def __init__(self, startx, starty, env):

        self.cum_rew = 0
        self.x = startx
        self.y = starty

        # define the maximum x and y values
        self.max_x = env.width - 1
        self.max_y = env.height - 1

        self.has_ball = False

        self.env = env

    def next_pos(self, action):
        x = copy(self.x)
        y = copy(self.y)

        if action == Actions.North:
            if y - 1 >= 0:
                y -= 1
        elif action == Actions.South:
            if y + 1 < self.max_y:
                y += 1
        elif action == Actions.West:
            if x - 1 > 0:
                x -= 1
        elif action == Actions.East:
            if x + 1 < self.max_x:
                x += 1

        return [x, y]

    def take_action(self, action):
        """apply the supplied action"""

        # move in the direction of the specified action
        if action == Actions.North:
            if self.y - 1 >= 0:
                self.y -= 1
        elif action == Actions.South:
            if self.y + 1 <= self.max_y:
                self.y += 1
        elif action == Actions.West:
            if self.x - 1 >= 0:
                self.x -= 1
            elif self.x == 0 and self.y > 0 and self.y < 3 and self.has_ball:
                self.x -= 1
        elif action == Actions.East:
            if self.x + 1 <= self.max_x:
                self.x += 1
            elif self.x == self.max_x and self.y > 0 and self.y < 3 and self.has_ball:
                self.x += 1


class BabyRobotEnv_v1(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()

        # dimensions of the grid
        self.width = kwargs.get("width", 5)
        self.height = kwargs.get("height", 4)

        # define the maximum x and y values
        self.max_x = self.width - 1
        self.max_y = self.height - 1

        # there are 5 possible actions: move N,E,S,W and stay
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

        # print(f"Position of agent 1 : {[self.x1,self.y1]}")
        # print(f"Position of agent 2 : {[self.x2,self.y2]}")

        return np.array([self.x1, self.y1, self.x2, self.y2])

    def render(self, mode):
        pass


class BabyRobotEnv_v2(BabyRobotEnv_v1):

    metadata = {"render_modes": ["human"]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.winner = 0
        # the start and end positions in the grid
        # - by default these are the top-left and bottom-right respectively
        self.start1 = kwargs.get(
            "start", [randint(0, self.max_x / 2), randint(0, self.max_y)]
        )
        self.start2 = kwargs.get(
            "start", [randint(self.max_x / 2, self.max_x), randint(0, self.max_y)]
        )

        # Baby Robot's initial position
        # - by default this is the grid start
        self.initial_pos1 = kwargs.get("initial_pos", self.start1)
        self.initial_pos2 = kwargs.get("initial_pos", self.start2)

        # Initialize agents
        self.agents = []
        self.agents.append(
            Agent(startx=self.initial_pos1[0], starty=self.initial_pos1[1], env=self)
        )
        self.agents.append(
            Agent(startx=self.initial_pos2[0], starty=self.initial_pos2[1], env=self)
        )

    def reset(self):
        # reset Baby Robot's position in the grid
        self.agents[0].x = randint(0, self.max_x / 2)
        self.agents[0].y = randint(0, self.max_y)

        self.agents[1].x = randint(self.max_x / 2, self.max_x)
        self.agents[1].y = randint(0, self.max_y)

        # print(f"Position of agent 1 : {[self.agents[0].x,self.agents[0].y]}")
        # print(f"Position of agent 2 : {[self.agents[1].x,self.agents[1].y]}")

        rand = randint(0, 1)

        if rand == 0:
            self.agents[0].has_ball = True
            self.agents[1].has_ball = False
        else:
            self.agents[0].has_ball = False
            self.agents[1].has_ball = True

        return np.array(
            [self.agents[0].x, self.agents[0].y, self.agents[1].x, self.agents[1].y]
        )

    def same_pos(self):
        # Check if agents have the same position
        return (
            self.agents[0].x == self.agents[1].x
            and self.agents[0].y == self.agents[1].y
        )

    def change_posession(self):
        # Make the ball go to the other agent
        if self.agents[0].has_ball:
            self.agents[1].has_ball = True
            self.agents[0].has_ball = False
        else:
            self.agents[1].has_ball = False
            self.agents[0].has_ball = True

    def make_moves(self, first, action1, action2):
        if first == 0:
            # Agent 1 plays first
            if self.agents[0].has_ball:
                if self.agents[0].next_pos(action1) == [
                    self.agents[1].x,
                    self.agents[1].y,
                ]:  # Check if the agent tries to move to a defender
                    self.change_posession()
                    # Give the posession to the defender
                else:
                    self.agents[0].take_action(action1)
                    if self.agents[1].next_pos(action2) != [
                        self.agents[0].x,
                        self.agents[0].y,
                    ]:  # Check if the agent tries to a valid place
                        self.agents[1].take_action(action1)
                        # Make the move

            else:
                if self.agents[0].next_pos(action1) != [
                    self.agents[1].x,
                    self.agents[1].y,
                ]:
                    self.agents[0].take_action(action1)
                    if self.agents[1].next_pos(action2) == [
                        self.agents[0].x,
                        self.agents[0].y,
                    ]:  # Check if the agent tries to move to a defender
                        self.change_posession()
                    else:
                        self.agents[1].take_action(action2)

        else:
            if self.agents[0].next_pos(action1) != [
                self.agents[1].x,
                self.agents[1].y,
            ]:
                self.agents[0].take_action(action1)
                if self.agents[1].next_pos(action2) == [
                    self.agents[0].x,
                    self.agents[0].y,
                ]:  # Check if the agent tries to move to a defender
                    self.change_posession()
                else:
                    self.agents[1].take_action(action2)
            else:
                if self.agents[0].has_ball:
                    if self.agents[0].next_pos(action1) == [
                        self.agents[1].x,
                        self.agents[1].y,
                    ]:  # Check if the agent tries to move to a defender
                        self.change_posession()
                        # Give the posession to the defender
                    else:
                        self.agents[0].take_action(action1)
                        if self.agents[1].next_pos(action2) != [
                            self.agents[0].x,
                            self.agents[0].y,
                        ]:  # Check if the agent tries to a valid place
                            self.agents[1].take_action(action2)
                            # Make the move

    def check_agents(self):
        # Check if agents end up in the same spot and change the possession of the ball if needed
        if self.same_pos():
            self.change_posession()

    def check_goal(self):
        for agent in self.agents:
            if agent.x > self.max_x:
                return 0
            elif agent.x < 0:
                return 1

    def step(self, action1, action2):

        done1, done2 = False, False

        first_to_move = randint(0, 1)

        self.make_moves(first_to_move, action1, action2)

        # set the 'done' flag if one of the agents has scoared a goal
        scoring_agent = self.check_goal()

        if scoring_agent == 0:
            reward1 = 10
            reward2 = -10
            done1 = True
            self.winner = 1
        elif scoring_agent == 1:
            reward2 = 10
            reward1 = -10
            done2 = True
            self.winner = 2
        else:
            reward1 = 0
            reward2 = 0

        # print(f"Agent 1 x : {self.agents[0].x} y : {self.agents[0].y}")
        # print(f"Agent 2 x : {self.agents[1].x} y : {self.agents[1].y}")

        obs1 = [self.agents[0].x, self.agents[0].y]
        obs2 = [self.agents[1].x, self.agents[1].y]

        info = {}

        # print(f"obs1 : {obs1}")
        # print(f"obs2 : {obs2}")
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

            if self.agents[0].has_ball:
                print("Ball posession : agent 1")
            else:
                print("Ball posession : agent 2")
        else:
            super().render(mode=mode)  # just raise an exception


# env = BabyRobotEnv_v2()

# env.reset()
# for i in range(0, 10):
#     env.reset()

#     env.agents[0].cum_rew = 0
#     env.agents[1].cum_rew = 0
#     done = False
#     while not done:

#         # choose a random action
#         action1 = env.action_space.sample()
#         action2 = env.action_space.sample()

#         # take the action and get the information from the environment
#         new_state1, new_state2, reward1, reward2, done, info = env.step(
#             action1=action1, action2=action2
#         )

#         env.agents[0].cum_rew += reward1
#         env.agents[1].cum_rew += reward2

#         # show the current position and reward
#         env.render(action1=action1, action2=action2, reward1=reward1, reward2=reward2)
#     print(f"Cumulative reward of agent1 = {env.agents[0].cum_rew}")
#     print(f"Cumulative reward of agent2 = {env.agents[1].cum_rew}")
#     print(" ")
#     print(" ")
