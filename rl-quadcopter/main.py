import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agents.agent import MavicAir_Agent
from task import Task

num_episodes = 1000
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)
agent = MavicAir_Agent(task) 

scores = []
fig, ax = plt.subplots(1)
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.score, agent.best_score, agent.noise_scale), end="")
            scores.append(agent.score)
            break
ax.plot(scores)
plt.show()