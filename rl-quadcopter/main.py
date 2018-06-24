import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agents.agent import DDPG
from tasks.takeoff import Task
from collections import deque
                                     # time limit of the episode
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities

num_episodes = 500
target_pos = np.array([0., 0., 100.])
task = Task(target_pos=target_pos, init_pose=init_pose, init_angle_velocities=init_angle_velocities, 
    init_velocities=init_velocities)
agent = DDPG(task) 
worst_score = float('inf')
best_score = float('-inf')

reward_labels = ['episode', 'reward', 'rolling10', 'rolling100']
reward_results = {x : [] for x in reward_labels}

rolling_score_10 = deque(maxlen=10)
rolling_score_100 = deque(maxlen=100)
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    score = 0
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        score += reward
        best_score = max(best_score , score)
        worst_score = min(worst_score , score)
        if done:
            rolling_score_10.append(score)
            rolling_score_100.append(score)
            print("\rEpisode = {:4d}, score = {:7.3f}, best = {:7.3f} , worst = {:7.3f}), rolling = {:7.3f}/{:7.3f}".format(
               i_episode, score, best_score, worst_score, np.mean(rolling_score_10), np.mean(rolling_score_100)), end="")
            break
    reward_results['rolling10'].append(np.mean(rolling_score_10))
    reward_results['rolling100'].append(np.mean(rolling_score_100))
    reward_results['episode'].append(i_episode)
    reward_results['reward'].append(score)

fig, ax = plt.subplots(1, figsize=(15,8))
ax.plot(reward_results['episode'], reward_results['reward'], label='reward/episode')
ax.plot(reward_results['episode'], reward_results['rolling10'], label='10-episode moving average')
ax.plot(reward_results['episode'], reward_results['rolling100'], label='100-episode moving average')
ax.legend()
ax.set_ylabel('reward')
ax.set_xlabel('episodes')
plt.show()