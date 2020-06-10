import gym

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

env = TravianEnv(village_info_dict_of_dicts, building_info, 10, 500000)
env.reset()
n_actions = env.village_n*22+1
state_dim = env.village_n*34


import torch
import torch.nn as nn
import torch.nn.functional as F

network = nn.Sequential()

network.add_module('layer1', nn.Linear(state_dim, 16))
network.add_module('relu1', nn.ReLU())
network.add_module('layer2', nn.Linear(16, 16))
network.add_module('relu1', nn.ReLU())
network.add_module('layer2', nn.Linear(16, 16))
network.add_module('relu1', nn.ReLU())
network.add_module('layer2', nn.Linear(16, 16))
network.add_module('relu1', nn.ReLU())
network.add_module('layer2', nn.Linear(16, 16))
network.add_module('relu1', nn.ReLU())
network.add_module('layer3', nn.Linear(16, n_actions))


def get_action(state, epsilon=0, cheating=0, waiting=0):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
    """
    should_explore3 = np.random.binomial(n=1, p=waiting)
    if should_explore3:
      return 22

    should_explore2 = np.random.binomial(n=1, p=cheating)
    if should_explore2:
      chosen_action = np.random.choice(range(10, 18))
      return int(chosen_action)

    if env.X['village0']['time_remaining'] == 0 and env.count_time_pace() == 0:
      state = torch.tensor(state, dtype=torch.float32)
      q_values = network(state).detach().numpy()
      for act in range(n_actions):
        if not env.is_available_and_rr(act)[0]:
          q_values[act] = 0
      greedy_action = np.argmax(q_values[:21])

      should_explore = np.random.binomial(n=1, p=epsilon)

      if should_explore:
        chosen_action = np.random.choice(range(n_actions))
      else:
        chosen_action = greedy_action

    else:
      state = torch.tensor(state, dtype=torch.float32)
      q_values = network(state).detach().numpy()
      for act in range(n_actions):
        if not env.is_available_and_rr(act)[0]:
          q_values[act] = 0

      greedy_action = np.argmax(q_values)

      should_explore = np.random.binomial(n=1, p=epsilon)

      if should_explore:
        chosen_action = np.random.choice(range(n_actions))
      else:
        chosen_action = greedy_action

    return int(chosen_action)


def compute_td_loss(states, actions, rewards, next_states, is_done, gamma=0.999, check_shapes=False):
    """ Compute td loss using torch operations only """
    states = torch.tensor(
        states, dtype=torch.float32)    # shape: [batch_size, state_size]
    actions = torch.tensor(actions, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, state_size]
    next_states = torch.tensor(next_states, dtype=torch.float32)
    is_done = torch.tensor(is_done, dtype=torch.uint8)  # shape: [batch_size]

    # get q-values for all actions in current states
    predicted_qvalues = network(states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[
      range(states.shape[0]), actions
    ]

    # compute q-values for all actions in next states
    predicted_next_qvalues = network(next_states)

    # compute V*(next_states) using predicted next q-values
    next_state_values = torch.max(predicted_next_qvalues, dim = -1)[0]
    assert next_state_values.dtype == torch.float32

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    target_qvalues_for_actions = rewards + gamma * next_state_values

    # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = torch.where(
        is_done, rewards, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    return loss

opt = torch.optim.Adam(network.parameters(), lr = 0.001)


def generate_session(env, epsilon=0, cheating=0, waiting=0, train=False):
    """play env with approximate q-learning agent and train it at the same time"""
    total_reward = 0
    s = env.reset()
    done = False
    logs = [np.array(s)]
    while not done:
        a = get_action(s, epsilon=epsilon, cheating=cheating, waiting=waiting)
        all_info = env.step(a)
        wow = list(env.X['village0'].values())[:6]
        wow.append(env.res_growths[0])
        wow.append([env.granary_capacities[0]])
        wow.append([env.storage_capacities[0]])
        wow.append([env.boost[0]])
        wow.append([env.X['village0']['time_remaining']])
        wow.append([env.current_time])
        to_print = np.array([item for sublist in wow for item in sublist])
        next_s = to_print
        r = all_info[1]
        done = all_info[2]
        if train:
            opt.zero_grad()
            compute_td_loss([s], [a], [r], [next_s], [done]).backward()
            opt.step()
        if any(s[i] != next_s[i] for i in range(len(list(s)))):
            logs.append(np.insert(s, 0, a))
        total_reward += r
        s = next_s
    return total_reward, s, logs

rew = 0
epsilon = 0.5
cheating = 0
waiting = 0
for epoch in range(1050):
    session_rewards = generate_session(env, epsilon=epsilon, cheating=cheating,
                                       waiting = waiting, train=True)
    print(generate_session(env)[0], session_rewards[0], 'epsilon = ',
          epsilon, ', cheating = ', cheating, ', waiting = ', waiting)
    if waiting <= 0.15:
      waiting *=0.999
    else:
      waiting *= 0.94
    epsilon *= 0.999
    cheating *= 0.99
    waiting *= 0.92
    if generate_session(env)[0] > 93:
        print("You Win!")
        print(session_rewards[1])
        break

final_logs = generate_session(env)[2]
col = ['action', 'farm1', 'farm2', 'farm3', 'farm4', 'farm5', 'farm6', 'mine1', 'mine2',
       'mine3', 'mine4', 'lumber1', 'lumber2', 'lumber3', 'lumber4', 'pit1',
       'pit2', 'pit3', 'pit4', 'granary', 'storage', 'main',
       'corn', 'iron', 'wood', 'loam', 'corn/hour',
       'iron/hour', 'wood/hour', 'loam/hour',
       'granary capacity', 'storage capacity', 'boost',
       'time remaining', 'current time']
df = pd.DataFrame(final_logs, columns=col)
act_s = list(df['action'].values)
new_act_s = []
for j in act_s:
    j = int(j)
    if j < 6:
        new_j = 'improve farm № ' + str(j + 1)
    elif j < 10:
        new_j = 'improve mine № ' + str(j - 6 + 1)
    elif j < 14:
        new_j = 'improve lumber № ' + str(j - 10 + 1)
    elif j < 18:
        new_j = 'improve pit № ' + str(j - 14 + 1)
    elif j < 21:
        b = ['granary', 'storage', 'main'][j - 18]
        new_j = 'improve ' + b
    elif j == 21:
        new_j = 'use gold'
    elif j == 22:
        new_j = 'wait'

    new_act_s.append(new_j)
df['action'] = new_act_s
df.drop(0)

df.to_excel('logs', index = False)