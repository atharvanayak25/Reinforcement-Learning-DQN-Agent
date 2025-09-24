import torch
import torch.nn as nn
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import os
import matplotlib
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta
import numpy as np
# env = gymnasium.make("CartPole-v1", render_mode="human", use_lidar=True)

DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok = True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameters_set = hyperparameter_set
        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size'] 
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']   
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params',{})
        self.max_episodes = hyperparameters['max_episodes']
        self.enable_double_dqn = hyperparameters['enable_double_dqn']

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.png')


        self.loss_fn = nn.MSELoss()
        self.optimizer = None


    def run(self, is_training = True, render = False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training Starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
        
        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        # epsilon_history = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())


            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)

            epsilon_history = []

            step_count = 0

            best_reward = -9999999

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            policy_dqn.eval()


        for episode in itertools.count():
        # for episode in range(self.max_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device = device)


            terminated = False
            # truncated = False
            episode_reward = 0.0

            
            while (not terminated and episode_reward < self.stop_on_reward):
            # while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else: 
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)


                if is_training:
                    

                    memory.append((state, action, new_state, reward, terminated))


                    step_count += 1

                state = new_state
                
                # Checking if the player is still alive
                # if terminated:
                #     break
            
                rewards_per_episode.append(episode_reward)

            if is_training: 
                if episode_reward > best_reward:
                        # increase_pct = (episode_reward - best_reward) / best_reward * 100
                        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New Best Reward {episode_reward: 0.1f} ({(episode_reward - best_reward)/best_reward*100:+.1f}%) at Episode {episode}, Saving Model..." 
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward
                        # break


                        current_time = datetime.now()
                        if current_time - last_graph_update_time > timedelta(seconds = 10):
                            self.save_graph(rewards_per_episode, epsilon_history)
                            last_graph_update_time = current_time

                # epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                # epsilon_history.append(epsilon)

                if len(memory)>self.mini_batch_size:

                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
    

    
    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure()

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        # plt.subplot_adjust(wspace = 1.0, hspace = 1.0)
        plt.tight_layout()

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
    
    
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # for state, action, new_state, reward, terminated in mini_batch:

        #     if terminated:
        #         target = reward
        #     else:
        #         with torch.no_grad():
        #             target_q = reward + self.discount_factor_g * target_dqn(new_state).max()

        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            
            if self.enable_double_dqn:
                best_action_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
                
            else:
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]


        current_q = policy_dqn(states).gather(dim=1, index = actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()            
        self.optimizer.step()


if __name__ == '__main__':
    # agent = Agent('cartpole1')
    # agent.run(is_training=True, render=True)

    parser = argparse.ArgumentParser(description='Train or Test Model')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training Mode', action = 'store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training = True)
    else:
        dql.run(is_training = False, render = True)

        # env.close()