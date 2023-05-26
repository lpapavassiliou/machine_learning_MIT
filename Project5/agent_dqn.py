"""Tabular QL agent"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 300
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# nn achitecture definition for the policy network pi(state) = (action, object)
class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for each (action, object) tuple given an input state vector.
    The input is the state coordinates (int), while the outputs are Q values (float).
    
    state               Q values
    ----------------------------------
                    [                 ]
                    |  Q(s, (0,0))    |
                    |  Q(s, (0,1))    |
        network     |     ...         |
    s ------------> |  Q(s, (1,0))    |
                    |     ...         |
                    [  Q(s,(n_a,n_o)) ]
                    
    """

    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, object_dim*action_dim)

    def forward(self, x):
        x = F.relu(self.state_encoder(x))
        x = F.relu(self.hidden_layer(x))
        x = F.relu(self.output_layer(x))
        q_values = x
        return q_values

def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index

def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    return index // NUM_OBJECTS, index % NUM_OBJECTS


def epsilon_greedy(state_vector, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (torch.FloatTensor): extracted vector representation
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    # TODO Your code here
    import random
    
    if random.random() < epsilon:
        # pick random action
        action_index, object_index = random.randint(0, NUM_ACTIONS-1), random.randint(0, NUM_OBJECTS-1)
        
    else:
        # forward the network to get Q values of current state
        with torch.no_grad():
            Q_values  = Q_network.forward(state_vector)
        
        # pick the action associated with the highest Q value
        action_index, object_index = index2tuple(torch.argmax(Q_values))
        
    return action_index, object_index



# pragma: coderesponse template
def deep_q_learning(current_state_vector, action_index, object_index, reward,
                    next_state_vector, terminal):
    """Updates the weights of the DQN for a given transition

    Args:
        current_state_vector (torch.FloatTensor): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (torch.FloatTensor): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    # TODO Your code here
    
    # compute target y
    if terminal:   # consider the reward only
        y = reward
    else:          # consider the reward plus the discounted future reward
        with torch.no_grad():
            q_next_state = Q_network(next_state_vector)
        y = reward + GAMMA*torch.max(q_next_state)

    # compute loss
    loss = (Q_network(current_state_vector)[tuple2index(action_index, object_index)] - y)**2

    # backpropagate and update weights
    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()
    


def run_episode(for_training):
    """
        Runs one episode
        If for training, update Q function
        If for testing, computes and return cumulative discounted reward
    """
    # TODO Your code here
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = 0

    # initialize the game for the episode
    current_room_desc, current_quest_desc, terminal = framework.newGame()
    time = 0
    
    # simulate the game
    while not terminal:
        
        # Build current state from the description
        current_state_desc = current_room_desc + current_quest_desc
        current_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(current_state_desc, dictionary))

        # Choose next action and execute
        action_idx, object_idx = epsilon_greedy(current_state_vector, epsilon)
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(current_room_desc, current_quest_desc, action_idx, object_idx)

        # Build next state from description
        next_state_desc = next_room_desc + next_quest_desc
        next_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(next_state_desc, dictionary))
        
        if for_training:
            # update Q-function
            deep_q_learning(current_state_vector, action_idx, object_idx, reward, next_state_vector, terminal)
            
        if not for_training:
            # update reward
            epi_reward += (GAMMA**time)*reward

        # prepare next step
        time += 1

    if not for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global Q_network
    global policy_optimizer
    Q_network = DQN(state_dim, NUM_ACTIONS, NUM_OBJECTS)
    policy_optimizer = optim.SGD(Q_network.parameters(), lr=ALPHA)

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)
    state_dim = len(dictionary)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()


run()