# from tensorboardcolab import *
#
# tbc = TensorBoardColab()
import tensorboard
import gym
import numpy as np
import tensorflow as tf
import collections
import matplotlib.pyplot as plt

# graphic properties
plt.rcParams['image.cmap'] = 'RdYlGn'
plt.rcParams['figure.figsize'] = [15.0, 6.0]
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 30

env = gym.make('MountainCarContinuous-v0')
# env._max_episode_steps = 10000
env.seed(1)
np.random.seed(1)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    return x / x.sum(axis=0) # only difference


class PolicyNetwork:
    def __init__(self, state_size, action_size, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.learning_rate = tf.placeholder(tf.float32, name="lr")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], "state")
            self.R_t = tf.placeholder(dtype=tf.float32, name="total_rewards")
            self.W1 = tf.get_variable("W1", [self.state_size, 20],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [20], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [20, 20], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [20], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [20, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyperparameters
state_size = 6
action_size = 3

max_episodes = 5000
max_steps = 10001
discount_factor = 0.99
learning_rate = 0.001
learning_rate_value = 0.001  # 0.005
learning_rate_decay = 0.999

render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size)
value_est = ValueNetwork(state_size, action_size, learning_rate=learning_rate_value)

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    score_log = []
    # tb_summary_writer = tbc.get_writer()
    tb_summary_writer = tf.summary.FileWriter('tensorboard/output')
    tb_summary_writer.flush()
    tb_summary_writer.add_graph(sess.graph)
    value_net_loss_summary = tf.Summary()
    policy_net_loss_summary = tf.Summary()
    total_rewards_summary = tf.Summary()
    avg_rewards_per_100_eps_summary = tf.Summary()

    for episode in range(max_episodes):
        state = env.reset()
        state = np.append(state, [0] * (state_size - len(state)))
        state = state.reshape([1, state_size])
        episode_transitions = []
        i_decay = 1.0
        value_net_loss_container = []
        policy_net_loss_container = []

        for step in range(max_steps):
            output = sess.run(policy.output, {policy.state: state})
            # actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            # if actions_distribution[0] == max(actions_distribution):
            #     action = [-0.6]
            # elif actions_distribution[1] == max(actions_distribution):
            #     action = [0.05]
            # else:
            #     action = [0.8]
            # actions_distribution = softmax(actions_distribution[:1])
            # action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            new_value = ((output[0] - 0.0) / (1.0 - 0.0)) * (1.0 - (-1.0)) + (-1.0)
            action = new_value[0]
            next_state, reward, done, _ = env.step([action])

            next_state = np.append(next_state, [0] * (state_size - len(next_state)))
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[0] = 1
            episode_transitions.append(
                Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

            # Calculate TD Target
            value_curr = sess.run(value_est.value_estimate, {value_est.state: state})
            # if value_next #if s' is terminal
            if not done:
                value_next = sess.run(value_est.value_estimate, {value_est.state: next_state})
            else:
                value_next = 0

            td_target = reward + discount_factor * value_next
            td_error = td_target - value_curr

            if learning_rate > 0.0001:
                learning_rate = learning_rate * learning_rate_decay ** episode
            else:
                learning_rate = 0.0001

            # Update the policy estimator
            feed_dict_pol = {policy.state: state, policy.R_t: td_error * i_decay, policy.action: action_one_hot,
                             policy.learning_rate: learning_rate}
            _, loss = sess.run([policy.optimizer, policy.loss], feed_dict_pol)
            policy_net_loss_container.append(loss)

            # Update the value estimator
            feed_dict_val = {value_est.state: state, value_est.R_t: td_target}
            _, loss = sess.run([value_est.optimizer, value_est.loss], feed_dict_val)
            value_net_loss_container.append(loss)

            if done:
                score_log.append(episode_rewards[episode])
                total_rewards_summary.value.add(tag='Total rewards per episode - summary',
                                                simple_value=episode_rewards[episode])
                tb_summary_writer.add_summary(total_rewards_summary, episode)

                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    avg_rewards_per_100_eps_summary.value.add(tag='Last 100 episodes AVG rewards',
                                                              simple_value=average_rewards)
                    tb_summary_writer.add_summary(avg_rewards_per_100_eps_summary, episode)
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > 90:
                    print(' Solved at episode: ' + str(episode))
                    plt.plot(range(len(score_log)), score_log[0:(len(score_log))], 'o')
                    plt.title("Total reward per episode")
                    plt.show()
                    solved = True

                break
            state = next_state
            i_decay = i_decay * discount_factor

        if solved:
            break

        # tensor board writing
        avg_value_loss_summary = np.mean(value_net_loss_container)
        value_net_loss_summary.value.add(tag='Value Network: AVG Loss summary', simple_value=avg_value_loss_summary)
        tb_summary_writer.add_summary(value_net_loss_summary, episode)
        avg_policy_loss_summary = np.mean(policy_net_loss_container)
        policy_net_loss_summary.value.add(tag='Policy Network: AVG Loss summary', simple_value=avg_policy_loss_summary)
        tb_summary_writer.add_summary(policy_net_loss_summary, episode)

    tb_summary_writer.close()
