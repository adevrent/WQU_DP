import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

# Define the parameters for the problem
num_states = 10  # M
state_space = np.linspace(0.5, 9.5, num_states)
action_space = np.array([0, 1, 2])
learning_rate = 1e-5  # Learning rate in weight updates

# Define the transition dynamics
def transition(state, action):
  demand = np.random.normal(loc=0.5, scale=0.3)
  next_state = max(min(state + action - demand, 10), 0)
  return next_state

# Define the reward function
def reward(state, action):
  reward = 2.0 * state - 1.0 * action
  return reward

# Feature function (for quadratic functions)
def feature_function(state):
  return np.array([1, state, state**2])

# function approximation (quadratic function)
def fun_approximation(state, weights):
  return weights.dot(feature_function(state))

# Compute empirical expectation of g(s,a) + \gamma * V(s',weights)
def expectation_Qfun(state, action, weights, discount_factor=0.7):
  n_sim = 20
  Q_value = 0
  for _ in range(n_sim):
    Q_value += 
    ######COMPLETE HERE!!!!!!#######
  return (Q_value / n_sim)

# Perform fitted value iteration
def fitted_value_iteration(state_space, action_space,  max_iterations=1001, epsilon=0.1):
  num_states = len(state_space)
  num_actions = len(action_space)
  weights = np.array([0.1, 0.1, 0.1])

  for step in range(max_iterations):
    # Update v[i]
    v = np.zeros(num_states)
    for i in range(num_states):
      v[i] = max(expectation_Qfun(state_space[i], action, weights) for action in action_space)     
    
    # Update weights using gradient descent
    for _ in range(500):
      gradient_FVI = np.zeros(3)
      for i in range(num_states):
        gradient_FVI += 
        ######COMPLETE HERE!!!!!!#######
      weights -= learning_rate * gradient_FVI

    if step % 10 == 0:
      print("iteration: %d, grad norm: %.2f" % (step, np.linalg.norm(gradient_FVI)))
    if np.linalg.norm(gradient_FVI) < epsilon:
      break
  print("Stop. Gradient norm: %.2f" % np.linalg.norm(gradient_FVI))
  return weights, v

# Print the optimal weights, value function, and the optimal policy at s=1.
optimal_weights, optimal_v = fitted_value_iteration(state_space, action_space)
print("Optimal Weights:")
print(optimal_weights)

optimal_value_function = [fun_approximation(state, optimal_weights) for state in state_space]
print("\nOptimal Value Function:")
print(optimal_value_function)
plt.plot(state_space, optimal_value_function)
plt.xticks(state_space)
plt.scatter(state_space, optimal_v, color="red")

Q_Value_at_1 = [expectation_Qfun(1, action, optimal_weights) for action in action_space]
optimal_action_at_1 = np.argmax(Q_Value_at_1)  
print("\nThe optimal action at s=1 is", action_space[optimal_action_at_1])
