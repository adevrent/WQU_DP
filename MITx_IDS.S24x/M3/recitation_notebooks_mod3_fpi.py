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
  n_sim = 3  # for acceleration
  Q_value = 0
  for _ in range(n_sim):
    Q_value += reward(state, action) + discount_factor * fun_approximation(transition(state, action), weights)
    ######COMPLETE HERE!!!!!!#######
  return (Q_value / n_sim)

# Compute empirical expectation of a fixed-policy value function
def Vfun_fixed_policy(state, policy, weights, discount_factor=0.7):
  n_sim = 10
  E_V_value = 0
  for _ in range(n_sim):
    n_traj = 33
    ######COMPLETE HERE!!!!!!#######
    state_list = [state]
    for _ in range(n_traj):
      st, act = state_list[-1], policy(state_list[-1])
      state_list.append(transition(st, act)) 
    V_value = np.array([discount_factor**(i) * reward(s, policy(s)) for (i, s) in enumerate(state_list)]).sum()
    ######COMPLETE HERE!!!!!!#######
    E_V_value += V_value
  return (E_V_value / n_sim)

# Perform fitted policy iteration
def fitted_policy_iteration(state_space, action_space,  max_iterations=501, epsilon=0.1):
  num_states = len(state_space)
  num_actions = len(action_space)
  weights = np.array([0.1, 0.1, 0.1])

  best_grad = 1e10
  for step in range(max_iterations):
    # Update v[i]
    v = np.zeros(num_states)
    for i in range(num_states):
      def optimal_policy(st):
        act = np.argmax([expectation_Qfun(st, action, weights) for action in action_space])
        return act
      v[i] = Vfun_fixed_policy(state_space[i], optimal_policy, weights)
    
    # Update weights using gradient descent
    for _ in range(500):
      gradient_FPI = np.zeros(3)
      for i in range(num_states):
        gradient_FPI += 2 * feature_function(state_space[i]) * (fun_approximation(state_space[i], weights) - v[i])
        ######COMPLETE HERE!!!!!!#######
      weights -= learning_rate * gradient_FPI
      if np.linalg.norm(gradient_FPI) < np.linalg.norm(best_grad):
        best_grad = gradient_FPI
        best_weights = weights.copy()
        best_v = v.copy()

    if step % 10 == 0:
      print("iteration: %d, grad norm: %.2f, best grad norm: %.2f" 
            % (step, np.linalg.norm(gradient_FPI), np.linalg.norm(best_grad)))
    if np.linalg.norm(gradient_FPI) < epsilon:
      break
  print("Stop. Gradient norm: %.2f" % np.linalg.norm(best_grad))
  return best_weights, best_v

# Print the optimal weights, value function, and the optimal policy at s=1.
optimal_weights, optimal_v = fitted_policy_iteration(state_space, action_space)
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
