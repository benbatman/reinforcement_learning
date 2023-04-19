########################################

# Generic implementation of an evolution strategy algorithm

# solver = EvolutionStrategy()

# while True:

#   # ask the ES to give us a set of candidate solutions
#   solutions = solver.ask()

#   # create an array to hold the fitness results.
#   fitness_list = np.zeros(solver.popsize)

#   # evaluate the fitness for each given solution.
#   for i in range(solver.popsize):
#     fitness_list[i] = evaluate(solutions[i])

#   # give list of fitness results back to ES
#   solver.tell(fitness_list)

#   # get best parameter, fitness from ES
#   best_solution, best_fitness = solver.result()

#   if best_fitness > MY_REQUIRED_FITNESS:
#     break

#########################################

#############################################
# The least verbose interface is via the optimize method::
#         es.optimize(objective_func)
#         res = es.result

#     More verbosely, the optimization is done using the
#     methods `stop`, `ask`, and `tell`::
#         while not es.stop():
#             solutions = es.ask()
#             es.tell(solutions, [cma.ff.rosen(s) for s in solutions])
#             es.disp()
#         es.result_pretty()

#############################################



import numpy as np 
import cma 
import matplotlib.pyplot as plt 

from datetime import datetime
import sys

from ple import PLE 
from ple.games.flappybird import FlappyBird

import pygame
from pygame import image

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

class CMAES:

  # Initializing all the parameters and variables of the CMAES class
  def __init__(self, num_params, sigma_init=0.10, popsize=200, weight_decay=0.01, modulo=20):
    self.num_params = num_params
    self.sigma_init = sigma_init
    self.popsize = popsize
    self.weight_decay = weight_decay
    self.solutions = None
    self.modulo = modulo
    # Params for CMAEvolutionStrategy: (x0, sigma0, opts)
    # x0 : initial solution, starting point
    # sigma0 : initial standard deviation
    self.es = cma.CMAEvolutionStrategy(self.num_params * [0], self.sigma_init, {'popsize': self.popsize,})

  # Making a method that returns the value of the standard deviation (sigma parameter) in the CMA-ES optimizer
  def rms_stdev(self):
    sigma = self.es.result[6]
    return np.mean(np.sqrt(sigma*sigma))

  # Making a method that asks the CMA-ES optimizer to give us a set of candidate solutions
  def ask(self):
    # ask method gets the candidate solution. 
    # Returns a list of N-dimensional candidate solutions to be evaluated
    self.solutions = np.array(self.es.ask()) 
    return self.solutions

  # Making a method that gives the list of fitness results back to the CMA-ES optimizer
  def tell(self, reward_table_result):
    reward_table = -np.array(reward_table_result)
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay
    self.es.tell(self.solutions, (reward_table).tolist())

  # Making a method that returns the current result from the Evolution Strategy algorithm
  def current_param(self):
    return self.es.result[5]

  # Making a method that sets the parameters obtained by each optimizer into an array, but unnecessary to do that for CMAES so pass
  def set_mu(self, mu):
    pass

  def disp(self):
    return self.es.disp(modulo=self.modulo)

  # Making a method that returns the best set of optimized parameters
  def best_param(self):
    # CMAEvolutionStrategyResult returns a namedtuple with 'xbest' as the first value
    return self.es.result[0]

  # Making a method that returns all the results (NamedTuple)
  def result(self):
    result = self.es.result
    return (result[0], -result[1], -result[1], result[6])


HISTORY_LENGTH = 1 


# Create Flappy Bird environment wrapped in an OpenAI gym type API
# For each pipe it passes through it gains a positive reward of +1. 
# Each time a terminal state is reached it receives a negative reward of -1.
class Env:
    def __init__(self):
        self.game = FlappyBird(pipe_gap=125) 
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.env.init()
        self.env.getGameState = self.game.getGameState

        self.action_map = self.env.getActionSet() # [None, 119]
        self.frame_number = 0

    def step(self, action):
        action = self.action_map[action] # get action
        reward = self.env.act(action) # perform action and get reward 
        done = self.env.game_over() # Is the game over 
        obs = self.get_observation()

        # Save frames if agent is playing the game
        if len(sys.argv) > 3 and sys.argv[1] == 'play' and sys.argv[3] == 'save-frames':
            frame = pygame.pixelcopy.make_surface(self.env.getScreenRGB())
            image.save(frame, f"cma_flappy_frames/flappy_cma_frame_{self.frame_number}.jpeg") 
            self.frame_number += 1

        return obs, reward, done 

    def reset(self):
        self.env.reset_game()
        return self.get_observation()
    
    def get_observation(self):
        # game state returns a dictionary which describes the meaning of each value 
        # we only want the values 
        # Turn dictionary of obsevations to numpy array similar to OpenAI gym
        obs = self.env.getGameState()
        return np.array(list(obs.values()))

    def set_display(self, boolean_value):
        self.env.display_screen = boolean_value

    def get_screen(self):
        self.env.getScreenRGB()

    # Make a global environment to be used throughout the script
env = Env()

D = len(env.reset()) * HISTORY_LENGTH # input dimension
M = 50  # hidden layer size
K = 2 # output (number of actions) 

def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    e = np.exp(a-c)
    return e / e.sum(axis=-1, keepdims=True)

def relu(x):
    return x * (x > 0)

class ANN:
    def __init__(self, D, M, K, f=relu):
        self.D = D 
        self.M = M 
        self.K = K 
        self.f = f 

    def init(self):
        D, M, K = self.D, self.M, self.K 
        self.W1 = np.random.randn(D,M) / np.sqrt(D) 
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K) / np.sqrt(M) 
        self.b2 = np.zeros(K) 

    def forward(self, X):
        Z = self.f(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2)

    def sample_action(self, x):
        # Assume input is a single state of size (D,)
        # First make it (N,D) to fit ML conventions
        X = np.atleast_2d(x) 
        P = self.forward(X) 
        p = P[0] # get first row
        # return np.random.choice(len(p), p=p)
        return np.argmax(p) 

    def get_params(self):
        # Return a flat array of parameters 
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def get_params_dict(self):
        return {
            'W1': self.W1, 
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }

    def set_params(self, params):
        # Params is a flat list 
        # Unflatten into individual weights 
        D, M, K = self.D, self.M, self.K 
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M:D * M + M]
        self.W2 = params[D * M + M:D * M + M + M * K].reshape(M,K)
        self.b2 = params[-K:]


def evolution_strategy(
    f, 
    num_iters,
    cma, 
    ):

    reward_per_iteration = np.zeros(num_iters)

    for t in range(num_iters):
        t0 = datetime.now()
        #while not cma.es.stop():
        solutions = cma.ask() # ask strategy for solutions
        #print(np.array(solutions).shape)
        fitness_list = np.zeros(cma.es.popsize)
        
        for i in range(cma.es.popsize):
            # Input candidate solutions (future model params) into our reward function
            fitness_list[i] = f(solutions[i])
        
        cma.tell(fitness_list) #[f(x) for x in solutions]

        m = fitness_list.mean()
        reward_per_iteration[t] = m
        print(f"Iter: {t}, Avg Reward: {m} Max Reward: {fitness_list.max()} Episode Length: {episode_length} Duration: {datetime.now()-t0}")

                
    return cma.best_param(), reward_per_iteration


def reward_function(params): 
    model = ANN(D, M, K)
    model.set_params(params) 

    # Play one episode and return the total reward 
    episode_reward = 0
    global episode_length
    episode_length = 0 
    done = False 
    obs = env.reset()
    obs_dim = len(obs) 
    if HISTORY_LENGTH > 1:
        state = np.zeros(HISTORY_LENGTH*obs_dim)
        state[-obs_dim:] = obs 
    else:
        state = obs
    
    while not done: # while episode hasn't finished yet 
        # Get the action 
        action = model.sample_action(state) 

        # Perform the action and get the observation, reward and done flag
        obs, reward, done = env.step(action)

        # Update total reward 
        episode_reward += reward 
        episode_length += 1 

        # Update state 
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:] = obs 
        else:
            state = obs 
    return episode_reward


if  __name__ == '__main__':
    model = ANN(D,M,K) 

    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        # Play with a saved model 
        j = np.load('es_flappy_results_cma_es.npz')
        best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])

        # In case initial shapes are not correct 
        D, M = j['W1'].shape 
        K = len(j['b2'])
        model.D, model.M, model.K = D, M, K
    else: 
    # Train and save
        model.init()
        params = model.get_params()
        num_iters=175
        cma_es = CMAES(num_params=params)
        print(f"Total iterations agent will train: {num_iters}")
        best_params, rewards = evolution_strategy(
            f=reward_function,
            num_iters=num_iters,
            cma = cma_es
        )

        model.set_params(best_params) 
        np.savez('es_flappy_results_cma_es.npz',
        train=rewards, 
        **model.get_params_dict(),
        )

        #cma_es.es.logger.plot() # plot cma_es info


    env.set_display(True) 
    for _ in range(int(sys.argv[2])):
        print(f"Test: {reward_function(best_params)}")

