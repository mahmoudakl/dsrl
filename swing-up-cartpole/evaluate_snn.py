import os
import sys
import time
import csv
from datetime import datetime
from importlib import import_module
from tqdm import trange

import torch
import torch.nn.functional as F

from typing import Any
import gymnasium as gym
import numpy as np
import tensorflow as tf
from numpy.random import SeedSequence
from yaml import dump
from memory_buffer import ReplayBuffer
from model import TD3CriticNetwork, TD3ActorDSNN

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.others.environment import EnvironmentBatched
from Environments import ENV_REGISTRY, register_envs
from SI_Toolkit.computation_library import TensorFlowLibrary
from Utilities.csv_helpers import save_to_csv
from Utilities.generate_plots import generate_experiment_plots
from Utilities.utils import ConfigManager, CurrentRunMemory, OutputPath, SeedMemory, get_logger, nested_assignment_to_ordereddict


sys.path.append(os.path.join(os.path.abspath("."), "CartPoleSimulation"))  # Keep allowing absolute imports within CartPoleSimulation subgit
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
register_envs()  # Gym API: Register custom environments
logger = get_logger(__name__)

# td3 parameters
actor_learning_rate = 0.001
critic_learning_rate = 0.001
tau = 0.005
layer1_size = 400
layer2_size = 300
noise = 0.1
gamma = 0.99
warmup = 1000
batch_size = 100
learning_starts = 1000
update_actor_interval = 2
update_target_interval = 2
buffer_size = int(2e5)
normalize = False
episode_counter = 0
learn_step_counter = 0
policy_learn_step_counter = 0
time_step = 0

# snn parameters
alpha = 0.5
beta = 0.5
weight_scale = 1
threshold = 2.5
sim_time = 5
two_neuron_encoding = True
spiking = True


def normalize_state(state, max_obs):
    if two_neuron_encoding:
        two_neuron_max_obs = np.array([val for val in max_obs for _ in (0, 1)])
        return torch.tensor(state / two_neuron_max_obs, dtype=torch.float)

def choose_action(actor, observation, max_action, min_action, n_actions, max_obs):
    global time_step
    if normalize:
        observation = normalize_state(observation, max_obs)
        state = observation.clone().to(device)
    else:
        state = torch.tensor(observation, dtype=torch.float).clone().to(device)
    if spiking:
        state = state.unsqueeze(0).to(device)
        mu = actor.forward(state)[0].squeeze(0).to(device)
    else:
        mu = actor.forward(state).to(device)

    mu_prime = mu + torch.tensor(np.random.normal(scale=noise), dtype=torch.float,
                                 device=device).to(device)

    mu_prime = torch.clamp(mu_prime * max_action[0], min_action[0], max_action[0])
    time_step += 1

    action = mu_prime.cpu().detach().numpy()
    action = action.astype(np.float32)
    return action


def transform_state(state):
    state_ = []
    for i in state:
        if i > 0:
            state_.append(i)
            state_.append(0)
        else:
            state_.append(0)
            state_.append(abs(i))
    return np.array(state_)


def one_neuron_encoding(state):
    state_ = []
    for i in range(len(state)):
        if i == 0:
            continue
        else:
            if i % 2 == 0:
                state_.append(state[i])
            else:
                state_.append(-state[i])
    return np.array(state_)


def update_max_obs(state, max_obs):
    max_obs = np.maximum(max_obs, np.abs(state))
    return max_obs


def run_data_generator(controller_name: str, environment_name: str, config_manager: ConfigManager,
                       record_path=None):
    global time_step, episode_counter
    models = [file for file in os.listdir('snn_results/') if file.endswith(".pt")]
    models.sort()

    # Generate seeds and set timestamp
    timestamp = datetime.now()
    seed_entropy = config_manager("config")["seed_entropy"]
    if seed_entropy is None:
        seed_entropy = int(timestamp.timestamp())
        logger.info("No seed entropy specified. Setting to posix timestamp.")

    num_experiments = config_manager("config")["num_experiments"]
    seed_sequences = SeedSequence(entropy=seed_entropy).spawn(num_experiments)
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")

    controller_short_name = controller_name.replace("controller_", "").replace("_", "-")
    optimizer_short_name = config_manager("config_controllers")[controller_short_name]["optimizer"]
    optimizer_name = "optimizer_" + optimizer_short_name.replace("-", "_")
    CurrentRunMemory.current_optimizer_name = optimizer_name
    all_metrics = dict(
        total_rewards = [],
        timeout = [],
        terminated = [],
        truncated = [],
    )

    best_average = -np.inf
    best_average_after = np.inf
    smoothed_scores = []

    # Generate new seeds for environment and controller
    seeds = seed_sequences[0].generate_state(3)
    SeedMemory.set_seeds(seeds)

    config_controller = dict(config_manager("config_controllers")[controller_short_name])
    config_optimizer = dict(config_manager("config_optimizers")[optimizer_short_name])
    config_optimizer.update({"seed": int(seeds[1])})
    config_environment = dict(config_manager("config_environments")[environment_name])
    config_environment.update({"seed": int(seeds[0])})
    all_rewards = []

    ##### ----------------------------------------------- #####
    ##### ----------------- ENVIRONMENT ----------------- #####
    ##### --- Instantiate environment and call reset ---- #####
    if config_manager("config")["render_for_humans"]:
        render_mode = "human"
    elif config_manager("config")["save_plots_to_file"]:
        render_mode = "rgb_array"
    else:
        render_mode = None

    import matplotlib

    matplotlib.use("Agg")

    env: EnvironmentBatched = gym.make(environment_name, **config_environment,
                                       computation_lib=TensorFlowLibrary,
                                       render_mode=render_mode)
    CurrentRunMemory.current_environment = env
    obs, obs_info = env.reset(seed=config_environment["seed"])
    assert len(
        env.action_space.shape) == 1, f"Action space needs to be a flat vector, is Box with shape {env.action_space.shape}"

    # td3 variables
    max_action = env.action_space.high
    min_action = env.action_space.low
    max_obs = env.observation_space.high
    if two_neuron_encoding:
        input_dims = (env.observation_space.shape[0]*2)
    else:
        input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]

    memory = ReplayBuffer(buffer_size, input_dims, n_actions)

    for i in range(len(max_obs)):
        if max_obs[i] == np.inf:
            max_obs[i] = 1

    actor_architecture = [input_dims, layer1_size, layer2_size, n_actions]

    #update_network_parameters(actor, target_actor, critic_1, target_critic_1, critic_2,
    #                          target_critic_2, tau=1)

    model_rewards = []

    for j in range(len(models)):
        # networks
        actor = TD3ActorDSNN(actor_architecture, 0, alpha, beta, weight_scale, 1, threshold,
                             sim_time, actor_learning_rate, name='actor', device=device)
        m = models[0]
        print(models[0])
        w = torch.load('snn_results/{}'.format(models[0]), map_location=torch.device(device))
        actor.load_state_dict(w)
        reward_history = []
        model_rewards.append([])
        # Loop through episodes
        for i in trange(num_experiments):

            # Generate new seeds for environment and controller
            seeds = seed_sequences[i].generate_state(3)
            SeedMemory.set_seeds(seeds)

            config_controller = dict(config_manager("config_controllers")[controller_short_name])
            config_optimizer = dict(config_manager("config_optimizers")[optimizer_short_name])
            config_optimizer.update({"seed": int(seeds[1])})
            config_environment = dict(config_manager("config_environments")[environment_name])
            config_environment.update({"seed": int(seeds[0])})
            all_rewards = []

            episode_counter += 1
            print(episode_counter)
            obs, obs_info = env.reset(seed=config_environment["seed"])
            max_obs = update_max_obs(obs, max_obs)

            if two_neuron_encoding:
                obs = transform_state(obs)

            score = 0

            ##### ---------------------------------------------- #####
            ##### ----------------- CONTROLLER ----------------- #####
            controller_module = import_module(f"Control_Toolkit.Controllers.{controller_name}")
            controller: template_controller = getattr(controller_module, controller_name)(
                dt=env.dt,
                environment_name=ENV_REGISTRY[environment_name].split(":")[-1],
                control_limits=(env.action_space.low, env.action_space.high),
                initial_environment_attributes=env.environment_attributes)
            controller.configure(optimizer_name=optimizer_short_name, predictor_specification=config_controller["predictor_specification"])

            ##### ----------------------------------------------------- #####
            ##### ----------------- MAIN CONTROL LOOP ----------------- #####
            frames = []
            start_time = time.time()
            num_iterations = config_manager("config")["num_iterations"]
            for step in range(num_iterations):
                #action = controller.step(obs, updated_attributes=env.environment_attributes)
                action = choose_action(actor, obs, max_action, min_action, n_actions, max_obs)
                new_obs, reward, terminated, truncated, info = env.step(action)
                if two_neuron_encoding:
                    new_obs_one_neuron = new_obs
                    new_obs = transform_state(new_obs)
                if spiking:
                    new_obs = new_obs.reshape(input_dims)
                c_fun: CostFunctionWrapper = getattr(controller, "cost_function", None)
                if c_fun is not None:
                    assert isinstance(c_fun, CostFunctionWrapper)
                    # Compute reward from the cost function that the controller optimized
                    reward = -float(c_fun.get_stage_cost(
                        tf.convert_to_tensor(new_obs_one_neuron[np.newaxis, np.newaxis, ...]),  # Add batch / MPC horizon dimensions
                        tf.convert_to_tensor(action[np.newaxis, np.newaxis, ...]),
                        None
                    ))
                    all_rewards.append(reward)
                if config_controller.get("controller_logging", False):
                    controller.logs["realized_cost_logged"].append(np.array([-reward]).copy())
                    env.set_logs(controller.logs)
                if config_manager("config")["render_for_humans"]:
                    env.render()
                elif config_manager("config")["save_plots_to_file"]:
                    frames.append(env.render())

                done = terminated or truncated
                memory.store_transition(obs, action, reward, new_obs, done)
                score += reward

                time.sleep(1e-6)

                obs = new_obs

            # Print compute time statistics
            end_time = time.time()
            control_freq = num_iterations / (end_time - start_time)

            reward_history.append(score)
            model_rewards.append(score)
            avg_score = np.mean(reward_history[-100:])
            smoothed_scores.append(avg_score)

            #if avg_score > best_average:
            #    best_average = avg_score
            #    best_average_after = episode_counter

            print('Episode: ', episode_counter, 'training steps: ', learn_step_counter,
                  'score: %.1f' % score, 'Average Score: %.1f' % avg_score)

        print("Model: {}, Average Score: {}".format(j, np.mean(model_rewards[j])))

    # Close the env
    env.close()
    print('Best 100 episode average: ', best_average, ' reached at episode ', best_average_after,
          '.')
    return model_rewards

def prepare_and_run():
    import ruamel.yaml
    
    # Create a config manager which looks for '.yml' files within the list of folders specified.
    # Rationale: We want GUILD AI to be able to update values in configs that we include in this list.
    # We might intentionally want to exclude the path to a folder which does contain configs but should not be overwritten by GUILD. 
    config_manager = ConfigManager(".", "Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments")
    
    # Scan for any custom parameters that should overwrite the toolkits' config files:
    submodule_configs = ConfigManager("Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments").loaders
    for base_name, loader in submodule_configs.items():
        if base_name in config_manager("config").get("custom_config_overwrites", {}):
            data: ruamel.yaml.comments.CommentedMap = loader.load()
            update_dict = config_manager("config")["custom_config_overwrites"][base_name]
            nested_assignment_to_ordereddict(data, update_dict)
            loader.overwrite_config(data)
    
    # Retrieve required parameters from config:
    CurrentRunMemory.current_controller_name = config_manager("config")["controller_name"]
    CurrentRunMemory.current_environment_name = config_manager("config")["environment_name"]
    
    smoothed_scores = run_data_generator(controller_name=CurrentRunMemory.current_controller_name,
                                         environment_name=CurrentRunMemory.current_environment_name,
                                         config_manager=config_manager)
    return smoothed_scores

if __name__ == "__main__":
    smoothed_scores = prepare_and_run()