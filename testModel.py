import gym
import rsoccer_gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from algorithms.GaussianPolicy import GaussianMLP
from algorithms.Deterministic import DeterministicMLP
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer

import torch
import torch.nn as nn


# Wrap the environment
env = gym.make('SSLGoToBall-v0')
original_env = env
env = wrap_env(env)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
# instantiate the model (assumes there is a wrapped environment: env)
policy = GaussianMLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum")

print(next(policy.parameters()).is_cuda)

critic1 = DeterministicMLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=device,
             clip_actions=False)
critic2 = DeterministicMLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=device,
             clip_actions=False)
critict1 = DeterministicMLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=device,
             clip_actions=False)
critict2 = DeterministicMLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=device,
             clip_actions=False)
print(next(critic1.parameters()).is_cuda)
print(next(critic2.parameters()).is_cuda)
print(next(critict1.parameters()).is_cuda)
print(next(critict2.parameters()).is_cuda)
memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device)

# instantiate the agent's models
models = {}
models["policy"] = policy
models["critic_1"] = critic1
models["critic_2"] = critic2
models["target_critic_1"] = critict1
models["target_critic_2"] = critict2

for model in models.values():
    for param in model.parameters():
        param.requires_grad = True
# adjust some configuration if necessary
cfg_agent = SAC_DEFAULT_CONFIG.copy()


cfg_agent["batch_size"] = 256
cfg_agent["exploration"]["noise"] = GaussianNoise(mean=0, std=0.2, device="cuda:0")
cfg_agent["headless"] = False
# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = SAC(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

agent.load("")
trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg_agent)

trainer.eval()