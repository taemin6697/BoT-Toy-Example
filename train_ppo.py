import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from torch import nn
from model import BoTModel

#이 PPO 코드는 GPT로 만들어져 정확한 구현이 아닙니다 논문의 구현을 직접 참고하십시오.

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = BoTModel(d_model=512, head=2, layer_num=6, embodiment_num=6, mode='Hard').to(self.device)

    def reset(self):
        body_sensor = np.random.randn(1).astype(np.float32)
        arm_sensor = np.random.randint(0, 4, size=(5,)).astype(np.int64)
        self.state = (body_sensor, arm_sensor)
        return np.concatenate((body_sensor, arm_sensor.astype(np.float32)))

    def step(self, action):
        body_sensor, arm_sensor = self.state
        body_sensor = torch.tensor(body_sensor).unsqueeze(0).to(self.device)
        arm_sensor = torch.tensor(arm_sensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, _ = self.model(body_sensor, arm_sensor)

        next_body_sensor = outputs[0].cpu().numpy().flatten()
        next_arm_sensors = np.array([out.argmax().item() for out in outputs[1:]])

        self.state = (next_body_sensor, next_arm_sensors)
        observation = np.concatenate((next_body_sensor, next_arm_sensors.astype(np.float32)))

        reward = self._calculate_reward(next_body_sensor, action)
        done = False
        info = {}

        return observation, reward, done, info

    def _calculate_reward(self, observation, action):
        return float(action == 1)


class BoTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(BoTFeatureExtractor, self).__init__(observation_space, features_dim)
        self.bot_model = BoTModel(d_model=512, head=2, layer_num=6, embodiment_num=6, mode='Hard').to(get_device("auto"))

        self.feature_projection = nn.Linear(11, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        body_sensor = observations[:, :1]
        arm_sensor = observations[:, 1:].long()
        body_sensor = body_sensor.to(get_device("auto"))
        arm_sensor = arm_sensor.to(get_device("auto"))

        with torch.no_grad():
            outputs, _ = self.bot_model(body_sensor, arm_sensor)

        body_feature = outputs[0]
        arm_features = torch.cat(outputs[1:], dim=1)

        combined_features = torch.cat([body_feature, arm_features], dim=1)

        # 512차원으로 투영
        projected_features = self.feature_projection(combined_features)
        return projected_features



class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs,
                                                     features_extractor_class=BoTFeatureExtractor,
                                                     features_extractor_kwargs=dict(features_dim=512))


env = DummyVecEnv([lambda: CustomEnv()])

model = PPO(CustomActorCriticPolicy, env, verbose=1, learning_rate=0.001)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./ppo_checkpoints/', name_prefix='ppo_bot_model')

model.learn(total_timesteps=10000, callback=checkpoint_callback)
model.save("ppo_botmodel")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
