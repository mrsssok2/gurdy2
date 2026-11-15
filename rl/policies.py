"""Asymmetric actor-critic for SB3 PPO (step 2.7).

Observation: Dict {"policy": 195 actor inputs (proprioception), "privileged": 17 critic-only values}.
* Actor:  obs["policy"] -> MLP -> action mean   (deployable: needs only what the robot measures)
* Critic: concat(obs["policy"], obs["privileged"]) -> MLP -> value   (training only)

    PPO(AsymmetricActorCriticPolicy, env, policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))

transfer_from_flat() copies a flat (Box-obs) MlpPolicy: the actor exactly; the critic with zero
weights on the new privileged inputs, so it starts out predicting exactly what the flat critic did.
"""
from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SelectKeys(BaseFeaturesExtractor):
    """Concatenate selected Dict-observation keys (already flat vectors)."""

    def __init__(self, observation_space: gym.spaces.Dict, keys=("policy",)):
        dim = sum(int(observation_space[k].shape[0]) for k in keys)
        super().__init__(observation_space, features_dim=dim)
        self.keys = tuple(keys)

    def forward(self, obs):
        return torch.cat([obs[k] for k in self.keys], dim=1)


def _mlp(in_dim, sizes, act):
    layers, d = [], in_dim
    for h in sizes:
        layers += [nn.Linear(d, h), act()]
        d = h
    return nn.Sequential(*layers), d


class AsymmetricMlpExtractor(nn.Module):
    """Like SB3's MlpExtractor (same layer names), but actor and critic have different inputs."""

    def __init__(self, pi_dim, vf_dim, net_arch: dict, activation_fn):
        super().__init__()
        self.policy_net, self.latent_dim_pi = _mlp(pi_dim, net_arch["pi"], activation_fn)
        self.value_net, self.latent_dim_vf = _mlp(vf_dim, net_arch["vf"], activation_fn)

    def forward(self, features):
        pi, vf = features
        return self.forward_actor(pi), self.forward_critic(vf)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = SelectKeys
        kwargs["features_extractor_kwargs"] = {"keys": ("policy",)}
        self._n_extractors = 0
        super().__init__(*args, **kwargs)

    def make_features_extractor(self):
        # ActorCriticPolicy calls this twice when not sharing: 1st -> actor, 2nd -> critic.
        self._n_extractors += 1
        keys = ("policy",) if self._n_extractors == 1 else ("policy", "privileged")
        return SelectKeys(self.observation_space, keys)

    def _build_mlp_extractor(self):
        arch = self.net_arch if isinstance(self.net_arch, dict) else {"pi": self.net_arch, "vf": self.net_arch}
        self.mlp_extractor = AsymmetricMlpExtractor(self.pi_features_extractor.features_dim,
                                                    self.vf_features_extractor.features_dim,
                                                    arch, self.activation_fn)


@torch.no_grad()
def transfer_from_flat(asym: AsymmetricActorCriticPolicy, flat: ActorCriticPolicy) -> None:
    """Copy actor exactly; copy critic with zero weights on the extra (privileged) inputs."""
    asym.mlp_extractor.policy_net.load_state_dict(flat.mlp_extractor.policy_net.state_dict())
    asym.action_net.load_state_dict(flat.action_net.state_dict())
    asym.log_std.copy_(flat.log_std)
    fs, a_layers = flat.mlp_extractor.value_net, asym.mlp_extractor.value_net
    for i, (lf, la) in enumerate(zip(fs, a_layers)):
        if isinstance(lf, nn.Linear):
            if i == 0:
                la.weight.zero_()
                la.weight[:, :lf.in_features] = lf.weight
                la.bias.copy_(lf.bias)
            else:
                la.load_state_dict(lf.state_dict())
    asym.value_net.load_state_dict(flat.value_net.state_dict())
