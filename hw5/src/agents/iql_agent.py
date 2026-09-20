from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(nn.Module):
    aggregation_functions = {
        "min": torch.min,
        "mean": torch.mean,
        "max": torch.max,
    }
    
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        expectile: float,
        ensemble_aggregation: str,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim) # Q(s,a)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape) # V(s)

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.expectile = expectile
        self.ensemble_aggregation = ensemble_aggregation
        self.aggregation_func_chosen = self.aggregation_functions[ensemble_aggregation]

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.actor(observation).mode  # Take the mean (mode) action
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action[0])

    @staticmethod
    def iql_expectile_loss(
        adv: torch.Tensor, expectile: float,
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        # TODO(student): Implement the expectile loss
        
        # The expectile loss: we are more tolerant of positive advs.
        # In this way, we encourage the V(s) to be the larger quantile of Q(s,a).
        # Philosophically, this means we are optimisitc about ID actions.
        loss_weight = torch.where(adv > 0, expectile, 1 - expectile)
        
        # Input: adv = Q(s,a) - V(s) with shape (batch_size, 1)
        loss = loss_weight * (adv ** 2)
        
        return loss.mean()

    @torch.compile
    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update V(s) with expectile regression
        """
        # TODO(student): Compute the value loss
        v = self.value(observations) # output shape: (batch_size, 1)
        
        # Here we use an ensemble of two Q networks.
        # Therefore, the output shape of self.critic: (2, batch_size)
        # We use the minimum of the two Q networks as the Q to compute advantage.
        loss = self.iql_expectile_loss(
            self.aggregation_func_chosen(self.target_critic(observations, actions), dim=0) - v if self.ensemble_aggregation == "mean" else self.aggregation_func_chosen(self.target_critic(observations, actions), dim=0)[0] - v,
            self.expectile
        )

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        q = self.critic(observations, actions) # output shape: (2, batch_size)
        
        target = rewards + self.discount * dones * self.value(next_observations)
        target = target.detach() # stop gradient for target
        
        # target now has shape (batch_size, 1), change it to (2, batch_size) to match q's shape
        target = target.expand_as(q)
        
        loss = torch.sum(torch.mean(torch.square(
            target - q
        ), dim=1)) # mean over batch, sum over ensemble

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor using advantage-weighted regression
        """
        # TODO(student): Compute the actor loss
        dist = self.actor(observations)
        
        adv = self.aggregation_func_chosen(self.target_critic(observations, actions), dim=0) - self.value(observations) if self.ensemble_aggregation == "mean" else self.aggregation_func_chosen(self.target_critic(observations, actions), dim=0)[0] - self.value(observations)
        
        weights = torch.clamp(torch.exp(adv * self.alpha).detach(), max=100.0)
        loss = torch.mean(
            -dist.log_prob(actions) * weights
        )

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": loss,
            "mse": torch.mean((dist.mode - actions) ** 2),
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_v(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        
        # target_update_rate = 0.005
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)
