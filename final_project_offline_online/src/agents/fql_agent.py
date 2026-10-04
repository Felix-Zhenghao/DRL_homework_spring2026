from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        z = torch.randn((1, self.action_dim), device=ptu.device)
        denoise_direction = self.onestep_actor(obs=observation, acs=z) # omit `times` to use the default value of 0
        action = z + denoise_direction
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        action = noise
        for i in range(self.flow_steps):
            times = torch.full((*noise.shape[:-1], 1), i / self.flow_steps, device=ptu.device) # (B, 1)
            denoise_direction = self.bc_actor(obs=observation, acs=action, times=times)
            action = action + (1/self.flow_steps) * denoise_direction

        action = torch.clamp(action, -1, 1)
        return action

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
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        
        with torch.no_grad():
            noise = torch.randn_like(actions)
            next_actions = noise + self.onestep_actor(obs=next_observations, acs=noise) # omit `times` to use the default value of 0
            next_actions = torch.clamp(next_actions, -1, 1)
            
            next_q_values = self.target_critic(next_observations, next_actions) # (n_ensembles, B)
            next_q_values = next_q_values.mean(dim=0) # (B,)
            target_q_values = rewards + self.discount * (1 - dones) * next_q_values

        q = self.critic(observations, actions) # (n_ensembles, B)
        loss = F.mse_loss(q, target_q_values.unsqueeze(0).expand_as(q))

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
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        noises = torch.randn_like(actions)
        
        sampled_times = torch.randint(0, self.flow_steps, size=(actions.shape[0], 1), device=ptu.device) # (B, 1)
        sampled_times = sampled_times.float() / self.flow_steps # (B, 1)
        
        intermediate_actions = noises + sampled_times * (actions-noises)
        denoise_directions = self.bc_actor(obs=observations, acs=intermediate_actions, times=sampled_times)
        
        target_directions = actions - noises
        loss = F.mse_loss(denoise_directions, target_directions)

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        noise = torch.randn_like(actions)
        with torch.no_grad():
            target_actions = self.get_bc_action(observations, noise)
        predicted_actions = noise + self.onestep_actor(obs=observations, acs=noise) # omit `times` to use the default value of 0
        distill_loss = self.alpha * F.mse_loss(predicted_actions, target_actions)

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        onestep_actions = torch.clamp(predicted_actions, -1, 1)
        q_loss = -self.critic(observations, onestep_actions).mean()

        # Total loss.
        loss = distill_loss + q_loss

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    @torch.no_grad
    @torch.compile
    def update_target_critic(self) -> None:
        torch._foreach_lerp_(
            list(self.target_critic.parameters()),
            list(self.critic.parameters()),
            self.target_update_rate,
        )