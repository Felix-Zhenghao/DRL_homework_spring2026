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
        observation = ptu.from_numpy(np.asarray(observation))[None] # shape: (1, ob_dim)
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        
        # sample z ~ N(0,I_d) where d = self.action_dim
        z = torch.randn((1, self.action_dim), device=ptu.device)
        v = self.onestep_actor(obs=observation, acs=z) # times is the will be set to default: 0
        action = v + z
        
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        
        # Implementation protocal of flow BC policy training:
        # 1. Input timestep: i = 0/self.flow_steps, 1/self.flow_steps, ..., (self.flow_steps-1)/self.flow_steps
        # 2. Output target of self.bc_actor: v = a-z
        action = noise.clone()  # Preserve the original noise for the one-step actor.
        for i in range(self.flow_steps):
            times = torch.full((*noise.shape[:-1], 1), fill_value=i/self.flow_steps, device=ptu.device)
            action += 1/self.flow_steps * self.bc_actor(obs=observation, acs=action, times=times)
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
        q = self.critic(observations, actions) # (n_ensembles, B)
        with torch.no_grad():
            noises = torch.randn_like(actions, device=ptu.device)
            vs_onestep_actor = self.onestep_actor(obs=next_observations, acs=noises)
            next_actions = torch.clamp(vs_onestep_actor + noises, -1, 1)
            target_q = self.target_critic(next_observations, next_actions).mean(dim=0) # (B,)
            target_q = target_q.unsqueeze(0).expand_as(q) # (n_ensembles, B)

        loss = F.mse_loss(q, rewards + self.discount * (1 - dones) * target_q)

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
        
        noises = torch.randn_like(actions, device=ptu.device)
        times = torch.randint(
            low=0, high=self.flow_steps,
            size=actions.shape[:-1] + (1,),
            device=ptu.device, dtype=torch.float32
        ) # shape: (batch_size, 1)
        times = times / self.flow_steps # shape: (batch_size, 1)
        intermediate_actions = noises + times*(actions-noises)
        
        vs = self.bc_actor(obs=observations, acs=intermediate_actions, times=times)
        target = actions-noises
        loss = F.mse_loss(vs, target)

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
        noises = torch.randn_like(actions, device=ptu.device)
        times = torch.randint(
            low=0, high=self.flow_steps-1,
            size=actions.shape[:-1] + (1,),
            device=ptu.device, dtype=torch.float32
        ) # shape: (batch_size, 1)
        times = times / (self.flow_steps-1)
        bc_actions = self.get_bc_action(observations, noises).detach() # (B, ac_dim)
        one_step_actions = self.onestep_actor(obs=observations, acs=noises) + noises # (B, ac_dim)
        
        distill_loss = F.mse_loss(one_step_actions, bc_actions) * self.alpha

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        critic_value = self.critic(observations, torch.clamp(one_step_actions, -1, 1)) # (n_ensembles, B)
        critic_value = critic_value.mean(dim=0) # (B,)
        q_loss = -critic_value.mean()
        
        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = F.mse_loss(one_step_actions, actions)

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
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

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)
