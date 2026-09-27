from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class QAMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float, # inverse temperature
        
        sigma: Callable[[torch.Tensor], torch.Tensor], # (B,horizon,) -> (B,horizon,), horizon != 1 if action chunking is used
        clip_adj: bool = False, # False keeps straight-through gradients through action clipping.
        use_bc_last_step_during_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.actor =  make_actor(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha
        
        self.sigma = sigma
        self.clip_adj = clip_adj
        self.use_bc_last_step_during_training = use_bc_last_step_during_training
        
    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None] # shape: (1, ob_dim)
        
        # sample z ~ N(0,I_d) where d = self.action_dim
        z = torch.randn((1, self.action_dim), device=ptu.device)
        
        # Sample with ODE flow
        with torch.no_grad():
            for i in range(self.flow_steps):
                # the times should be of shape (B,1), here B=1
                times = torch.full((observation.shape[0], 1), i/self.flow_steps, device=ptu.device)
                v = self.actor(obs=observation, acs=z, times=times)
                z = z + v * 1/self.flow_steps
        
            action = torch.clamp(z, -1, 1)
        return ptu.to_numpy(action)[0]
    
    @torch.compile
    def update_q(self, observations, actions, rewards, next_observations, dones):
        
        value = self.critic(observations, actions)
        
        with torch.no_grad():
            
            # step 1: sample the a' for \pi(a' | s') using the self.actor)
            next_actions = torch.randn_like(actions, device=ptu.device)
            for i in range(self.flow_steps):
                times = torch.full((observations.shape[0], 1), i/self.flow_steps, device=ptu.device)
                v = self.actor(obs=next_observations, acs=next_actions, times=times)
                next_actions = next_actions + v * 1/self.flow_steps
            next_actions = torch.clamp(next_actions, -1, 1)
            
            # step 2: get the target value using the self.target_critic
            # here we refer to the official impl: https://github.com/ColinQiyangLi/qam/blob/main/agents/ifql.py
            # we use pessimistic target value, i.e., mean(Q) - 0.5 * std(Q) (min Q is also fine)
            # self.target_critic has shape (n_ensembles, B)
            next_q_values = self.target_critic(next_observations, next_actions)
            next_q_values = next_q_values.mean(dim=0) - 0.5 * next_q_values.std(dim=0)
            target_value = rewards + self.discount * (1 - dones) * next_q_values
            
        loss = F.mse_loss(value, target_value)
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        return {
            "q_loss": loss,
            "q_mean": value.mean(),
            "q_max": value.max(),
            "q_min": value.min(),
        }
            
    
    @torch.compile
    def update_bc_actor(self, observations: torch.Tensor, actions: torch.Tensor):
        
        z = torch.randn_like(actions, device=ptu.device)
        target_v = actions - z
        
        times = torch.randint(0, self.flow_steps, (observations.shape[0], 1), device=ptu.device) / self.flow_steps
        intermediate_actions = z + (actions-z) * times
        v = self.bc_actor(obs=observations, acs=intermediate_actions, times=times)
        
        loss = F.mse_loss(v, target_v)
            
        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()
        
        return {"bc_v_matching_loss": loss}
        
    @torch.compile
    def update_actor(self, observations: torch.Tensor, actions: torch.Tensor):
        
        # step 0: sample actions from self.actor **and** store the intermediate actions for each flow step
        # we should sample with SDE rather than ODE
        # note that we don't store the computation graph for the whole flow
        # this eliminates the memory issue
        with torch.no_grad():
            policy_actions = torch.randn_like(actions, device=ptu.device)
            intermediate_actions = [] # t=0, ... , t=(self.flow_steps-1)/self.flow_steps
            for i in range(self.flow_steps):
                times = torch.full((observations.shape[0], 1), i/self.flow_steps, device=ptu.device)
                intermediate_actions.append(policy_actions)
                if self.use_bc_last_step_during_training and i == self.flow_steps - 1:
                    # Finish with a deterministic BC-policy ODE step for stability
                    v = self.bc_actor(obs=observations, acs=policy_actions, times=times)
                    policy_actions = policy_actions + v / self.flow_steps
                else:
                    v = self.actor(obs=observations, acs=policy_actions, times=times)
                    policy_actions = (
                        policy_actions
                        + ((1 + 0.5 * self.sigma(times)**2 * times / (1 - times)) * v
                        - 0.5 * self.sigma(times)**2 / (1 - times) * policy_actions) * 1/self.flow_steps
                        + ((1/self.flow_steps) ** 0.5) * self.sigma(times) * torch.randn_like(policy_actions, device=ptu.device)
                    )
        
        # step 1: compute g_1
        policy_actions = policy_actions.detach().requires_grad_(True)
        clipped_actions = torch.clamp(policy_actions, -1, 1)        
        if self.clip_adj:
            critic_actions = clipped_actions
        else:
            # Preserve clipped critic inputs while using an identity clipping Jacobian.
            # this is Straight-through gradient estimator for the clipping operation
            critic_actions = clipped_actions.detach() + (policy_actions - policy_actions.detach())
        values = torch.mean(self.critic(observations, critic_actions), dim=0)
        values = -self.alpha * values
        g1 = torch.autograd.grad(outputs=values.sum(), inputs=policy_actions)[0]
        
        # step 2: compute g_t using backward SDE using VJP for all t's
        # note for vjp: J_f(x)^T v = \nabla_x v^T f(x)
        def vjp(g, t, x):
            x = x.detach().requires_grad_(True)

            sigma_t = self.sigma(t)

            f_outputs = (
                (1 + 0.5 * sigma_t**2 * t / (1 - t))
                * self.bc_actor(obs=observations, acs=x, times=t)
                - 0.5 * sigma_t**2 / (1 - t) * x
            )

            return torch.autograd.grad(
                outputs=f_outputs,
                inputs=x,
                grad_outputs=g,
            )[0]
        
        # ======

        adjoints, flow_times = [], []
        g = g1.clone()
        for i in reversed(range(self.flow_steps)):
            t = torch.full((observations.shape[0], 1), i/self.flow_steps, device=ptu.device)
            g = g + 1/self.flow_steps * vjp(g, t, intermediate_actions[i])
            adjoints.append(g)
            flow_times.append(t)

        # Batch the independent matching losses; retain the original sum over time.
        xs = torch.cat(intermediate_actions, dim=0)
        ts = torch.cat(flow_times[::-1], dim=0)
        gs = torch.cat(adjoints[::-1], dim=0)
        obs = observations.repeat(self.flow_steps, 1)
        with torch.no_grad():
            base_v = self.bc_actor(obs=obs, acs=xs, times=ts)
        u = self.actor(obs=obs, acs=xs, times=ts) - base_v
        sigmas = self.sigma(ts)
        coefficient = (1 + 0.5 * sigmas**2 * ts / (1 - ts)) / sigmas
        loss = 0.5 * self.flow_steps * F.mse_loss(coefficient * u, -sigmas * gs)
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return {"lean_adjoint_matching_loss": loss}
        
    
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
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    @torch.no_grad()
    def update_target_critic(self) -> None:
        torch._foreach_lerp_(
            list(self.target_critic.parameters()),
            list(self.critic.parameters()),
            self.target_update_rate,
        )
