import torch
import torch.nn.functional as F
import numpy as np


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer


class TD3_SMR:

    def __init__(self, max_action: np.ndarray):

        self.train_steps = 0

        self.policy_noise = torch.tensor(args.policy_noise).to(torch.float32).detach().to(args.device)
        self.policy_noise_clip = torch.tensor(args.policy_noise_clip).to(torch.float32).detach().to(args.device)
        self.max_action = torch.from_numpy(max_action).to(torch.float32).detach().to(args.device)
        self.gamma = torch.tensor(args.gamma).to(torch.float32).detach().to(args.device)
        self.tau = torch.tensor(args.tau).to(torch.float32).detach().to(args.device)


    def train(
            self,
            actor: Actor,
            actor_target: Actor,
            critic: Critic,
            critic_target: Critic,
            replay_buffer: ReplayBuffer,
            actor_optimizer: torch.optim.Adam,
            critic_optimizer: torch.optim.Adam
        ):

        self.train_steps += 1


        replays = replay_buffer.sample()

        states = torch.stack([replay.state for replay in replays])
        actions = torch.stack([replay.action for replay in replays])
        rewards = torch.stack([replay.reward for replay in replays])
        next_states = torch.stack([replay.next_state for replay in replays])
        not_dones = torch.stack([replay.not_done for replay in replays])



        for M in range(args.smr_ratio):

            # 計算 target_Q
            with torch.no_grad():

                next_actions = actor_target(next_states)
                noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.policy_noise_clip , self.policy_noise_clip)
                next_actions = (next_actions + noise).clamp(-self.max_action , self.max_action)


                target_Q1 , target_Q2 = critic_target(next_states, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + not_dones * self.gamma * target_Q


            # 計算 Q1 , Q2
            current_Q1 , current_Q2 = critic(states, actions)

            # MSE loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # 反向傳播
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()


            # 訓練 actor 並更新 target network
            if self.train_steps % args.policy_frequency == 0:

                actor_actions = actor(states)

                actor_loss = -critic.forward_Q1(states, actor_actions).mean()

                # 反向傳播
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 更新 target network
                with torch.no_grad():
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)





















