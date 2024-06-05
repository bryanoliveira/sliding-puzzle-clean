# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import datetime
import json
import os
import random
import re
import time
from dataclasses import dataclass
import yaml

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import sliding_puzzles
import wrappers

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanrl"
    """the wandb's project name"""
    wandb_entity: str = "bryanoliveira"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SlidingPuzzle-v0"
    env_configs: str = None
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    """the agent hidden size"""
    hidden_size: int = 512
    """the number of hidden layers"""
    hidden_layers: int = 0

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime: num_envs * num_steps)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime: batch_size // num_minibatches)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    backbone_variant: str = "dinov2_vits14"
    """the backbone variant of the agent"""
    backbone_perc_unfrozen: float = 0
    """the percentage of the backbone to be unfrozen"""

    checkpoint_load_path: str = None
    """the path to the checkpoint to load"""
    checkpoint_param_filter: str = ".*"
    """the filter to load checkpoint parameters"""
    checkpoint_every: int = 10000


def make_env(env_id, idx, capture_video, run_name, env_configs):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_configs)
            env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/videos")
        else:
            env = gym.make(env_id, **env_configs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        img_obs = min(env.observation_space.shape[-1], env.observation_space.shape[0]) in (3, 4)
        if "ALE" in env_id or "NoFrameskip" in env_id:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            if img_obs:
                env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.FrameStack(env, 4)
        elif img_obs:
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = wrappers.ChannelFirstImageWrapper(env)
            env = wrappers.NormalizedImageWrapper(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, hidden_size, hidden_layers, backbone_variant="dinov2_vits14", backbone_perc_unfrozen=0):
        super().__init__()
        # configure backbone
        self.dino = torch.hub.load("facebookresearch/dinov2", backbone_variant)
        layers = list(self.dino.parameters())
        unfrozen_layers = int(len(layers) * backbone_perc_unfrozen)
        print(f"Unfreezing {unfrozen_layers} out of {len(layers)} Dino layers")
        for i, p in enumerate(layers):
            p.requires_grad_(p.requires_grad and (i >= len(layers) - unfrozen_layers))

        # configure transforms
        self.dino_transforms = transforms.Compose([
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2) / 255.0),  # Change(B, H, W, C) to (B, C, H, W)
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ])

        # convert possibly stacked images to a single linear layer
        self.dino_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                self.dino.embed_dim * (
                    envs.single_observation_space.shape[0] 
                    if len(envs.single_observation_space.shape) == 4 
                    else 1
                ),
                hidden_size,
            ),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            *[
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
            ] * hidden_layers,
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            *[
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
            ] * hidden_layers,
            layer_init(nn.Linear(hidden_size, envs.single_action_space.n), std=0.01),
        )

    def encoder(self, x):
        frame_stacked = len(x.shape) == 5
        if frame_stacked:
            batch_size, frame_stack, w, h, c = x.shape
            x = x.view(batch_size * frame_stack, w, h, c)
        else:
            batch_size = x.shape[0]
        x = self.dino_transforms(x)
        x = self.dino(x)
        if frame_stacked:
            x = x.reshape(batch_size, frame_stack, -1)
        x = x.reshape(batch_size, -1)
        x = self.dino_linear(x)
        return x

    def get_value(self, x):
        x = self.encoder(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.encoder(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def save_checkpoint(agent, optimizer, global_step, run_name):
    checkpoint_path = f"runs/{run_name}/checkpoint_{global_step}.pth"
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(agent, optimizer, checkpoint_path, param_filter):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    regex = re.compile(param_filter)
    filtered_state_dict = {k: v for k, v in checkpoint['agent_state_dict'].items() if regex.match(k)}
    agent_state_dict = agent.state_dict()
    agent_state_dict.update(filtered_state_dict)
    agent.load_state_dict(agent_state_dict)
    if len(filtered_state_dict) == len(checkpoint['agent_state_dict']):
        return checkpoint['global_step']
    else:
        print("Loaded params: ", filtered_state_dict.keys())
        return 0


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.exp_name += "_" + args.env_id.replace("/", "").replace("-", "").lower()
    if args.env_configs:
        args.exp_name += "_" + args.env_configs.replace("{", "").replace("}", "").replace(":", "_").replace(",", "_").replace(" ", "").replace('"', "")
    run_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{args.exp_name}_{args.seed}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group=args.exp_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    with open(f"runs/{run_name}/config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"device: {device}")

    # env setup
    env_configs = json.loads(args.env_configs) if args.env_configs else {}
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, env_configs) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args.hidden_size, args.hidden_layers, args.backbone_variant, args.backbone_perc_unfrozen).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(agent)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)


    if args.checkpoint_load_path:
        global_step = load_checkpoint(agent, optimizer, args.checkpoint_load_path, args.checkpoint_param_filter)
    else:
        save_checkpoint(agent, optimizer, global_step, run_name)

    pbar = tqdm(range(1, args.num_iterations + 1), desc="iteration")
    for iteration in pbar:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        successes = []
        returns = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        successes.append(info.get("is_success", 0))
                        returns.append(info["episode"]["r"])

        if returns:
            successes = np.mean(successes)
            returns = np.mean(returns)
            writer.add_scalar("charts/mean_episodic_success", successes, global_step)
            writer.add_scalar("charts/mean_episodic_return", returns, global_step)
            pbar.set_postfix_str(f"step={global_step}, return={returns:.2f}, success={successes:.2f}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if iteration % args.checkpoint_every == 0:
            save_checkpoint(agent, optimizer, global_step, run_name)

    envs.close()
    writer.close()