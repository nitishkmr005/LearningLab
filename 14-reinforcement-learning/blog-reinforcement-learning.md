# Reinforcement Learning: From MDPs to RLHF and GRPO

> A comprehensive guide to RL fundamentals, deep RL algorithms, and modern LLM alignment techniques — for ML Engineers and AI Engineers.

---

## Table of Contents

1. [RL Fundamentals](#1-rl-fundamentals)
2. [Value Functions and Bellman Equations](#2-value-functions-and-bellman-equations)
3. [Dynamic Programming](#3-dynamic-programming)
4. [Monte Carlo Methods](#4-monte-carlo-methods)
5. [Temporal-Difference Learning](#5-temporal-difference-learning)
6. [Deep Q-Networks (DQN)](#6-deep-q-networks-dqn)
7. [Policy Gradient Methods](#7-policy-gradient-methods)
8. [Proximal Policy Optimization (PPO)](#8-proximal-policy-optimization-ppo)
9. [Soft Actor-Critic (SAC)](#9-soft-actor-critic-sac)
10. [Model-Based RL](#10-model-based-rl)
11. [RLHF: Reinforcement Learning from Human Feedback](#11-rlhf-reinforcement-learning-from-human-feedback)
12. [DPO and Variants](#12-dpo-and-variants)
13. [GRPO: Group Relative Policy Optimization](#13-grpo-group-relative-policy-optimization)
14. [RL Libraries and Environments](#14-rl-libraries-and-environments)
15. [References](#15-references)

---

## 1. RL Fundamentals

Reinforcement learning is the study of how an **agent** learns to take **actions** in an **environment** to maximize cumulative **reward** over time.

The core loop:

```
State s_t → Agent selects action a_t → Environment transitions to s_{t+1} → Agent receives reward r_t
```

**Markov Decision Process (MDP)** formalizes this as a tuple (S, A, P, R, γ):
- **S**: state space
- **A**: action space
- **P(s'|s,a)**: transition probability
- **R(s,a)**: reward function
- **γ ∈ [0,1]**: discount factor

The **Markov property** says the next state depends only on the current state and action, not on history — `P(s_{t+1}|s_t, a_t, s_{t-1}, ...) = P(s_{t+1}|s_t, a_t)`.

**Return** is the discounted sum of future rewards:
```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... = Σ_{k=0}^∞ γ^k r_{t+k}
```

γ close to 0 = short-sighted agent (myopic). γ close to 1 = patient agent valuing long-term rewards. γ must be < 1 for infinite horizons to ensure the return is finite.

```python
import numpy as np

def compute_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Compute discounted returns G_t for a trajectory."""
    returns = []
    G = 0.0
    for r in reversed(rewards):                               # work backwards from end of episode
        G = r + gamma * G                                     # G_t = r_t + γ * G_{t+1}
        returns.insert(0, G)                                  # prepend to maintain time order
    return returns

rewards = [0, 0, 1, 0, -1, 0, 1]                            # sparse rewards
returns = compute_returns(rewards, gamma=0.9)
print(f"Rewards:  {rewards}")
print(f"Returns: {[round(g, 3) for g in returns]}")
```

🎯 **Interview prep:** "Why discount future rewards?" — Three reasons: (1) mathematical convergence for infinite horizons, (2) uncertainty about future (a reward now is more certain than a reward later), (3) matches human preference for immediate rewards.

---

## 2. Value Functions and Bellman Equations

**Value functions** estimate expected future return from a state or state-action pair under a policy π.

**State-value function:**
```
V^π(s) = E_π[G_t | s_t = s] = E_π[r_t + γV^π(s_{t+1}) | s_t = s]
```

**Action-value function (Q-function):**
```
Q^π(s,a) = E_π[G_t | s_t = s, a_t = a] = E_π[r_t + γV^π(s_{t+1}) | s_t = s, a_t = a]
```

**Bellman expectation equations** express V and Q recursively:

```
V^π(s) = Σ_a π(a|s) · [R(s,a) + γ · Σ_{s'} P(s'|s,a) V^π(s')]
Q^π(s,a) = R(s,a) + γ · Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a')
```

**Bellman optimality equations** give the optimal value functions V* and Q*:

```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]
Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
```

The optimal policy is greedy with respect to Q*: `π*(s) = argmax_a Q*(s,a)`.

```python
import numpy as np

def bellman_update(V: np.ndarray, transitions: np.ndarray,
                   rewards: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """One step of Bellman update for policy evaluation."""
    # transitions: (n_states, n_states) — transition probabilities under current policy
    # rewards: (n_states,) — expected reward from each state
    n_states = len(V)
    V_new = np.zeros(n_states)
    for s in range(n_states):
        V_new[s] = rewards[s] + gamma * transitions[s] @ V    # r + γ * Σ P(s'|s) * V(s')
    return V_new
```

---

## 3. Dynamic Programming

Dynamic programming (DP) solves MDPs when the model (P and R) is known. The two fundamental algorithms:

**Policy Evaluation:** iteratively apply the Bellman expectation equation until V^π converges. Solves the system of linear equations `V^π = R + γPV^π`.

**Policy Iteration:** alternate between policy evaluation and policy improvement (act greedily with respect to current V^π). Guaranteed to converge to the optimal policy in finite steps for finite MDPs.

**Value Iteration:** directly apply the Bellman optimality equation. Converges to V* without explicit policy representation:

```python
def value_iteration(n_states: int, n_actions: int,
                    P: np.ndarray, R: np.ndarray,
                    gamma: float = 0.9, tol: float = 1e-6) -> tuple:
    """Value iteration: find optimal value function V* and policy π*."""
    V = np.zeros(n_states)                                    # initialize V
    while True:
        V_new = np.zeros(n_states)
        for s in range(n_states):
            # Q(s,a) for all actions
            Q_sa = [R[s,a] + gamma * P[s,a,:] @ V for a in range(n_actions)]
            V_new[s] = max(Q_sa)                              # V*(s) = max_a Q*(s,a)
        delta = np.max(np.abs(V_new - V))                     # convergence check
        V = V_new
        if delta < tol:
            break
    # Extract greedy policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        Q_sa = [R[s,a] + gamma * P[s,a,:] @ V for a in range(n_actions)]
        policy[s] = np.argmax(Q_sa)
    return V, policy
```

DP is exact but requires knowing P and R — impossible for most real-world problems (how do you know the exact weather transition probabilities?). Model-free methods (TD, MC) learn from interaction instead.

---

## 4. Monte Carlo Methods

Monte Carlo (MC) methods learn from **complete episodes** of experience. No model required — just sample trajectories.

**MC prediction:** estimate V^π by averaging the returns observed after each visit to state s:

```
V(s) ← V(s) + α(G_t - V(s))
```

**MC control (on-policy):** use ε-greedy policy to explore; improve policy after each episode. ε-greedy: with probability ε, take a random action; otherwise, act greedily.

MC has high variance (returns depend on many future random events) but zero bias (the expected return is exactly V^π).

```python
import numpy as np
from collections import defaultdict

def mc_prediction(episodes: list[list[tuple]], gamma: float = 0.99) -> dict:
    """First-visit MC prediction: estimate V(s) from a list of episodes."""
    # episodes: list of [(state, action, reward), ...]
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = {}

    for episode in episodes:
        states = [t[0] for t in episode]
        rewards = [t[2] for t in episode]
        G = compute_returns(rewards, gamma)                   # discounted returns

        visited = set()
        for t, (state, _, _) in enumerate(episode):
            if state not in visited:                          # first-visit only
                visited.add(state)
                returns_sum[state] += G[t]                   # accumulate return
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]  # running average

    return V
```

---

## 5. Temporal-Difference Learning

TD learning combines ideas from DP (bootstrap from estimates) and MC (learn from experience). Unlike MC, TD updates after **every step** using a one-step return estimate:

```
TD(0): V(s_t) ← V(s_t) + α[r_t + γV(s_{t+1}) - V(s_t)]
```

The term `r_t + γV(s_{t+1}) - V(s_t)` is the **TD error** — how much better (or worse) things turned out than expected.

**SARSA (on-policy):** update Q(s,a) using the action actually taken in the next state:
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

**Q-Learning (off-policy):** update Q(s,a) using the greedy action in the next state — learns the optimal Q regardless of the exploration policy:
```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
```

```python
import numpy as np

def q_learning(n_states: int, n_actions: int, episodes: int = 1000,
               alpha: float = 0.1, gamma: float = 0.99,
               epsilon: float = 0.1) -> np.ndarray:
    """Tabular Q-learning on a simple grid world."""
    Q = np.zeros((n_states, n_actions))                       # initialize Q table

    def step(state: int, action: int) -> tuple[int, float, bool]:
        next_state = (state + action) % n_states              # mock transition
        reward = 1.0 if next_state == n_states - 1 else 0.0  # reward at goal
        done = next_state == n_states - 1
        return next_state, reward, done

    for episode in range(episodes):
        state = 0                                             # start state
        for _ in range(100):                                  # max steps per episode
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)         # explore
            else:
                action = np.argmax(Q[state])                  # exploit

            next_state, reward, done = step(state, action)
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error              # Q update

            state = next_state
            if done:
                break
    return Q
```

🎯 **Interview prep:** "What's the difference between SARSA and Q-learning?" — SARSA is on-policy: it updates Q using the actual next action taken (ε-greedy). Q-learning is off-policy: it updates Q using the greedy next action (max Q), regardless of what was actually done. Q-learning learns the optimal policy even while exploring; SARSA learns the ε-greedy policy.

---

## 6. Deep Q-Networks (DQN)

Tabular Q-learning fails for large or continuous state spaces (Atari has 128×128×3 pixel states). **DQN** ([Mnih et al., 2013](https://arxiv.org/abs/1312.5602)) approximates Q(s,a) with a neural network.

Two critical stabilization tricks:
1. **Experience replay:** store transitions (s, a, r, s') in a replay buffer; sample random mini-batches. Breaks temporal correlations that destabilize training.
2. **Target network:** maintain a separate "target" network θ⁻ for computing TD targets. Update it slowly (copy θ→θ⁻ every K steps). Prevents the moving-target problem where the Q network chases its own unstable predictions.

```python
import torch
import torch.nn as nn
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)                         # one Q value per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)                  # circular buffer

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)         # random mini-batch

    def __len__(self):
        return len(self.buffer)

def dqn_update(q_net: DQN, target_net: DQN, buffer: ReplayBuffer,
               optimizer, gamma: float = 0.99, batch_size: int = 32):
    if len(buffer) < batch_size:
        return
    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]           # greedy target
        td_targets = rewards + gamma * next_q * (1 - dones)  # Bellman target

    loss = nn.MSELoss()(q_values, td_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Improvements to DQN:**
- **Double DQN** ([Van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)): use the online network to select the action, the target network to evaluate it — removes overestimation bias
- **Dueling DQN:** separate value stream V(s) and advantage stream A(s,a) — learns state values faster from states where all actions are equivalent
- **Prioritized Experience Replay:** sample transitions proportional to their TD error magnitude — spend more time on surprising transitions

---

## 7. Policy Gradient Methods

Value-based methods (DQN) are hard to apply to continuous action spaces (you can't take argmax over a continuous action). Policy gradient methods directly parameterize the policy `π_θ(a|s)` and optimize it with gradient ascent.

**REINFORCE** ([Williams, 1992](https://link.springer.com/article/10.1007/BF00992696)): the policy gradient theorem gives:

```
∇_θ J(θ) = E_π[G_t · ∇_θ log π_θ(a_t|s_t)]
```

Intuitively: increase the probability of actions that led to high returns, decrease it for low returns.

**Baseline subtraction:** subtract a baseline b(s) (usually V(s)) from the return to reduce variance without introducing bias:

```
∇_θ J(θ) = E_π[(G_t - b(s_t)) · ∇_θ log π_θ(a_t|s_t)]
```

The **advantage function** A(s,a) = Q(s,a) - V(s) measures how much better action a is than average — the ideal baseline.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)                 # action probabilities

def reinforce_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """REINFORCE policy gradient loss."""
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize returns
    return -torch.sum(log_probs * returns)                    # maximize expected return
```

**Actor-critic** methods maintain both a policy (actor) and a value function (critic). The critic estimates the baseline; the actor updates using the advantage.

---

## 8. Proximal Policy Optimization (PPO)

**PPO** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) is the dominant deep RL algorithm for continuous control and LLM alignment. It stabilizes policy gradient training by constraining how much the policy can change per update.

The **clipped surrogate objective:**

```
L_CLIP(θ) = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

where `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` is the probability ratio. The clip prevents the policy from changing too much in a single update — if r > 1+ε (positive advantage action is being made more likely), the gradient is cut off.

**Generalized Advantage Estimation (GAE)** ([Schulman et al., 2016](https://arxiv.org/abs/1506.02438)) provides low-variance advantage estimates by blending TD(0) and Monte Carlo advantages:

```
A_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

where `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD error and λ ∈ [0,1] controls the bias-variance trade-off.

```python
def compute_gae(rewards: list[float], values: list[float], dones: list[bool],
                gamma: float = 0.99, lam: float = 0.95) -> list[float]:
    """Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae     # GAE recursion
        advantages.insert(0, gae)
    return advantages

def ppo_clip_loss(log_probs_new: torch.Tensor, log_probs_old: torch.Tensor,
                  advantages: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    """PPO clipped surrogate loss."""
    ratio = torch.exp(log_probs_new - log_probs_old)          # π_θ / π_old
    surr1 = ratio * advantages                                 # standard policy gradient
    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages    # clipped version
    return -torch.mean(torch.min(surr1, surr2))               # pessimistic (min) objective
```

PPO hyperparameters that matter most: clip range ε (typically 0.1–0.2), GAE λ (0.95), learning rate (3e-4 for actor, often lower for critic), entropy coefficient (0.01 to encourage exploration).

---

## 9. Soft Actor-Critic (SAC)

**SAC** ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)) adds an entropy term to the RL objective — the agent is rewarded both for high return and for acting as randomly as possible:

```
J(π) = Σ_t E[r_t + α · H(π(·|s_t))]
```

where H(π) is the entropy of the policy. This **maximum entropy framework** encourages exploration (high entropy) while still achieving good performance. SAC is the standard algorithm for continuous control tasks (robotics, physics simulation).

Key properties:
- Off-policy (sample efficient — can replay old experience)
- Entropy regularization prevents collapse to deterministic policies
- Automatic temperature tuning: learn α such that target entropy is achieved
- Actor parameterized as squashed Gaussian (tanh(N(μ,σ))) for bounded continuous actions

SAC consistently outperforms PPO on continuous control benchmarks like MuJoCo locomotion. PPO is preferred for discrete action spaces and LLM fine-tuning (where the "environment" is the reward model).

---

## 10. Model-Based RL

Model-free RL (DQN, PPO, SAC) learns from millions of environment interactions. **Model-based RL** learns a world model (predicting s' and r given s and a) and plans inside it — dramatically improving sample efficiency.

**Dyna architecture** ([Sutton, 1991](https://dl.acm.org/doi/10.1145/122344.122377)): mix real environment transitions with simulated transitions from the world model. More model rollouts = better sample efficiency at the cost of model errors accumulating.

**MBPO** ([Janner et al., 2019](https://arxiv.org/abs/1906.08253)): short model rollouts (k=1–5 steps) from real starting states, combined with real data in a SAC replay buffer. Short rollouts limit model error accumulation.

**AlphaGo / AlphaZero** ([Silver et al., 2017](https://arxiv.org/abs/1712.01815)): combine a learned policy/value network with MCTS (Monte Carlo Tree Search) planning in the world model. The network provides initial move probabilities and value estimates; MCTS improves them by lookahead search.

---

## 11. RLHF: Reinforcement Learning from Human Feedback

RLHF ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) is the standard pipeline for aligning LLMs to human preferences. Three phases:

**Phase 1 — Supervised Fine-Tuning (SFT):** fine-tune the pre-trained LLM on high-quality human demonstrations of desired behavior.

**Phase 2 — Reward Model (RM) training:** collect human preference data — pairs of model responses (y_w, y_l) to the same prompt, labeled as "preferred" and "rejected." Train a scalar reward model:

```
L_RM = -E[log σ(r_θ(x, y_w) - r_θ(x, y_l))]
```

where σ is the sigmoid function. This is a Bradley-Terry model of human preferences.

**Phase 3 — PPO fine-tuning:** maximize expected reward while staying close to the SFT policy:

```
L_PPO = E[r_θ(x, y)] - β · KL(π_PPO || π_SFT)
```

The KL term prevents reward hacking — without it, the model collapses to generating text that maximizes the reward model but sounds nonsensical.

```python
def rlhf_reward_model_loss(preferred_rewards: torch.Tensor,
                            rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry loss for reward model training."""
    return -torch.mean(torch.log(torch.sigmoid(preferred_rewards - rejected_rewards)))

def ppo_rlhf_objective(rewards: torch.Tensor, kl_divergence: torch.Tensor,
                        beta: float = 0.1) -> torch.Tensor:
    """PPO objective with KL penalty for LLM alignment."""
    return torch.mean(rewards - beta * kl_divergence)         # maximize reward, penalize KL
```

🎯 **Interview prep:** "What is reward hacking in RLHF?" — the policy learns to maximize the proxy reward model score rather than actual human preferences. Example: the model generates very long, repetitive answers that the reward model rates highly but humans find annoying. The KL penalty limits how far the policy drifts from SFT.

---

## 12. DPO and Variants

**DPO (Direct Preference Optimization)** ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) eliminates the reward model training step. It shows that the optimal RLHF policy can be expressed directly in terms of preference data:

```
L_DPO = -E[log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))]
```

Intuitively: increase the log probability of preferred responses relative to the reference policy, decrease it for rejected responses.

```python
def dpo_loss(log_probs_preferred: torch.Tensor,
             log_probs_rejected: torch.Tensor,
             ref_log_probs_preferred: torch.Tensor,
             ref_log_probs_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """DPO loss from preference pairs."""
    # Log ratios relative to reference policy
    preferred_ratio = log_probs_preferred - ref_log_probs_preferred   # log(π/π_ref) for preferred
    rejected_ratio = log_probs_rejected - ref_log_probs_rejected      # log(π/π_ref) for rejected
    logits = beta * (preferred_ratio - rejected_ratio)                # scaled difference
    return -torch.mean(torch.log(torch.sigmoid(logits)))              # binary cross entropy

# Advantages of DPO vs PPO for LLM alignment:
# - No separate reward model (simpler pipeline)
# - No online sampling during training (more stable)
# - Competitive performance on instruction following and helpfulness
# - Disadvantage: no online exploration; can't improve beyond the preference dataset
```

**DPO variants:**
- **RAFT (Reward rAnked Fine-Tuning):** generate multiple responses, rank by a reward model, fine-tune on the top responses (supervised, not RL)
- **KTO** ([Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)): uses individual (good/bad) labels instead of pairs — more data-efficient
- **SimPO:** simplified DPO that uses sequence-level average log probabilities instead of token-level, removing the reference model

---

## 13. GRPO: Group Relative Policy Optimization

**GRPO** ([DeepSeek, 2025](https://arxiv.org/abs/2501.12599)) is the training algorithm behind DeepSeek R1 — a variant of PPO that eliminates the value network (critic), significantly reducing memory cost.

In standard PPO for LLMs:
- Policy network (LLM): ~70B parameters
- Value network (critic): another ~7-70B parameters
- Total: 2–3× the LLM memory budget

GRPO replaces the value network with **group sampling**: for each prompt, generate G outputs (typically G=8), compute the reward for each, and estimate the advantage as the normalized deviation within the group:

```
Advantage_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)
```

No value model needed — the baseline is estimated from the group of outputs itself.

```python
import torch

def grpo_advantage(rewards: torch.Tensor, group_size: int = 8) -> torch.Tensor:
    """Compute GRPO advantages by normalizing within groups."""
    # rewards: (batch_size * group_size,)
    rewards = rewards.view(-1, group_size)                    # (batch_size, group_size)
    mean = rewards.mean(dim=1, keepdim=True)                  # group mean
    std = rewards.std(dim=1, keepdim=True) + 1e-8             # group std
    advantages = (rewards - mean) / std                       # normalized advantage
    return advantages.view(-1)                                 # flatten

def grpo_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
              advantages: torch.Tensor, beta: float = 0.04,
              ref_log_probs: torch.Tensor = None,
              eps: float = 0.2) -> torch.Tensor:
    """GRPO loss: clipped PPO objective + KL penalty against reference policy."""
    ratio = torch.exp(log_probs - old_log_probs)              # π_θ / π_old
    clipped = torch.clamp(ratio, 1-eps, 1+eps)
    policy_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))

    if ref_log_probs is not None:
        kl_penalty = torch.mean(log_probs - ref_log_probs)    # KL from reference
        return policy_loss + beta * kl_penalty
    return policy_loss

# TRL (HuggingFace) provides a GRPOTrainer implementation
# from trl import GRPOTrainer, GRPOConfig
```

**GRPO in practice (DeepSeek R1):** the reward signal is purely outcome-based — correctness on math problems (verified by a symbolic checker) and format compliance (correct use of `<think>` tags). No human preference labels needed. This scales alignment to new domains wherever correctness can be verified automatically.

🎯 **Interview prep:** "What's the advantage of GRPO over PPO for LLM training?" — eliminates the value network (saves ~50% memory), uses group sampling as a stable baseline estimate, and is simpler to implement. The main limitation: requires an accurate verifiable reward signal (works well for math/code, harder for open-ended tasks).

---

## 14. RL Libraries and Environments

**Gymnasium** (formerly OpenAI Gym): standard interface for RL environments. Classic environments:
- `CartPole-v1` — balance a pole on a cart (discrete actions)
- `MuJoCo` (HalfCheetah, Humanoid) — continuous control locomotion
- `Atari` (Pong, Breakout) — pixel-based game environments

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()                        # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

**Stable-Baselines3** ([stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/)): clean, well-tested PyTorch implementations of PPO, SAC, DQN, A2C, TD3. Best for getting results quickly:

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)
model.learn(total_timesteps=100_000)
model.save("ppo_cartpole")
```

**TRL (Hugging Face)** ([huggingface.co/docs/trl](https://huggingface.co/docs/trl)): RLHF/DPO/GRPO training framework for LLMs. Provides `SFTTrainer`, `RewardTrainer`, `PPOTrainer`, `DPOTrainer`, `GRPOTrainer`. The standard toolkit for LLM alignment research.

---

## 15. References

### Foundational

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction (2nd ed.).](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)

### Deep RL

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning (DQN).](https://arxiv.org/abs/1312.5602)
- [Van Hasselt et al. (2015). Deep Reinforcement Learning with Double Q-learning.](https://arxiv.org/abs/1509.06461)
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms (PPO).](https://arxiv.org/abs/1707.06347)
- [Schulman et al. (2016). High-Dimensional Continuous Control Using GAE.](https://arxiv.org/abs/1506.02438)
- [Haarnoja et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy RL.](https://arxiv.org/abs/1801.01290)

### Model-Based RL

- [Janner et al. (2019). When to Trust Your Model: MBPO.](https://arxiv.org/abs/1906.08253)
- [Silver et al. (2017). Mastering Chess and Shogi by Self-Play with AlphaZero.](https://arxiv.org/abs/1712.01815)

### LLM Alignment

- [Ouyang et al. (2022). Training Language Models to Follow Instructions with Human Feedback (InstructGPT/RLHF).](https://arxiv.org/abs/2203.02155)
- [Rafailov et al. (2023). Direct Preference Optimization (DPO).](https://arxiv.org/abs/2305.18290)
- [Ethayarajh et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization.](https://arxiv.org/abs/2402.01306)
- [DeepSeek (2025). DeepSeek-R1: Incentivizing Reasoning in LLMs via RL (GRPO).](https://arxiv.org/abs/2501.12599)
- [Multi-Agent RL: Lowe et al. (2017). Multi-Agent Actor-Critic (MADDPG).](https://arxiv.org/abs/1706.02275)

### Libraries

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [HuggingFace TRL Documentation](https://huggingface.co/docs/trl/index)
- [Gymnasium](https://gymnasium.farama.org/)
