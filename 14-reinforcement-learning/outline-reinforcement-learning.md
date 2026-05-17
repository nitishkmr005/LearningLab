# 13 — Reinforcement Learning

Exhaustive learning path for reinforcement learning: classical algorithms, deep RL, policy optimization, and RLHF for language models.

---

## 01 — RL Fundamentals
Agent, environment, state, action, reward, episode; Markov property; Markov Decision Process (MDP); return; discount factor γ; policy π.
- https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- http://incompleteideas.net/book/the-book-2nd.html

## 02 — Value Functions & Bellman Equations
State-value V(s), action-value Q(s,a); Bellman expectation equation; Bellman optimality equation; relationship between V and Q.
- https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#value-functions

## 03 — Dynamic Programming
Policy evaluation; policy improvement; policy iteration; value iteration; tabular setting; limitations (requires full model).
- http://incompleteideas.net/book/the-book-2nd.html (Chapter 4)

## 04 — Monte Carlo Methods
First-visit vs every-visit MC; MC prediction; MC control with exploring starts; on-policy vs off-policy; importance sampling.
- http://incompleteideas.net/book/the-book-2nd.html (Chapter 5)

## 05 — Temporal-Difference Learning
TD(0) prediction; SARSA (on-policy); Q-Learning (off-policy); TD vs MC trade-offs; eligibility traces; TD(λ).
- https://www.davidsilver.uk/teaching/ (Lecture 4)

## 06 — Deep Q-Networks (DQN)
Experience replay; target network; ε-greedy exploration; Double DQN; Dueling DQN; Prioritized Experience Replay.
- https://arxiv.org/abs/1312.5602
- https://arxiv.org/abs/1509.06461

## 07 — Policy Gradient Methods
REINFORCE; likelihood ratio trick; baseline subtraction for variance reduction; actor-critic; advantage function A(s,a).
- https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
- https://arxiv.org/abs/1506.02438

## 08 — Proximal Policy Optimization (PPO)
Clipped surrogate objective; entropy bonus; GAE (Generalized Advantage Estimation); PPO-Clip vs PPO-KL; hyperparameters.
- https://arxiv.org/abs/1707.06347
- https://spinningup.openai.com/en/latest/algorithms/ppo.html

## 09 — Trust Region Policy Optimization (TRPO)
KL-divergence constraint; natural gradient; conjugate gradient; monotonic improvement guarantee; comparison with PPO.
- https://arxiv.org/abs/1502.05477

## 10 — Soft Actor-Critic (SAC)
Maximum entropy RL; entropy regularization; soft Bellman equations; reparameterization trick; automatic temperature tuning.
- https://arxiv.org/abs/1801.01290
- https://spinningup.openai.com/en/latest/algorithms/sac.html

## 11 — Model-Based RL
World models; Dyna architecture; MBPO; AlphaZero MCTS; planning with learned models; sample efficiency vs model error.
- https://arxiv.org/abs/1906.08253
- https://arxiv.org/abs/1712.01815

## 12 — Multi-Agent RL
Cooperative vs competitive settings; CTDE (centralized training, decentralized execution); MADDPG; self-play; Nash equilibria.
- https://arxiv.org/abs/1706.02275

## 13 — RLHF (Reinforcement Learning from Human Feedback)
Reward model training from preference data; PPO fine-tuning; InstructGPT pipeline; KL penalty vs clipping; reward hacking.
- https://arxiv.org/abs/2203.02155
- https://huggingface.co/blog/rlhf

## 14 — Direct Preference Optimization (DPO) & Variants
DPO derivation (implicit reward); RAFT; RLAIF; KTO; SimPO; comparison with RLHF; when to prefer DPO over PPO.
- https://arxiv.org/abs/2305.18290
- https://arxiv.org/abs/2402.01306

## 15 — RL Libraries & Environments
OpenAI Gym / Gymnasium; Stable-Baselines3; RLlib (Ray); Atari, MuJoCo, MiniGrid; TRL (Hugging Face) for RLHF.
- https://stable-baselines3.readthedocs.io/en/master/
- https://huggingface.co/docs/trl/index

## 16 — GRPO (Group Relative Policy Optimization)
DeepSeek R1's training algorithm; group sampling to estimate baseline reward without a value model; advantage normalization within group; vs PPO for LLM fine-tuning; lower memory cost; TRL implementation.
- https://arxiv.org/abs/2501.12599
- https://huggingface.co/docs/trl/grpo_trainer
