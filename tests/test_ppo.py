"""Plain unit tests for isocover.ppo: shapes and GAE sanity checks."""

import torch

from isocover import ppo


def test_gae_shapes():
  t, n = 10, 4
  rewards = torch.randn(t, n)
  values = torch.randn(t, n)
  dones = torch.zeros(t, n, dtype=torch.bool)
  last_value = torch.randn(n)
  adv, ret = ppo.gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95, episodic=True)
  assert adv.shape == (t, n)
  assert ret.shape == (t, n)


def test_gae_zero_reward_zero_value_gives_zero_advantage():
  t, n = 5, 3
  rewards = torch.zeros(t, n)
  values = torch.zeros(t, n)
  dones = torch.zeros(t, n, dtype=torch.bool)
  last_value = torch.zeros(n)
  adv, ret = ppo.gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95, episodic=True)
  assert torch.allclose(adv, torch.zeros(t, n))
  assert torch.allclose(ret, torch.zeros(t, n))


def test_gae_episodic_vs_nonepisodic_differ_across_done():
  t, n = 6, 1
  rewards = torch.ones(t, n)
  values = torch.zeros(t, n)
  dones = torch.zeros(t, n, dtype=torch.bool)
  dones[2, 0] = True  # an episode ends after step 2
  last_value = torch.zeros(n)
  adv_ep, _ = ppo.gae(rewards, values, dones, last_value, gamma=0.9, lam=0.9, episodic=True)
  adv_non, _ = ppo.gae(rewards, values, dones, last_value, gamma=0.9, lam=0.9, episodic=False)
  # Before the done step, bootstrapping through it (non-episodic) must give a
  # STRICTLY larger advantage than stopping at it (episodic).
  assert float(adv_non[0, 0]) > float(adv_ep[0, 0])


def test_combined_advantage_matches_weighted_sum():
  cfg = ppo.PPOConfig(ext_coef=2.0, int_coef=1.0)
  adv_e = torch.tensor([1.0, 2.0])
  adv_i = torch.tensor([0.5, -1.0])
  got = ppo.combined_advantage(adv_e, adv_i, cfg)
  assert torch.allclose(got, 2.0 * adv_e + 1.0 * adv_i)


def test_recurrent_actor_critic_step_shapes():
  class _Body(torch.nn.Module):
    def forward(self, frames):
      return frames.float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1).expand(-1, 8)

  agent = ppo.RecurrentActorCritic(_Body(), feat_dim=8, n_actions=4,
                                    critics=("ext", "int"), lstm_hidden=16, seed=0)
  frame = torch.randint(0, 255, (5, 10, 10, 3), dtype=torch.uint8)
  logits, values, state = agent.step(frame)
  assert logits.shape == (5, 4)
  assert set(values.keys()) == {"ext", "int"}
  assert values["ext"].shape == (5,)
  assert state[0].shape == (1, 5, 16)


def test_recurrent_actor_critic_sequence_matches_step_replay():
  class _Body(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.lin = torch.nn.Linear(3, 8)

    def forward(self, frames):
      return self.lin(frames.float().mean(dim=(1, 2)))

  torch.manual_seed(0)
  agent = ppo.RecurrentActorCritic(_Body(), feat_dim=8, n_actions=4,
                                    critics=("value",), lstm_hidden=16, seed=0)
  agent.eval()
  t, n = 4, 3
  frames = torch.randint(0, 255, (t, n, 5, 5, 3), dtype=torch.uint8)
  done = torch.zeros(t, n, dtype=torch.bool)
  reset_before = ppo.reset_before_from_done(done)

  with torch.no_grad():
    seq_logits, seq_values, _ = agent.sequence(frames, None, reset_before)
    state = None
    step_logits = []
    for tt in range(t):
      lg, vals, state = agent.step(frames[tt], state)
      step_logits.append(lg)
    step_logits = torch.stack(step_logits, dim=0)

  assert torch.allclose(seq_logits, step_logits, atol=1e-5)


def test_reset_before_from_done_puts_zero_in_first_row():
  done = torch.tensor([[True, False], [False, True], [True, True]])
  reset_before = ppo.reset_before_from_done(done)
  assert not bool(reset_before[0].any())
  assert torch.equal(reset_before[1], done[0])
  assert torch.equal(reset_before[2], done[1])
