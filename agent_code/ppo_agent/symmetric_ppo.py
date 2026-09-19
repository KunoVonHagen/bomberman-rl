from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance

from .symmetry import (
    N_SYMMETRIES,
    action_perm_torch,
    transform_features_torch,
    transform_grid_torch,
    transform_masks_torch,
)


class SymmetricMaskablePPO(MaskablePPO):
    symmetry_enabled: bool = True
    symmetry_coef: float = 0.5
    symmetry_value_coef: float = 0.5

    def _dist_and_values(self, obs, masks):
        """Same computation as policy.evaluate_actions, but returns the whole distribution."""
        pol = self.policy
        features = pol.extract_features(obs)
        if pol.share_features_extractor:
            latent_pi, latent_vf = pol.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = pol.mlp_extractor.forward_actor(pi_features)
            latent_vf = pol.mlp_extractor.forward_critic(vf_features)
        dist = pol._get_action_dist_from_latent(latent_pi)
        dist.apply_masking(masks)
        return dist, pol.value_net(latent_vf)

    def _symmetry_terms(self, rollout_data, logp):
        obs = rollout_data.observations
        masks = rollout_data.action_masks
        valid = masks.bool() if masks.dtype != th.bool else masks
        b = valid.shape[0]
        ks = th.randint(1, N_SYMMETRIES, (b,), device=valid.device)

        obs_t = dict(obs)
        obs_t["grid_tensor"] = transform_grid_torch(obs["grid_tensor"], ks)
        obs_t["features"] = transform_features_torch(obs["features"], ks)
        dist_t, values_t = self._dist_and_values(obs_t, transform_masks_torch(valid, ks))

        logp_back = th.gather(dist_t.distribution.logits, 1, action_perm_torch(ks))
        diff = logp - logp_back
        zero = th.zeros_like(diff)
        kl_pq = th.where(valid, logp.exp() * diff, zero).sum(1)
        kl_qp = th.where(valid, -logp_back.exp() * diff, zero).sum(1)
        sym_kl = 0.5 * (kl_pq + kl_qp).mean()

        sym_value_loss = F.mse_loss(rollout_data.returns, values_t.flatten())
        return sym_kl, sym_value_loss

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        use_sym = bool(self.symmetry_enabled) and (self.symmetry_coef > 0 or self.symmetry_value_coef > 0)

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        sym_kls, sym_value_losses = [], []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                dist, values = self._dist_and_values(rollout_data.observations, rollout_data.action_masks)
                log_prob = dist.log_prob(actions)
                entropy = dist.entropy()
                logp_all = dist.distribution.logits

                values = values.flatten()
                advantages = rollout_data.advantages
                if getattr(self, "normalize_advantage", True):
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                if use_sym:
                    sym_kl, sym_value_loss = self._symmetry_terms(rollout_data, logp_all)
                    loss = (loss
                            + self.symmetry_coef * sym_kl
                            + self.symmetry_value_coef * self.vf_coef * sym_value_loss)
                    sym_kls.append(sym_kl.item())
                    sym_value_losses.append(sym_value_loss.item())

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if sym_kls:
            self.logger.record("train/sym_kl", np.mean(sym_kls))
            self.logger.record("train/sym_value_loss", np.mean(sym_value_losses))