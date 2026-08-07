from __future__ import annotations

import argparse

import torch

from agent_code.my_agent.gym_environment import BombermanGymEnv, ACTION_INDICES
from agent_code.my_agent.callbacks import act as expert_act, setup as expert_setup
from agent_code.my_agent.dagger_utils import dagger_collect

from config import DEFAULT_CONFIG, TrainingConfig
from checkpoint_manager import CheckpointManager
from opponent_pool import OpponentPool
from train import build_world_args, build_model, make_train_env, architecture_info


class ExpertPolicy:

    def __init__(self, env: BombermanGymEnv):
        self.env = env

    def predict(self, obs, deterministic: bool = True):
        game_state = self.env.get_state_for_agent(self.env.agent)
        action_str = expert_act(self.env.agent, game_state)
        return ACTION_INDICES[action_str], None


def collect_expert_demonstrations(cfg: TrainingConfig, opponents, log_dir: str, n_episodes: int):
    world_args = build_world_args(cfg, log_dir, save_replay=False)
    env = BombermanGymEnv(world_args, opponents=opponents, layer_config=cfg.env.layer_config)
    expert_setup(env.agent)
    transitions = dagger_collect(env, policy=ExpertPolicy(env), n_episodes=n_episodes)
    env.close()
    return transitions


def bc_train(model, transitions, n_epochs: int = 20, batch_size: int = 256,
             lr: float = 1e-4, device: str | None = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = model.policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    grid = torch.as_tensor(transitions.obs["grid_tensor"], dtype=torch.float32)
    feats = torch.as_tensor(transitions.obs["features"], dtype=torch.float32)
    acts = torch.as_tensor(transitions.acts, dtype=torch.long)

    n = len(acts)
    print(f"[BC] training on {n} (state, expert-action) pairs for {n_epochs} epochs on {device}")
    print(f"[BC] action label distribution: {torch.bincount(acts, minlength=6).tolist()}  "
          f"(order matches ACTIONS = UP,RIGHT,DOWN,LEFT,WAIT,BOMB)")

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss, total_correct = 0.0, 0

        for b, start in enumerate(range(0, n, batch_size)):
            idx = perm[start:start + batch_size]
            batch_obs = {
                "grid_tensor": grid[idx].to(device),
                "features": feats[idx].to(device),
            }
            batch_acts = acts[idx].to(device)

            dist = policy.get_distribution(batch_obs)
            log_probs = dist.log_prob(batch_acts)
            loss = -log_probs.mean()

            optimizer.zero_grad()
            loss.backward()

            if epoch == 0 and b == 0:
                grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
                n_with_grad = sum(1 for p in policy.parameters() if p.grad is not None)
                n_total = sum(1 for _ in policy.parameters())
                print(f"[BC][diag] first-batch total grad norm={grad_norm:.6f}  "
                      f"params_with_grad={n_with_grad}/{n_total}")

            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_correct += (dist.distribution.probs.argmax(dim=-1) == batch_acts).sum().item()

        print(f"[BC] epoch {epoch + 1:>3}/{n_epochs}  loss={total_loss / n:.4f}  acc={total_correct / n:.3f}")

    policy.to("cpu")


def overfit_sanity_check(model, transitions, n_samples: int = 64, n_steps: int = 300,
                          lr: float = 1e-3, device: str | None = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = model.policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    grid = torch.as_tensor(transitions.obs["grid_tensor"][:n_samples], dtype=torch.float32).to(device)
    feats = torch.as_tensor(transitions.obs["features"][:n_samples], dtype=torch.float32).to(device)
    acts = torch.as_tensor(transitions.acts[:n_samples], dtype=torch.long).to(device)
    batch_obs = {"grid_tensor": grid, "features": feats}

    print(f"[overfit-check] trying to drive loss -> 0 on {n_samples} fixed samples over {n_steps} steps")

    param_snapshot_start = {name: p.detach().clone() for name, p in policy.named_parameters()}

    for step in range(n_steps):
        dist = policy.get_distribution(batch_obs)
        loss = -dist.log_prob(acts).mean()
        acc = (dist.distribution.probs.argmax(dim=-1) == acts).float().mean().item()

        optimizer.zero_grad()
        loss.backward()

        if step in (0, 1, 2):
            norms = {}
            for name, p in policy.named_parameters():
                if p.grad is None:
                    norms.setdefault(name.split(".")[0] + "." + name.split(".")[1] if "." in name else name, []).append(("NONE", 0.0))
                    continue
                top = ".".join(name.split(".")[:2]) if "." in name else name
                norms.setdefault(top, []).append((name, p.grad.norm().item()))
            print(f"[overfit-check][diag] step {step} per-submodule grad norms:")
            for top, entries in norms.items():
                total = sum(v for _, v in entries)
                print(f"    {top:40s} total_grad_norm={total:.8f}  (n_params={len(entries)})")

        optimizer.step()

        if step % 25 == 0 or step == n_steps - 1:
            print(f"[overfit-check] step {step:>4}  loss={loss.item():.4f}  acc={acc:.3f}")

    total_param_delta = 0.0
    for name, p in policy.named_parameters():
        total_param_delta += (p.detach() - param_snapshot_start[name]).norm().item()
    print(f"[overfit-check] total L2 parameter change over {n_steps} steps: {total_param_delta:.6f}")

    with torch.no_grad():
        final_dist = policy.get_distribution(batch_obs)
        final_probs = final_dist.distribution.probs  # (n_samples, 6)
        preds = final_probs.argmax(dim=-1)

        pred_counts = torch.bincount(preds, minlength=6).tolist()
        true_counts = torch.bincount(acts, minlength=6).tolist()
        per_class_std = final_probs.std(dim=0).mean().item()

        print(f"[overfit-check] predicted class counts : {pred_counts}")
        print(f"[overfit-check] true label class counts: {true_counts}")
        print(f"[overfit-check] mean std of predicted probs across samples "
              f"(near 0.0 => output is ~input-independent): {per_class_std:.6f}")
        print(f"[overfit-check] example predicted prob vectors (first 3 samples):")
        for i in range(min(3, n_samples)):
            print(f"    sample {i} (true={acts[i].item()}): {final_probs[i].tolist()}")

    policy.to("cpu")


def run(cfg: TrainingConfig, n_episodes: int, n_epochs: int, diagnose_only: bool = False) -> None:
    ckman = CheckpointManager.new(cfg, {**architecture_info(cfg), "bootstrap": "behavior_cloning"})
    print(f"Starting BC-bootstrap run '{cfg.run_name}' in {ckman.run_dir}")

    pool = OpponentPool(ckman, cfg.self_play)
    opponents = pool.current_opponents()

    print(f"Collecting {n_episodes} expert demonstration episodes...")
    transitions = collect_expert_demonstrations(cfg, opponents, str(ckman.logs_dir), n_episodes)

    env = make_train_env(cfg, opponents, str(ckman.logs_dir))
    model = build_model(env, cfg, str(ckman.tensorboard_dir))

    if diagnose_only:
        overfit_sanity_check(model, transitions)
        env.close()
        return

    bc_train(model, transitions, n_epochs=n_epochs)

    ckpt_dir = ckman.save_checkpoint(
        model,
        timesteps=0,
        extra_metadata={
            "bootstrap": "behavior_cloning",
            "bc_episodes": n_episodes,
            "bc_epochs": n_epochs,
            "opponents": pool.last_opponent_descriptions(),
        },
    )
    env.close()

    print(f"Saved BC-pretrained checkpoint -> {ckpt_dir}")
    print(f"Now run:  python train.py --resume {cfg.run_name} --checkpoint {ckpt_dir.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200, help="Number of expert demonstration episodes to collect")
    p.add_argument("--epochs", type=int, default=20, help="Number of BC training epochs over the collected data")
    p.add_argument("--diagnose", action="store_true",
                   help="Skip real BC training; instead try to overfit a tiny fixed slice of the "
                        "collected data to check whether the optimization loop itself is working")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(DEFAULT_CONFIG, n_episodes=args.episodes, n_epochs=args.epochs, diagnose_only=args.diagnose)