import json
import os
import queue
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from time import sleep, time
from tqdm import tqdm

import settings as s
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


class Timekeeper:
    def __init__(self, interval):
        self.interval = interval
        self.next_time = None

    def is_due(self):
        return self.next_time is None or time() >= self.next_time

    def note(self):
        self.next_time = time() + self.interval

    def wait(self):
        if not self.is_due():
            duration = self.next_time - time()
            sleep(duration)


def world_controller(world, n_rounds, *,
                     gui, every_step, turn_based, make_video, update_interval,
                     progress_callback=None):
    if make_video and not gui.screenshot_dir.exists():
        gui.screenshot_dir.mkdir()

    gui_timekeeper = Timekeeper(update_interval)

    def render(wait_until_due):
        # If every step should be displayed, wait until it is due to be shown
        if wait_until_due:
            gui_timekeeper.wait()

        if gui_timekeeper.is_due():
            gui_timekeeper.note()
            # Render (which takes time)
            gui.render()
            pygame.display.flip()

    user_input = None
    round_range = range(n_rounds) if progress_callback is not None else tqdm(range(n_rounds))
    for _ in round_range:
        world.new_round()
        while world.running:
            # Only render when the last frame is not too old
            if gui is not None:
                render(every_step)

                # Check GUI events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in ESCAPE_KEYS:
                            world.end_round()
                        elif key_pressed in s.INPUT_MAP:
                            user_input = s.INPUT_MAP[key_pressed]

            # Advances step (for turn based: only if user input is available)
            if world.running and not (turn_based and user_input is None):
                world.do_step(user_input)
                user_input = None
            else:
                # Might want to wait
                pass

        # Save video of last game
        if make_video:
            gui.make_video()

        # Render end screen until next round is queried
        if gui is not None:
            do_continue = False
            while not do_continue:
                render(True)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in s.INPUT_MAP or key_pressed in ESCAPE_KEYS:
                            do_continue = True

        if progress_callback is not None:
            progress_callback()

    world.end()


def _run_worker(args_dict, agents, n_rounds, worker_id, result_path, progress_queue):
    worker_args = Namespace(**args_dict)
    worker_args.n_rounds = n_rounds
    worker_args.no_gui = True
    worker_args.make_video = False
    worker_args.save_stats = str(result_path)
    worker_args.log_dir = str(Path(args_dict["log_dir"]) / f"worker_{worker_id}")

    base_match_name = args_dict.get("match_name") or "match"
    worker_args.match_name = f"{base_match_name}_w{worker_id}"

    base_seed = args_dict.get("seed")
    worker_args.seed = None if base_seed is None else base_seed + worker_id

    world = BombeRLeWorld(worker_args, agents)
    world_controller(world, n_rounds,
                     gui=None, every_step=False, turn_based=False,
                     make_video=False, update_interval=worker_args.update_interval,
                     progress_callback=lambda: progress_queue.put(1))


def _merge_results(result_paths):
    merged = {"by_agent": defaultdict(lambda: defaultdict(int)), "by_round": {}}
    for path in result_paths:
        if not path.exists():
            continue
        with open(path) as file:
            data = json.load(file)
        for agent_name, stats in data.get("by_agent", {}).items():
            for key, value in stats.items():
                merged["by_agent"][agent_name][key] += value
        merged["by_round"].update(data.get("by_round", {}))
    return {
        "by_agent": {name: dict(stats) for name, stats in merged["by_agent"].items()},
        "by_round": merged["by_round"],
    }


def run_parallel(args, agents, n_workers):
    n_rounds = args.n_rounds
    base, remainder = divmod(n_rounds, n_workers)
    chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]

    tmp_dir = Path(args.log_dir).parent / "results" / "_parallel_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args).copy()
    progress_queue = mp.Queue()
    processes = []
    result_paths = []
    for worker_id, n_worker_rounds in enumerate(chunks):
        if n_worker_rounds == 0:
            continue
        result_path = tmp_dir / f"worker_{worker_id}.json"
        result_paths.append(result_path)
        p = mp.Process(target=_run_worker, args=(args_dict, agents, n_worker_rounds, worker_id, result_path, progress_queue))
        p.start()
        processes.append(p)

    completed = 0
    with tqdm(total=n_rounds, desc="rounds played") as pbar:
        while completed < n_rounds:
            try:
                progress_queue.get(timeout=1)
            except queue.Empty:
                if all(not p.is_alive() for p in processes):
                    break
                continue
            completed += 1
            pbar.update(1)

    for p in processes:
        p.join()

    merged = _merge_results(result_paths)

    if args.save_stats is not False:
        if args.save_stats is not True:
            file_name = args.save_stats
        elif args.match_name is not None:
            file_name = f"results/{args.match_name}.json"
        else:
            from datetime import datetime
            file_name = f"results/{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.json"

        out_path = Path(file_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as file:
            json.dump(merged, file, indent=4, sort_keys=True)


def main(argv = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument("--seed", type=int, help="Reset the world's random number generator to a known number for reproducibility")

    play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
    play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?', help='Store the game as .pt for a replay')
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

    group = play_parser.add_mutually_exclusive_group()
    group.add_argument("--skip-frames", default=False, action="store_true", help="Play several steps per GUI render.")
    group.add_argument("--no-gui", default=False, action="store_true", help="Deactivate the user interface and play as fast as possible.")

    play_parser.add_argument("--parallel", type=int, default=1,
                             help="Run this many worker processes in parallel, splitting --n-rounds between them. Requires --no-gui.")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--turn-based", default=False, action="store_true",
                         help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1,
                         help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

        # Video?
        sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?',
                         help="Make a video from the game")

    args = parser.parse_args(argv)
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1
        args.match_name = Path(args.replay).name

    has_gui = not args.no_gui
    if has_gui:
        if not LOADED_PYGAME:
            raise ValueError("pygame could not loaded, cannot run with GUI")

    # Initialize environment and agents
    if args.command_name == "play":
        agents = []
        if args.train == 0 and not args.continue_without_training:
            args.continue_without_training = True
        if args.my_agent:
            agents.append((args.my_agent, len(agents) < args.train))
            args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))

        if args.parallel > 1:
            if has_gui:
                raise ValueError("--parallel requires --no-gui")
            run_parallel(args, agents, args.parallel)
            return
        world = BombeRLeWorld(args, agents)
        every_step = not args.skip_frames
    elif args.command_name == "replay":
        world = ReplayWorld(args)
        every_step = True
    else:
        raise ValueError(f"Unknown command {args.command_name}")

    # Launch GUI
    if has_gui:
        gui = GUI(world)
    else:
        gui = None
    world_controller(world, args.n_rounds,
                     gui=gui, every_step=every_step, turn_based=args.turn_based,
                     make_video=args.make_video, update_interval=args.update_interval)


if __name__ == '__main__':
    main()