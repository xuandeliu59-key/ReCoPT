


import argparse
import os
import datetime
import numpy as np
import torch.nn as nn
from typing import Callable
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback 
from envs.RL_env import RL_Env
from models.code_embedding import Extractor
from transformations.code_modifier import CodeModifier
from models.OOD_detector import MahalanobisScorer



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def mask_fn(env):
    return env.unwrapped.current_mask.copy()


class DetailedLoggingCallback(BaseCallback):
    STRATEGY = {69: "Normalized", 70: "Obfuscated", 71: "SemanticSwap", 72: "CasePerturb"}

    def __init__(self, target_episodes=300, log_dir="training_logs", verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.target_episodes = target_episodes
        self._init_done = False
        self._worker_episodes = {}     
        self._worker_buffers = {}      

        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _init_workers(self, num_workers):
        for wid in range(num_workers):
            
            path = os.path.join(self.log_dir, f"worker_{wid}_{self.timestamp}.txt")
            self._worker_files[wid] = open(path, "w", encoding="utf-8", buffering=1)
            self._worker_episodes[wid] = 0
            self._worker_buffers[wid] = ""
            self._worker_files[wid].write(
                f"Worker {wid} log | started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
        self._init_done = True

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        if not self._init_done:
            self._worker_files = {}
            self._init_workers(len(infos))

        for worker_id in range(len(infos)):
            info = infos[worker_id]
            phase = info.get("executed_phase", "UNK")
            mask = info.get("executed_mask", np.zeros(73, dtype=np.int8))
            action = actions[worker_id]
            reward = rewards[worker_id]
            distance = info.get("current_distance", -1)
            is_legal = "OK" if mask[action] == 1 else "ERR"
            strategy = self.STRATEGY.get(action, "")

            line = (
                f"phase={phase.ljust(5)} | action={str(action).rjust(2)}"
                f"{'('+strategy+')' if strategy else '':<14}"
                f" | reward={reward:+.4f} | dist={distance:7.2f} | {is_legal}\n"
            )
            self._worker_buffers[worker_id] += line

            if dones[worker_id]:
                self.episode_count += 1
                self._worker_episodes[worker_id] += 1
                self._worker_buffers[worker_id] += (
                    f"Episode {self._worker_episodes[worker_id]} complete "
                    f"(global {self.episode_count})\n\n"
                )
                
                self._worker_files[worker_id].write(self._worker_buffers[worker_id])
                self._worker_files[worker_id].flush()
                self._worker_buffers[worker_id] = ""

                if self.episode_count % 100 == 0:
                    print(f"[Progress] Completed {self.episode_count} episodes")

                if self.episode_count >= self.target_episodes:
                    print(f"\n[Done] Reached {self.target_episodes} episodes\n")
                    self._close_files()
                    return False

        return True

    def _close_files(self):
        for wid in sorted(self._worker_files.keys()):
            if self._worker_buffers.get(wid):
                self._worker_files[wid].write(self._worker_buffers[wid])
            self._worker_files[wid].write(
                f"\nWorker {wid} completed {self._worker_episodes.get(wid, 0)} episodes\n"
            )
            self._worker_files[wid].close()
        self._worker_files.clear()
        self._worker_buffers.clear()
        self._init_done = False

    def _on_training_end(self) -> None:
        if self._init_done and hasattr(self, '_worker_files'):
            self._close_files()


class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_ent=0.08, final_ent=0.01, verbose=0):
        super(EntropyDecayCallback, self).__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent

    def _on_step(self) -> bool:
        progress_remaining = self.model._current_progress_remaining 
        current_ent = self.final_ent + progress_remaining * (self.initial_ent - self.final_ent)
        self.model.ent_coef = current_ent
        return True



def main():
    parser = argparse.ArgumentParser(description="Train 73-action MaskablePPO")
    parser.add_argument("--dataset", required=True, help="Training directory containing Java files")
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--output_model", default="outputs/code_perturb_ppo_73action")
    parser.add_argument("--log_dir", default="outputs/training_logs")
    parser.add_argument("--checkpoint_dir", default="outputs/models_checkpoints")
    parser.add_argument("--tensorboard_dir", default="outputs/tensorboard")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Training directory not found: {dataset_path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
    
    
    print("Initializing the training environment and OOD baseline...")
    extractor = Extractor()
    scorer = MahalanobisScorer(extractor)
    scorer.fit_from_folder(dataset_path)

    modifier = CodeModifier(worker_id=0)
    base_env = RL_Env(
        extractor=extractor,
        modifier=modifier,
        code_folder=dataset_path,
        scorer=scorer,
        mode="stacked",
        max_steps_cst=15,
        max_steps_token=8,
        max_steps_mixed=8
    )
    env = ActionMasker(base_env, mask_fn)
    env.reset(seed=42)
    
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[1024, 512, 256], vf=[512, 512, 256])
    )
    
    
    model = MaskablePPO(
        "MlpPolicy", 
        env=env, 
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(3e-4), 
        n_steps=512,        
        batch_size=512,
        ent_coef=0.08,      
        gamma=0.95,         
        verbose=1,
        tensorboard_log=args.tensorboard_dir
    )
    
    print("\nTraining started")
    
    target_episodes_num = args.episodes
    EXPECTED_TOTAL_TIMESTEPS = target_episodes_num * 25
    
    custom_callback = DetailedLoggingCallback(target_episodes=target_episodes_num, log_dir=args.log_dir)
    entropy_callback = EntropyDecayCallback(initial_ent=0.08, final_ent=0.01)
    
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_java_ckpt"
    )
    
    callbacks = CallbackList([custom_callback, entropy_callback, checkpoint_callback])
    
    
    try:
        model.learn(total_timesteps=EXPECTED_TOTAL_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print("\nInterrupted; saving progress...")
    finally:
        
        model.save(args.output_model)
        env.close()
        
        print(f"\n[Done] Model saved to {args.output_model}.zip")
        print(f"Checkpoints are available in {args.checkpoint_dir}")

if __name__ == "__main__":
    main()