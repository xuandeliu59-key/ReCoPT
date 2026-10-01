
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import glob

from models.reward_function import DynamicRewardCalculator

class RL_Env(gym.Env):
    def __init__(self, extractor, modifier, code_folder, scorer,
                 max_steps_per_phase=15, step_penalty=0.02, distance_threshold=30.0,
                 mode="stacked",
                 max_steps_cst=None, max_steps_token=None, max_steps_mixed=None):
        super(RL_Env, self).__init__()
        self.extractor = extractor
        self.modifier = modifier
        self.distance_threshold = distance_threshold
        self.max_steps_per_phase = max_steps_per_phase  

        
        self.max_steps = {
            "CST":   max_steps_cst   if max_steps_cst   is not None else max_steps_per_phase,
            "TOKEN": max_steps_token if max_steps_token is not None else max_steps_per_phase,
            "MIXED": max_steps_mixed if max_steps_mixed is not None else max_steps_per_phase,
        }
        self.scorer = scorer
        self.mode = mode

        self.reward_calculator = DynamicRewardCalculator(step_penalty=step_penalty)

        
        self.action_space = spaces.Discrete(73)

        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(771,), dtype=np.float32
        )

        
        self.cst_isolation_mask   = np.array([1]*69 + [0]*4, dtype=np.int8)
        self.token_isolation_mask = np.array([0]*69 + [1]*4, dtype=np.int8)
        self.mixed_isolation_mask = np.array([0]*69 + [1]*4, dtype=np.int8)

        self.code_folder = code_folder
        self.code_files = []
        self.current_file_index = 0

        if os.path.exists(self.code_folder):
            search_pattern = os.path.join(self.code_folder, "**", "*.java")
            self.code_files = glob.glob(search_pattern, recursive=True)
            print(f"Environment initialized with {len(self.code_files)} Java files")
            np.random.shuffle(self.code_files)
        else:
            print(f"Warning: directory not found: {self.code_folder}")

    
    
    

    def _get_obs(self, code_string):
        base_vector = self.extractor.get_embedding(code_string)
        if self.current_phase == "CST":
            flag = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif self.current_phase == "TOKEN":
            flag = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:  
            flag = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.append(base_vector, flag).astype(np.float32)

    def _get_reward_distance(self, observation):
        return float(np.linalg.norm(
            observation[:768] - self.reward_reference_embedding
        ))

    
    
    

    def _get_isolated_mask(self, code_string):
        next_target_token = self.pending_tokens[0] if len(self.pending_tokens) > 0 else None
        base_mask = self.modifier.get_all_valid_actions(code_string, target_token=next_target_token)
        base_mask[0] = 1

        if self.current_phase == "CST":
            isolated = base_mask * self.cst_isolation_mask
        elif self.current_phase == "TOKEN":
            isolated = base_mask * self.token_isolation_mask
        else:  
            isolated = base_mask * self.mixed_isolation_mask

        isolated[0] = 1
        return isolated

    
    
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_phase = "CST"
        self.phase_step = 0
        self.pending_tokens = []

        chosen_file_path = None
        if len(self.code_files) > 0:
            if self.current_file_index >= len(self.code_files):
                self.np_random.shuffle(self.code_files)
                self.current_file_index = 0
                print("\nStarting a new pass\n")

            chosen_file_path = self.code_files[self.current_file_index]
            print(f"\nTraining file: {chosen_file_path}")
            with open(chosen_file_path, 'r', encoding='utf-8') as f:
                self.original_code = f.read()
            self.current_file_index += 1

        self.modifier.reset_cache()
        self.current_code = self.original_code
        self.original_vector = self._get_obs(self.original_code)
        
        
        self.reward_reference_embedding = self.original_vector[:768].copy()

        self.original_score = self.scorer.get_outlier_score(self.original_code)
        self.previous_distance = 0.0
        self.previous_score = self.original_score
        self.phase_started_above_threshold = False

        
        self.best_cst_code = self.original_code
        self.best_cst_objective = 0.0
        self.best_token_code = self.original_code
        self.best_token_objective = 0.0
        self.best_mixed_code = self.original_code
        self.best_mixed_objective = 0.0

        self.final_cst_code_saved = None
        self.final_cst_score_saved = None
        self.final_token_code_saved = None
        self.final_mixed_code_saved = None

        self.current_mask = self._get_isolated_mask(self.current_code)
        info = {
            "loaded_file": chosen_file_path,
            "phase": self.current_phase,
            "action_mask": self.current_mask
        }
        return self.original_vector, info

    
    
    

    def step(self, action):
        executed_phase = self.current_phase
        executed_mask = self.current_mask.copy()

        
        if action == 0:
            return self._handle_noop(executed_phase, executed_mask)

        self.phase_step += 1

        
        current_target_token = None
        if self.current_phase in ("TOKEN", "MIXED") and len(self.pending_tokens) > 0:
            current_target_token = self.pending_tokens.pop(0)

        self.current_code = self.modifier.apply_transformation(
            self.current_code, action, target_token=current_target_token
        )
        current_vector = self._get_obs(self.current_code)

        current_distance = self._get_reward_distance(current_vector)
        current_score = self.scorer.get_outlier_score(self.current_code)

        delta_dist = current_distance - self.previous_distance
        delta_score = current_score - self.previous_score

        reward = self.reward_calculator.compute_step_reward(delta_dist, delta_score)
        current_abs_obj = self.reward_calculator.compute_absolute_objective(
            current_distance, current_score, self.original_score
        )

        self.previous_score = current_score
        self.previous_distance = current_distance

        
        self._update_best(current_abs_obj)

        
        
        
        phase_success = (not self.phase_started_above_threshold
                         and current_distance >= self.distance_threshold)
        time_limit_reached = self.phase_step >= self.max_steps[self.current_phase]
        queue_empty = (self.current_phase in ("TOKEN", "MIXED")
                       and len(self.pending_tokens) == 0)

        terminated, truncated = False, False

        if phase_success or time_limit_reached or queue_empty:
            if phase_success:
                reward += 1.0
            terminated, truncated = self._advance_phase()

            
            
            
            
            
            if not (terminated or truncated):
                current_vector = self.original_vector.copy()

        self.current_mask = self._get_isolated_mask(self.current_code)

        info = self._build_info(executed_phase, executed_mask,
                                terminated or truncated, current_distance)
        return current_vector, float(reward), terminated, truncated, info

    
    
    

    def _advance_phase(self):
        if self.current_phase == "CST":
            return self._exit_cst()
        elif self.current_phase == "TOKEN":
            return self._exit_token()
        else:  
            return self._exit_mixed()

    def _exit_cst(self):
        self.final_cst_code_saved = self.current_code
        self.final_cst_score_saved = self.previous_score

        if self.mode == "stacked":
            self._enter_mixed()
            return False, False
        else:
            
            self._enter_token(from_code=self.original_code)
            return False, False

    def _exit_token(self):
        self.final_token_code_saved = self.current_code

        if self.mode == "both":
            self._enter_mixed()
            return False, False
        
        return True, False

    def _exit_mixed(self):
        self.final_mixed_code_saved = self.current_code
        return True, False

    
    
    

    def _enter_token(self, from_code):
        self.current_phase = "TOKEN"
        self.current_code = from_code
        self.pending_tokens = self.modifier.extract_target_tokens(from_code)
        self.phase_step = 0
        self.original_vector = self._get_obs(from_code)
        self.previous_distance = self._get_reward_distance(self.original_vector)
        self.previous_score = self.original_score
        self.phase_started_above_threshold = (
            self.previous_distance >= self.distance_threshold
        )
        self.best_token_code = from_code
        self.best_token_objective = 0.0

    def _enter_mixed(self):
        self.current_phase = "MIXED"
        self.current_code = self.final_cst_code_saved
        self.pending_tokens = self.modifier.extract_target_tokens(
            self.final_cst_code_saved
        )
        self.phase_step = 0
        if self.final_cst_score_saved is None:
            self.final_cst_score_saved = self.scorer.get_outlier_score(
                self.final_cst_code_saved
            )
        self.previous_score = self.final_cst_score_saved
        self.original_vector = self._get_obs(self.final_cst_code_saved)
        self.previous_distance = self._get_reward_distance(self.original_vector)
        self.phase_started_above_threshold = (
            self.previous_distance >= self.distance_threshold
        )
        self.best_mixed_code = self.final_cst_code_saved
        self.best_mixed_objective = (
            self.reward_calculator.compute_absolute_objective(
                self.previous_distance,
                self.previous_score,
                self.original_score
            )
        )

    
    
    

    def _update_best(self, current_abs_obj):
        if self.current_phase == "CST":
            if current_abs_obj > self.best_cst_objective:
                self.best_cst_objective = current_abs_obj
                self.best_cst_code = self.current_code
        elif self.current_phase == "TOKEN":
            if current_abs_obj > self.best_token_objective:
                self.best_token_objective = current_abs_obj
                self.best_token_code = self.current_code
        elif self.current_phase == "MIXED":
            if current_abs_obj > self.best_mixed_objective:
                self.best_mixed_objective = current_abs_obj
                self.best_mixed_code = self.current_code

    def _handle_noop(self, executed_phase, executed_mask):
        exit_vector = self._get_obs(self.current_code)
        current_distance = self._get_reward_distance(exit_vector)

        
        
        exit_reward_distance = (
            0.0 if self.phase_started_above_threshold else current_distance
        )

        reward = self.reward_calculator.calculate_phase_exit_reward(
            exit_reward_distance, self.distance_threshold,
            is_cst_phase=(self.current_phase == "CST")
        )

        terminated, truncated = self._advance_phase()

        
        
        if not (terminated or truncated):
            next_vector = self.original_vector.copy()
        else:
            next_vector = exit_vector

        self.current_mask = self._get_isolated_mask(self.current_code)

        info = self._build_info(executed_phase, executed_mask,
                                terminated or truncated, current_distance)
        return next_vector, float(reward), terminated, truncated, info

    def _build_info(self, executed_phase, executed_mask, is_done, current_distance):
        return {
            "executed_phase": executed_phase,
            "executed_mask": executed_mask,
            "current_code": self.current_code,
            "phase": self.current_phase,
            "action_mask": self.current_mask,
            "best_cst_code": self.best_cst_code if is_done else None,
            "final_cst_code": self.final_cst_code_saved if is_done else None,
            "best_token_code": self.best_token_code if is_done else None,
            "final_token_code": self.final_token_code_saved if is_done else None,
            "best_mixed_code": self.best_mixed_code if is_done else None,
            "final_mixed_code": self.final_mixed_code_saved if is_done else None,
            "current_distance": float(current_distance)
        }

    def close(self):
        if hasattr(self, 'modifier'):
            self.modifier.close()
        super().close()
