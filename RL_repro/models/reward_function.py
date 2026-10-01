
import numpy as np

class DynamicRewardCalculator:
    def __init__(self, step_penalty=0.02, use_dynamic=False, w_dist=0.7, w_score=0.3, ema_alpha=0.99):
        self.step_penalty = step_penalty
        self.use_dynamic = use_dynamic
        self.ema_alpha = ema_alpha
        
        
        total_weight = w_dist + w_score
        self.fixed_w_dist = w_dist / total_weight
        self.fixed_w_score = w_score / total_weight

        
        self.mu_dist = 0.0
        self.var_dist = 1.0  
        self.mu_score = 0.0
        self.var_score = 1.0
        
        
        self.current_w_dist = self.fixed_w_dist
        self.current_w_score = self.fixed_w_score

    def compute_step_reward(self, delta_dist: float, delta_score: float) -> float:
        if self.use_dynamic:
            
            self.mu_dist = self.ema_alpha * self.mu_dist + (1 - self.ema_alpha) * delta_dist
            self.var_dist = self.ema_alpha * self.var_dist + (1 - self.ema_alpha) * ((delta_dist - self.mu_dist) ** 2)
            
            self.mu_score = self.ema_alpha * self.mu_score + (1 - self.ema_alpha) * delta_score
            self.var_score = self.ema_alpha * self.var_score + (1 - self.ema_alpha) * ((delta_score - self.mu_score) ** 2)
            
            
            w_dist_raw = 1.0 / (2 * (self.var_dist + 1e-5))
            w_score_raw = 1.0 / (2 * (self.var_score + 1e-5))
            
            
            w_total = w_dist_raw + w_score_raw
            self.current_w_dist = w_dist_raw / w_total
            self.current_w_score = w_score_raw / w_total
        else:
            
            self.current_w_dist = self.fixed_w_dist
            self.current_w_score = self.fixed_w_score
        
        
        reward = self.current_w_dist * delta_dist + self.current_w_score * delta_score - self.step_penalty
        return float(reward)

    def compute_absolute_objective(self, current_distance: float, current_score: float, original_score: float) -> float:
        return self.current_w_dist * current_distance + self.current_w_score * (current_score - original_score)

    def calculate_phase_exit_reward(self, current_distance: float, distance_threshold: float, is_cst_phase: bool) -> float:
        if is_cst_phase:
            return 1.0 if current_distance >= distance_threshold else -0.3
        else:
            return 1.0 if current_distance >= distance_threshold else 0.0