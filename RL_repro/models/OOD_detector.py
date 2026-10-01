import os
import torch
from pytorch_ood.detector import Mahalanobis
import numpy as np
import torch.nn as nn

class MahalanobisScorer:
    def __init__(self, extractor):
        self.extractor = extractor
        self.detector = Mahalanobis(nn.Identity) 
        self.is_fitted = False
        
        self.baseline_mean = 0.0
        self.baseline_std = 1.0

    
    def extract_features_only(self, folder_path: str) -> torch.Tensor:
        print(f"[Main] Scanning '{folder_path}' for code features...")
        id_features = []
        
        with torch.no_grad():
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    if filename.endswith(".java"):
                        file_path = os.path.join(root, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                code_string = f.read()
                                vec = self.extractor.get_embedding(code_string)
                                feature_tensor = torch.tensor(vec, dtype=torch.float32).squeeze()
                                
                                id_features.append(feature_tensor.cpu()) 
                        except Exception as e:
                            pass
                        
        if len(id_features) == 0:
            raise ValueError(f"No features found in '{folder_path}'")
            
        normal_features_tensor = torch.stack(id_features)
        print(f"[Main] Extracted features for {len(normal_features_tensor)} samples")
        return normal_features_tensor

    
    def fit_from_tensor(self, normal_features_tensor: torch.Tensor):
        labels = torch.zeros(len(normal_features_tensor), dtype=torch.long)
        self.detector.fit_features(normal_features_tensor, labels)

        with torch.no_grad():
            baseline_scores = self.detector.predict_features(normal_features_tensor).cpu().numpy()
            self.baseline_mean = float(np.mean(baseline_scores))
            self.baseline_std = float(np.std(baseline_scores))
            
            if self.baseline_std < 1e-6:
                self.baseline_std = 1.0
                
        self.is_fitted = True

    
    def fit_from_folder(self, folder_path: str):
        tensor_data = self.extract_features_only(folder_path)
        self.fit_from_tensor(tensor_data)
        print(f"Fit complete: mean={self.baseline_mean:.4f}, std={self.baseline_std:.4f}")

    def get_outlier_score(self, code_string: str) -> float:
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted")
            
        with torch.no_grad():
            vec = self.extractor.get_embedding(code_string)
            feature_tensor = torch.tensor(vec, dtype=torch.float32).squeeze().unsqueeze(0)
            original_score = self.detector.predict_features(feature_tensor).item()
            linear_score = (original_score - self.baseline_mean) / self.baseline_std
            score = max(0.0, linear_score) 
            
        return score