


import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

class Extractor:
    def __init__(self, model_name="microsoft/unixcoder-base"):
        print(f"Loading feature extractor: {model_name}")
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            clean_up_tokenization_spaces=True
        ) 
        self.model = AutoModel.from_pretrained(model_name) 
        
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() 
        
        
        print(f"Model loaded on {self.device}")

    def get_embedding(self, code_string: str) -> np.ndarray:
        tokens = self.tokenizer(
            code_string, 
            return_tensors='pt', 
            max_length=1024, 
            truncation=True, 
            padding='max_length'
        )
        
        
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            
        
        cls_embedding = outputs.last_hidden_state[0, 0, :] 
        
        
        return cls_embedding.cpu().numpy().astype(np.float32)