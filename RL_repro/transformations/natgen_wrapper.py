













import os
import sys

_NATGEN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NatGen")
if _NATGEN_ROOT not in sys.path:
    sys.path.insert(0, _NATGEN_ROOT)

from src.data_preprocessors.transformations.for_while_transformation import ForWhileTransformer
from src.data_preprocessors.transformations.block_swap_transformations import BlockSwap
from src.data_preprocessors.transformations.confusion_remove import ConfusionRemover


class NatGenWrapper:

    
    NATGEN_ACTIONS = {1, 2, 3, 5}

    

    def __init__(self, parser_path: str):
        if not os.path.exists(parser_path):
            raise FileNotFoundError(
                f"tree-sitter parser not found: {parser_path}\n"
                f"Build the NatGen parser first."
            )

        self.parser_path = parser_path
        self.language = "java"

        
        self._for_while = None
        self._block_swap = None
        self._confusion = None

    
    
    

    @property
    def for_while(self):
        if self._for_while is None:
            self._for_while = ForWhileTransformer(self.parser_path, self.language)
        return self._for_while

    @property
    def block_swap(self):
        if self._block_swap is None:
            self._block_swap = BlockSwap(self.parser_path, self.language)
        return self._block_swap

    @property
    def confusion(self):
        if self._confusion is None:
            self._confusion = ConfusionRemover(self.parser_path, self.language)
        return self._confusion

    
    
    

    
    
    

    def _apply_single(self, code_string, transformer, func_index):
        saved = transformer.transformations
        transformer.transformations = [saved[func_index]]
        try:
            transformed, meta = transformer.transform_code(code_string)
            return transformed, meta.get("success", False)
        finally:
            transformer.transformations = saved

    def transform(self, code_string: str, action_id: int):
        try:
            if action_id == 1:
                
                transformed, success = self._apply_single(code_string, self.for_while, 0)
            elif action_id == 2:
                
                transformed, success = self._apply_single(code_string, self.for_while, 1)
            elif action_id == 3:
                
                transformed, meta = self.block_swap.transform_code(code_string)
                success = meta.get("success", False)
            elif action_id == 5:
                
                transformed, success = self._apply_single(code_string, self.confusion, 1)
            else:
                return code_string, False

            return transformed, success

        except Exception:
            return code_string, False

    def close(self):
        self._for_while = None
        self._block_swap = None
        self._confusion = None
        
