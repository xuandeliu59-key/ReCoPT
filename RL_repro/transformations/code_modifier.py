
import os

import numpy as np
from transformations.java_bridge_memory import JavaDaemon, OpenRewriteJPypeEngine, OPENREWRITE_RECIPES





from transformations.natgen_wrapper import NatGenWrapper




from transformations.token_renamer import TokenRenamer

class CodeModifier:
    def __init__(self, worker_id=0):
        
        
        
        
        
        
        
        
        
        self.num_actions = 73

        self.reciprocal_pairs = []
        self.dead_actions = set()
        self.is_cache_initialized = False

        
        
        
        
        transformations_dir = os.path.dirname(os.path.abspath(__file__))
        _natgen_parser = os.path.join(
            transformations_dir, "NatGen", "parser", "languages.so"
        )
        self.natgen = NatGenWrapper(_natgen_parser)

        spat_jar = os.path.join(transformations_dir, "spat-1.0.jar")
        rewrite_jar = os.path.join(transformations_dir, "openrewrite-1.0.jar")

        
        self.spat_daemon = JavaDaemon(spat_jar, log_prefix=f"SPAT_Worker_{worker_id}")
        self.rewrite_daemon = OpenRewriteJPypeEngine(
            rewrite_jar=rewrite_jar,
            spat_jar=spat_jar,
            log_prefix=f"JPype_Worker_{worker_id}"
        )

        
        
        
        self.token_renamer = TokenRenamer(_natgen_parser)

    def reset_cache(self):
        self.dead_actions = set()
        self.is_cache_initialized = False

    def extract_target_tokens(self, code_string, sort_rule="frequency"):
        try:
            return self.token_renamer.extract_variable_names(code_string)
        except Exception:
            return []

    
    
    
    def _strip_comments(self, code_string):
        
        if '//' not in code_string and '/*' not in code_string:
            return code_string
        try:
            from tree_sitter import Language, Parser
            _parser_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "NatGen", "parser", "languages.so"
            )
            if not hasattr(self, '_cs_parser'):
                _lang = Language(_parser_path, "java")
                self._cs_parser = Parser()
                self._cs_parser.set_language(_lang)
            root = self._cs_parser.parse(code_string.encode()).root_node
            if root.has_error:
                return code_string
            comments = []
            def _walk(n):
                if str(n.type) in ('line_comment', 'block_comment'):
                    comments.append((n.start_byte, n.end_byte))
                for c in n.children:
                    _walk(c)
            _walk(root)
            if not comments:
                return code_string
            data = code_string.encode()
            for s, e in reversed(comments):
                data = data[:s] + data[e:]
            return data.decode()
        except Exception:
            return code_string

    def apply_java_transformation(self, code_string, action_id):
        
        
        
        code_string = self._strip_comments(code_string)

        if 1 <= action_id <= 17:
            
            
            
            
            
            if action_id in NatGenWrapper.NATGEN_ACTIONS:
                transformed, success = self.natgen.transform(code_string, action_id)
                if success:
                    return transformed
                return code_string
            
            
            
            return self.spat_daemon.execute_rule(code_string, str(action_id), timeout=5.0)
        elif 18 <= action_id <= 68:
            recipe_name = OPENREWRITE_RECIPES.get(action_id)
            if recipe_name:
                return self.rewrite_daemon.execute_rule(code_string, recipe_name)
        return code_string

    def apply_transformation(self, code_string, action, target_token=None):
        if action == 0:
            return code_string

        if 1 <= action <= 68:
            return self.apply_java_transformation(code_string, action)

        elif 69 <= action <= 72:
            if target_token is None:
                return code_string
            
            code_string = self._strip_comments(code_string)
            transformed, success = self.token_renamer.rename_variable(
                code_string, target_token, action
            )
            return transformed if success else code_string

        return code_string

    def get_all_valid_actions(self, code_string, target_token=None):
        mask = np.zeros(self.num_actions, dtype=np.int8)

        def clean_code(code):
            if not isinstance(code, str):
                return ""
            return code.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        clean_original = clean_code(code_string)

        if not self.is_cache_initialized:
            self.dead_actions = set()
            for act in range(self.num_actions):
                
                if act >= 69 and target_token is None:
                    continue
                
                if 1 <= act <= 68 and target_token is not None:
                    continue
                try:
                    new_code = self.apply_transformation(code_string, act, target_token=target_token)
                    if clean_code(new_code) != clean_original:
                        mask[act] = 1
                except Exception:
                    pass
            
            paired_actions = set()
            for a, b in self.reciprocal_pairs:
                paired_actions.add(a)
                paired_actions.add(b)
                if mask[a] == 0 and mask[b] == 0:
                    self.dead_actions.add(a)
                    self.dead_actions.add(b)
            
            for act in range(1, 69): 
                if act not in paired_actions:
                    if mask[act] == 0:
                        self.dead_actions.add(act)
                        
            self.is_cache_initialized = True
            return mask

        for act in range(self.num_actions):
            
            if act >= 69 and target_token is None:
                continue
            
            if 1 <= act <= 68 and target_token is not None:
                continue
            
            if act in self.dead_actions and act < 69:
                continue
            try:
                new_code = self.apply_transformation(code_string, act, target_token=target_token)
                if clean_code(new_code) != clean_original:
                    mask[act] = 1
            except Exception:
                pass
                
        mask[0] = 1 
        return mask
        
    def close(self):
        self.spat_daemon.close()
        self.rewrite_daemon.close()
        self.natgen.close()
        self.token_renamer.close()