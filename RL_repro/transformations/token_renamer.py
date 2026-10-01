














import random
from typing import List, Tuple

from tree_sitter import Language, Parser
from transformations.token_renamer_scope import ScopeAwareTokenRenamer


class LegacyTokenRenamer:

    
    
    NOT_VAR_PTYPE = {
        "function_declarator",
        "class_declaration",
        "method_declaration",
        "function_definition",
        "function_declaration",
        "call",
        "local_function_statement",
        
        "scoped_identifier",    
        "import_declaration",    
        "field_access",          
        "method_invocation",     
        "type_arguments",        
        "package_declaration",   
        "superclass",            
        "super_interfaces",      
        "object_type",           
    }

    
    SEMANTIC_MAP = {
        "result": "error",      "count": "flag",        "index": "offset",
        "length": "width",      "value": "key",         "data": "buffer",
        "input": "output",      "source": "target",     "start": "finish",
        "end": "begin",         "min": "max_val",       "max": "min_val",
        "size": "capacity",     "name": "label",        "type": "category",
        "status": "state",      "response": "request",  "request": "response",
        "current": "previous",  "next": "previous_elem","first": "last_elem",
        "last": "first_elem",   "old": "newest",        "new": "oldest",
        "temp": "buffer",       "sum": "total",         "total": "aggregate",
        "found": "missing",     "id": "identifier",     "key": "entry",
        "list": "collection",   "map": "dictionary",    "set": "group",
        "array": "sequence",    "file": "document",     "path": "route",
        "url": "endpoint",      "number": "amount",     "text": "payload",
        "flag": "marker",       "error": "warning",     "msg": "signal",
    }

    def __init__(self, parser_path: str):
        self._lang = Language(parser_path, "java")
        self._parser = Parser()
        self._parser.set_language(self._lang)

    
    
    

    def extract_variable_names(self, code_string: str) -> List[str]:
        try:
            root = self._parse(code_string)
            raw = self._collect_identifiers(root, code_string)
            if not raw:
                return []

            freq = {}
            for name in raw:
                freq[name] = freq.get(name, 0) + 1

            unique = list(freq.keys())
            unique.sort(key=lambda v: (-freq[v], v))
            return unique
        except Exception:
            return []

    def rename_variable(
        self, code_string: str, target_var: str, strategy: int
    ) -> Tuple[str, bool]:
        if strategy not in (69, 70, 71, 72):
            return code_string, False

        try:
            root = self._parse(code_string)

            
            positions = []
            self._find_positions(root, target_var, code_string, positions)

            if not positions:
                return code_string, False

            
            replacement = self._make_replacement(target_var, strategy, code_string)

            
            code_bytes = code_string.encode()
            for start, end in reversed(sorted(positions)):
                code_bytes = (
                    code_bytes[:start]
                    + replacement.encode()
                    + code_bytes[end:]
                )

            return code_bytes.decode(), True

        except Exception:
            return code_string, False

    def close(self):
        pass

    
    
    

    def _parse(self, code_string: str):
        if isinstance(code_string, str):
            return self._parser.parse(code_string.encode()).root_node
        return self._parser.parse(code_string).root_node

    def _collect_identifiers(self, root, code_string: str) -> List[str]:
        names = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            if self._is_variable(node):
                b = code_string.encode()[node.start_byte : node.end_byte]
                names.append(b.decode())
            for child in node.children:
                queue.append(child)
        return names

    def _find_positions(self, root, target_var: str, code_string: str, out: list):
        queue = [root]
        while queue:
            node = queue.pop(0)
            if self._is_variable(node):
                text = code_string.encode()[
                    node.start_byte : node.end_byte
                ].decode()
                if text == target_var:
                    out.append((node.start_byte, node.end_byte))
            for child in node.children:
                queue.append(child)

    def _is_variable(self, node) -> bool:
        return (
            node.type in ("identifier", "variable_name")
            and str(node.parent.type) not in self.NOT_VAR_PTYPE
        )

    
    
    

    def _make_replacement(
        self, target_var: str, strategy: int, code_string: str
    ) -> str:
        if strategy == 69:
            return self._normalized_name(target_var, code_string)
        elif strategy == 70:
            return self._obfuscated_name()
        elif strategy == 71:
            return self._semantic_swap_name(target_var)
        else:  
            return self._case_perturb(target_var)

    def _normalized_name(self, target_var: str, code_string: str) -> str:
        all_vars = self.extract_variable_names(code_string)
        all_sorted = sorted(set(all_vars))
        try:
            idx = all_sorted.index(target_var)
        except ValueError:
            idx = 0
        return f"VAR_{idx}"

    def _obfuscated_name(self) -> str:
        length = random.randint(3, 6)
        first = random.choice("abcdefghijklmnopqrstuvwxyz")
        rest = "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length - 1)
        )
        return first + rest

    def _semantic_swap_name(self, target_var: str) -> str:
        lower = target_var.lower()
        if lower in self.SEMANTIC_MAP:
            return self.SEMANTIC_MAP[lower]
        return self._obfuscated_name()

    def _case_perturb(self, target_var: str) -> str:
        words = self._split_camel(target_var)
        if not words:
            return self._obfuscated_name()

        styles = ["snake", "pascal", "lower", "upper_snake"]
        random.shuffle(styles)

        for style in styles:
            if style == "snake":
                result = "_".join(w.lower() for w in words)
            elif style == "pascal":
                result = "".join(w[0].upper() + w[1:].lower() for w in words)
            elif style == "lower":
                result = "".join(w.lower() for w in words)
            else:
                result = "_".join(w.upper() for w in words)

            if result != target_var:
                return result

        return target_var + "_x"

    @staticmethod
    def _split_camel(name: str) -> List[str]:
        words = []
        current = ""
        for ch in name:
            if ch.isupper() and current:
                words.append(current)
                current = ch
            else:
                current += ch
        if current:
            words.append(current)
        return words if words else [name]


class TokenRenamer(ScopeAwareTokenRenamer):

    SEMANTIC_MAP = LegacyTokenRenamer.SEMANTIC_MAP

    def _unique_replacement(self, target_var, strategy, code_string, reserved):
        if strategy == 69:
            declared = sorted(set(self.extract_variable_names(code_string)))
            index = declared.index(target_var) if target_var in declared else 0
            base = f"VAR_{index}"
        elif strategy == 70:
            base = self._legacy_obfuscated_name()
        elif strategy == 71:
            base = self.SEMANTIC_MAP.get(
                target_var.lower(), self._legacy_obfuscated_name()
            )
        else:
            base = self._legacy_case_perturb(target_var)

        candidate = base
        suffix = 1
        while candidate in reserved or candidate in self._java_keywords():
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _legacy_obfuscated_name():
        length = random.randint(3, 6)
        first = random.choice("abcdefghijklmnopqrstuvwxyz")
        rest = "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length - 1)
        )
        return first + rest

    @staticmethod
    def _legacy_case_perturb(target_var):
        words = LegacyTokenRenamer._split_camel(target_var)
        styles = ["snake", "pascal", "lower", "upper_snake"]
        random.shuffle(styles)
        for style in styles:
            if style == "snake":
                result = "_".join(word.lower() for word in words)
            elif style == "pascal":
                result = "".join(
                    word[0].upper() + word[1:].lower() for word in words
                )
            elif style == "lower":
                result = "".join(word.lower() for word in words)
            else:
                result = "_".join(word.upper() for word in words)
            if result != target_var:
                return result
        return target_var + "_x"

    @staticmethod
    def _java_keywords():
        from transformations.token_renamer_scope import JAVA_KEYWORDS

        return JAVA_KEYWORDS
