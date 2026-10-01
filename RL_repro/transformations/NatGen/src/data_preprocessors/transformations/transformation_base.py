import os
from typing import Union, Tuple, List

import tree_sitter
from tree_sitter import Language, Parser


def get_ancestor_type_chains(
        node: tree_sitter.Node
) -> List[str]:
    types = [str(node.type)]
    while node.parent is not None:
        node = node.parent
        types.append(str(node.type))
    return types


class TransformationBase:
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        if not os.path.exists(parser_path):
            raise ValueError(
                f"Language parser does not exist at {parser_path}. Please run `setup.sh` to properly set the "
                f"environment!")
        self.lang_object = Language(parser_path, language)
        self.parser = Parser()
        self.parser.set_language(self.lang_object)
        pass

    def parse_code(
            self,
            code: Union[str, bytes]
    ) -> tree_sitter.Node:
        if isinstance(code, bytes):
            tree = self.parser.parse(code)
        elif isinstance(code, str):
            tree = self.parser.parse(code.encode())
        else:
            raise ValueError("Code must be character string or bytes string")
        return tree.root_node

    def get_tokens(
            self,
            code: bytes,
            root: tree_sitter.Node
    ) -> List[str]:
        tokens = []
        if root.type == "comment":
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if "list" not in str(parent.type) and len(parent.children) == 1:
                return tokens
            else:
                return [code[root.start_byte:root.end_byte].decode()]
        if len(root.children) == 0:
            tokens.append(code[root.start_byte:root.end_byte].decode())
        else:
            for child in root.children:
                tokens += self.get_tokens(code, child)
        return tokens

    def get_token_string(
            self,
            code: str,
            root: tree_sitter.Node
    ) -> str:
        tokens = self.get_tokens(code.encode(), root)
        return " ".join(tokens)

    def get_tokens_with_node_type(
            self,
            code: bytes,
            root: tree_sitter.Node
    ) -> Tuple[List[str], List[List[str]]]:
        tokens, types = [], []
        
        if root.type in ("comment", "line_comment", "block_comment"):
            return tokens, types
        if "string" in str(root.type):
            return [code[root.start_byte:root.end_byte].decode()], [["string"]]
        if len(root.children) == 0:
            tokens.append(code[root.start_byte:root.end_byte].decode())
            types.append(get_ancestor_type_chains(root))
        else:
            for child in root.children:
                _tokens, _types = self.get_tokens_with_node_type(code, child)
                tokens += _tokens
                types += _types
        return tokens, types

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        pass

    

    def _format_output(self, token_joined_code: str) -> str:
        import re as _re
        
        code = _re.sub(r' \. ', '.', token_joined_code)
        code = _re.sub(r' \. ', '.', code)
        
        code = _re.sub(r' \+ \+', '++', code)
        code = _re.sub(r' \- \-', '--', code)
        
        
        code = _re.sub(r'\[ ', '[', code)
        code = _re.sub(r' \]', ']', code)
        code = _re.sub(r'\( ', '(', code)
        code = _re.sub(r' \)', ')', code)

        
        for_semicolons = set()
        try:
            root = self.parse_code(code)
            if not root.has_error:
                self._fmt_find_for_semicolons(root, code.encode(), for_semicolons)
        except Exception:
            pass

        
        result = []
        indent = 0
        byte_pos = 0
        code_bytes = code.encode()
        i = 0
        while i < len(code):
            c = code[i]
            char_byte_len = len(c.encode())

            if c == '{':
                
                while result and result[-1] == ' ':
                    result.pop()
                result.append(' {')
                indent += 1
                result.append('\n' + '    ' * indent)
            elif c == '}':
                indent = max(0, indent - 1)
                
                while result and result[-1].strip() == '':
                    result.pop()
                result.append('\n' + '    ' * indent + '}')
                result.append('\n' + '    ' * indent)
            elif c == ';':
                result.append(';')
                if byte_pos in for_semicolons:
                    result.append(' ')
                else:
                    result.append('\n' + '    ' * indent)
            else:
                result.append(c)

            i += 1
            byte_pos += char_byte_len

        formatted = ''.join(result)
        
        formatted = _re.sub(r'\n\s*\n\s*\n', '\n\n', formatted)
        return formatted

    def _fmt_find_for_semicolons(self, node, code_bytes, positions):
        if str(node.type) == 'for_statement':
            for child in node.children:
                if str(child.type) == ';':
                    
                    sc = code_bytes[child.start_byte:child.end_byte].decode()
                    if sc == ';':
                        positions.add(child.start_byte)
                elif str(child.type) == '(':
                    
                    for gc in child.children:
                        self._fmt_find_for_semicolons(gc, code_bytes, positions)
                else:
                    continue
        else:
            for child in node.children:
                self._fmt_find_for_semicolons(child, code_bytes, positions)
