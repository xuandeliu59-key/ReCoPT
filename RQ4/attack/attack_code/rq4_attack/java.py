from __future__ import annotations

import keyword
import os
import re
from pathlib import Path

from tree_sitter import Language, Parser


ATTACK_CODE_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_SO = Path(
    os.environ.get(
        "RQ4_JAVA_LANGUAGE_SO",
        str(ATTACK_CODE_ROOT / "ITGen_official/python_parser/parser_folder/my-languages.so"),
    )
).expanduser().resolve()

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new", "package",
    "private", "protected", "public", "return", "short", "static", "strictfp", "super",
    "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void",
    "volatile", "while", "true", "false", "null", "var", "record", "sealed", "permits",
}
PROTECTED_NAMES = {
    "main", "args", "System", "String", "Math", "Integer", "Long", "Double", "Float",
    "Boolean", "Character", "Byte", "Short", "Object", "Scanner", "List", "Map", "Set",
}


class JavaTools:

    def __init__(self, language_so: Path | str | None = None):
        language_so = Path(language_so).expanduser().resolve() if language_so else LANGUAGE_SO
        if not language_so.is_file():
            raise FileNotFoundError(language_so)
        language = Language(str(language_so), "java")
        parser = Parser()
        parser.set_language(language)
        self.parser = parser

    def tree(self, code: str):
        return self.parser.parse(code.encode("utf-8"))

    def syntax_ok(self, code: str) -> bool:
        return not self.tree(code).root_node.has_error

    @staticmethod
    def _text(code_bytes: bytes, node) -> str:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def identifiers(self, code: str) -> tuple[list[str], list[str]]:
        code_bytes = code.encode("utf-8")
        root = self.tree(code).root_node
        variables: list[str] = []
        functions: list[str] = []
        variable_nodes = {
            "variable_declarator", "formal_parameter", "spread_parameter",
            "catch_formal_parameter", "resource", "enhanced_for_statement",
        }
        function_nodes = {"method_declaration", "constructor_declaration"}

        def visit(node) -> None:
            if node.type in variable_nodes:
                name = node.child_by_field_name("name")
                if name is not None:
                    variables.append(self._text(code_bytes, name))
            elif node.type in function_nodes:
                name = node.child_by_field_name("name")
                if name is not None:
                    functions.append(self._text(code_bytes, name))
            for child in node.children:
                visit(child)

        visit(root)
        valid_vars = _unique(name for name in variables if valid_identifier(name))
        valid_funcs = _unique(
            name for name in functions if valid_identifier(name) and name != "main"
        )
        return valid_vars, valid_funcs

    def token_texts(self, code: str) -> list[str]:
        code_bytes = code.encode("utf-8")
        values: list[str] = []

        def visit(node) -> None:
            if node.child_count == 0:
                values.append(self._text(code_bytes, node))
                return
            for child in node.children:
                visit(child)

        visit(self.tree(code).root_node)
        return values

    def replace_identifiers(self, code: str, replacements: dict[str, str]) -> str:
        code_bytes = code.encode("utf-8")
        root = self.tree(code).root_node
        edits: list[tuple[int, int, bytes]] = []

        def visit(node) -> None:
            if node.type == "identifier":
                old = self._text(code_bytes, node)
                new = replacements.get(old)
                if new and new != old:
                    parent = node.parent
                    is_qualified_method = (
                        parent is not None
                        and parent.type == "method_invocation"
                        and parent.child_by_field_name("name") == node
                        and parent.child_by_field_name("object") is not None
                    )
                    blocked = parent is not None and (
                        (parent.type == "field_access" and parent.child_by_field_name("field") == node)
                        or is_qualified_method
                        or parent.type in {
                            "scoped_identifier", "import_declaration", "package_declaration",
                            "type_identifier", "break_statement", "continue_statement",
                        }
                    )
                    if not blocked:
                        edits.append((node.start_byte, node.end_byte, new.encode("utf-8")))
            for child in node.children:
                visit(child)

        visit(root)
        for start, end, replacement in sorted(edits, reverse=True):
            code_bytes = code_bytes[:start] + replacement + code_bytes[end:]
        return code_bytes.decode("utf-8")

    def conservative_style_variants(self, code: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        compact_blank = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", code)
        candidates.append(("compact_blank_lines", compact_blank))
        spaces = re.sub(r"[ \t]+$", "", code, flags=re.MULTILINE)
        spaces = re.sub(r"\b(if|for|while|switch|catch)\s*\(", r"\1 (", spaces)
        candidates.append(("normalized_spacing", spaces))
        allman = re.sub(r"\)\s*\{", ")\n{", code)
        allman = re.sub(r"\b(else|finally)\s*\{", r"\1\n{", allman)
        candidates.append(("allman_braces", allman))
        knr = re.sub(r"\)\s*\n\s*\{", ") {", code)
        knr = re.sub(r"\b(else|finally)\s*\n\s*\{", r"\1 {", knr)
        candidates.append(("kr_braces", knr))
        result: list[tuple[str, str]] = []
        seen = {code}
        for name, candidate in candidates:
            if candidate not in seen and self.syntax_ok(candidate):
                seen.add(candidate)
                result.append((name, candidate))
        return result


def valid_identifier(name: str) -> bool:
    return bool(
        name
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and name not in JAVA_KEYWORDS
        and name not in PROTECTED_NAMES
        and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name)
    )


def _unique(values) -> list[str]:
    return list(dict.fromkeys(values))

