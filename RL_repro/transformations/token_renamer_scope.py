#!/usr/bin/env python3

import hashlib
import keyword
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tree_sitter import Language, Parser


JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false",
    "null", "record", "sealed", "permits", "non-sealed", "var", "yield",
}


@dataclass(frozen=True)
class Declaration:
    name: str
    start: int
    end: int
    scope_start: int
    scope_end: int
    available_from: int
    scope_kind: str


class ScopeAwareTokenRenamer:

    SEMANTIC_MAP = {
        "result": "error", "count": "flag", "index": "offset",
        "length": "width", "value": "key", "data": "buffer",
        "input": "output", "source": "target", "start": "finish",
        "end": "begin", "min": "max_val", "max": "min_val",
        "size": "capacity", "name": "label", "type": "category",
        "status": "state", "current": "previous", "sum": "total",
        "total": "aggregate", "id": "identifier", "flag": "marker",
    }

    DECLARATION_PARENTS = {
        "variable_declarator",
        "formal_parameter",
        "spread_parameter",
        "catch_formal_parameter",
        "resource",
        "enhanced_for_statement",
    }

    def __init__(self, parser_path: str):
        self._lang = Language(parser_path, "java")
        self._parser = Parser()
        self._parser.set_language(self._lang)

    def close(self):
        pass

    def extract_variable_names(self, code_string: str) -> List[str]:
        try:
            root = self._parse(code_string)
            declarations = self._collect_declarations(root, code_string)
            if not declarations:
                return []
            counts = {name: 0 for name in {decl.name for decl in declarations}}
            for node in self._identifier_nodes(root):
                name = self._text(node, code_string)
                if name in counts and self._resolve(node, name, declarations):
                    counts[name] += 1
            return sorted(counts, key=lambda name: (-counts[name], name))
        except Exception:
            return []

    def rename_variable(
        self, code_string: str, target_var: str, strategy: int
    ) -> Tuple[str, bool]:
        if strategy not in (69, 70, 71, 72):
            return code_string, False
        try:
            root = self._parse(code_string)
            if root.has_error:
                return code_string, False
            declarations = self._collect_declarations(root, code_string)
            targets = [decl for decl in declarations if decl.name == target_var]
            if not targets:
                return code_string, False

            positions = set()
            for decl in targets:
                positions.add((decl.start, decl.end))
            for node in self._identifier_nodes(root):
                if self._text(node, code_string) != target_var:
                    continue
                resolved = self._resolve(node, target_var, declarations)
                if resolved in targets:
                    positions.add((node.start_byte, node.end_byte))

            if not positions:
                return code_string, False
            reserved = self._all_identifiers(root, code_string) | JAVA_KEYWORDS
            replacement = self._unique_replacement(
                target_var, strategy, code_string, reserved
            )
            if replacement == target_var:
                return code_string, False

            data = code_string.encode("utf-8")
            encoded = replacement.encode("utf-8")
            for start, end in sorted(positions, reverse=True):
                data = data[:start] + encoded + data[end:]
            candidate = data.decode("utf-8")
            if not self.validate_structure(candidate):
                return code_string, False
            return candidate, True
        except Exception:
            return code_string, False

    def validate_structure(self, code_string: str) -> bool:
        try:
            root = self._parse(code_string)
            if root.has_error or root.type != "program":
                return False
            declarations = self._collect_declarations(root, code_string)
            for index, left in enumerate(declarations):
                for right in declarations[index + 1 :]:
                    if left.name != right.name:
                        continue
                    if self._declarations_conflict(left, right):
                        return False
            return True
        except Exception:
            return False

    def _parse(self, code_string: str):
        return self._parser.parse(code_string.encode("utf-8")).root_node

    @staticmethod
    def _text(node, code_string: str) -> str:
        return code_string.encode("utf-8")[node.start_byte : node.end_byte].decode(
            "utf-8"
        )

    def _identifier_nodes(self, root):
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "identifier":
                yield node
            stack.extend(reversed(node.children))

    def _all_identifiers(self, root, code_string: str):
        return {self._text(node, code_string) for node in self._identifier_nodes(root)}

    def _collect_declarations(self, root, code_string: str) -> List[Declaration]:
        declarations = []
        for node in self._identifier_nodes(root):
            parent = node.parent
            if parent is None or parent.type not in self.DECLARATION_PARENTS:
                continue
            name_node = parent.child_by_field_name("name")
            if name_node is None or not self._same_span(node, name_node):
                continue
            if parent.type == "variable_declarator" and self._has_ancestor(
                parent, "field_declaration"
            ):
                continue
            scope, available_from, kind = self._scope_for_declaration(parent)
            if scope is None:
                continue
            declarations.append(
                Declaration(
                    name=self._text(node, code_string),
                    start=node.start_byte,
                    end=node.end_byte,
                    scope_start=scope.start_byte,
                    scope_end=scope.end_byte,
                    available_from=available_from,
                    scope_kind=kind,
                )
            )
        return declarations

    def _scope_for_declaration(self, declaration_node):
        parent_type = declaration_node.type
        if parent_type == "enhanced_for_statement":
            return declaration_node, declaration_node.start_byte, "loop"
        if parent_type in {"formal_parameter", "spread_parameter"}:
            owner = self._nearest_ancestor(
                declaration_node,
                {"method_declaration", "constructor_declaration", "lambda_expression"},
            )
            if owner is None:
                return None, 0, ""
            body = owner.child_by_field_name("body") or owner
            return body, body.start_byte, "parameter"
        if parent_type == "catch_formal_parameter":
            owner = self._nearest_ancestor(declaration_node, {"catch_clause"})
            return (owner, owner.start_byte, "catch") if owner else (None, 0, "")

        loop = self._nearest_ancestor(
            declaration_node, {"for_statement", "enhanced_for_statement"}
        )
        block = self._nearest_ancestor(declaration_node, {"block"})
        if loop is not None and (
            block is None
            or loop.start_byte >= block.start_byte
            and loop.end_byte <= block.end_byte
            and declaration_node.start_byte < (
                (loop.child_by_field_name("body") or loop).start_byte
            )
        ):
            return loop, declaration_node.start_byte, "loop"
        if block is not None:
            return block, declaration_node.start_byte, "local"
        return None, 0, ""

    def _resolve(
        self, node, name: str, declarations: List[Declaration]
    ) -> Optional[Declaration]:
        position = node.start_byte
        if not self._is_reference_candidate(node):
            return None
        candidates = []
        for decl in declarations:
            if decl.name != name:
                continue
            if position == decl.start:
                return decl
            if not (decl.scope_start <= position < decl.scope_end):
                continue
            if decl.scope_kind != "parameter" and position < decl.available_from:
                continue
            candidates.append(decl)
        if not candidates:
            return None
        candidates.sort(
            key=lambda decl: (
                decl.scope_end - decl.scope_start,
                -decl.available_from,
            )
        )
        return candidates[0]

    def _is_reference_candidate(self, node):
        parent = node.parent
        if parent is None:
            return False
        if parent.type in {
            "annotation",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "method_declaration",
            "constructor_declaration",
            "package_declaration",
            "import_declaration",
            "scoped_identifier",
            "labeled_statement",
            "break_statement",
            "continue_statement",
        }:
            return False
        if parent.type == "method_invocation":
            method_name = parent.child_by_field_name("name")
            if method_name is not None and self._same_span(node, method_name):
                return False
        if parent.type == "field_access":
            field_name = parent.child_by_field_name("field")
            if field_name is not None and self._same_span(node, field_name):
                return False
        return True

    @staticmethod
    def _declarations_conflict(left: Declaration, right: Declaration) -> bool:
        left_contains_right = (
            left.scope_start <= right.start < left.scope_end
            and right.start >= left.available_from
        )
        right_contains_left = (
            right.scope_start <= left.start < right.scope_end
            and left.start >= right.available_from
        )
        return left_contains_right or right_contains_left

    def _unique_replacement(self, target, strategy, code_string, reserved):
        if strategy == 69:
            declared = sorted(set(self.extract_variable_names(code_string)))
            base = f"VAR_{declared.index(target) if target in declared else 0}"
        elif strategy == 70:
            base = "v_" + self._digest(code_string, target, strategy)[:8]
        elif strategy == 71:
            base = self.SEMANTIC_MAP.get(
                target.lower(), "v_" + self._digest(code_string, target, strategy)[:8]
            )
        else:
            base = self._case_perturb(target)
        base = re.sub(r"[^A-Za-z0-9_$]", "_", base)
        if not base or not re.match(r"[A-Za-z_$]", base):
            base = "v_" + base
        candidate = base
        suffix = 1
        while candidate in reserved or candidate in JAVA_KEYWORDS:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _digest(code_string, target, strategy):
        payload = f"{strategy}:{target}:{code_string}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _case_perturb(name):
        if "_" in name:
            parts = [part for part in name.split("_") if part]
            candidate = parts[0].lower() + "".join(part.title() for part in parts[1:])
        elif any(char.isupper() for char in name[1:]):
            candidate = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        else:
            candidate = name.upper()
        return candidate if candidate != name else name + "_x"

    @staticmethod
    def _same_span(left, right):
        return left.start_byte == right.start_byte and left.end_byte == right.end_byte

    @staticmethod
    def _has_ancestor(node, type_name):
        current = node.parent
        while current is not None:
            if current.type == type_name:
                return True
            if current.type in {"class_body", "method_declaration", "block"}:
                if type_name == "field_declaration" and current.type == "class_body":
                    return False
            current = current.parent
        return False

    @staticmethod
    def _nearest_ancestor(node, types):
        current = node.parent
        while current is not None:
            if current.type in types:
                return current
            current = current.parent
        return None
