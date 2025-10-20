#!/usr/bin/env python3
"""
Comprehensive Markdown → MathML Fixer

This module converts all LaTeX/MathJax-based math in Markdown files
to clean, standards-compliant MathML suitable for MkDocs and browsers
that support MathML Core (Safari, Firefox, Chrome 109+).

Architecture:
    - Mirrors py_fix_md.py structure
    - Replaces LaTeX math fixers with MathML converters
    - Preserves safe list/bold/syntax fixers
    - Adds automatic detection of \frac, subscript, superscript, etc.

Usage:
    python3 py_latex_to_mathml.py <file>
    python3 py_latex_to_mathml.py --category math_formatting <file>
    python3 py_latex_to_mathml.py --fixer latex_to_mathml <file>
    python3 py_latex_to_mathml.py --dry-run <file>
"""

import re
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
import argparse

# External library for LaTeX → MathML conversion
try:
    from latex2mathml.converter import convert as latex_to_mathml_convert
except ImportError:
    print("ERROR: latex2mathml library not installed")
    print("Install with: pip install latex2mathml")
    sys.exit(1)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class FixCategory(Enum):
    MATH_FORMATTING = "math_formatting"
    LIST_FORMATTING = "list_formatting"
    BOLD_FORMATTING = "bold_formatting"
    SYNTAX = "syntax"
    ALL = "all"


PROTECTED_LINE_PREFIXES = {'```', '|', 'http://', 'https://'}

# Context markers for inline vs block math
TABLE_ROW_PATTERN = re.compile(r'^\s*\|.*\|.*$')
CODE_FENCE_PATTERN = re.compile(r'^```')


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FixResult:
    fixer_name: str
    fixes_applied: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class FileFixResult:
    filepath: Path
    modified: bool
    fix_results: List[FixResult] = field(default_factory=list)
    total_fixes: int = 0
    error: Optional[str] = None


# ============================================================================
# BASE STRATEGY
# ============================================================================

class FixerStrategy(ABC):
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fix(self, content: str) -> Tuple[str, int]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def category(self) -> FixCategory:
        pass

    def is_protected(self, line: str) -> bool:
        return any(line.strip().startswith(p) for p in PROTECTED_LINE_PREFIXES)


# ============================================================================
# FIXERS
# ============================================================================

class LaTeXToMathMLFixer(FixerStrategy):
    """Convert $$...$$ LaTeX math to <math display="block">...</math> MathML using latex2mathml library."""

    @property
    def name(self): return "latex_to_mathml"

    @property
    def category(self): return FixCategory.MATH_FORMATTING

    def _latex_to_mathml(self, latex_expr: str, display_mode: str = "block") -> str:
        """Convert LaTeX to MathML using external library."""
        try:
            # Use library for conversion
            mathml = latex_to_mathml_convert(latex_expr.strip())

            # Fix display attribute (library defaults to inline)
            if display_mode == "block":
                mathml = mathml.replace('display="inline"', 'display="block"')

            return mathml
        except Exception as e:
            self.logger.warning(f"Failed to convert LaTeX: {latex_expr[:50]}... Error: {e}")
            # Return original wrapped in math tags as fallback
            return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{display_mode}"><mtext>LaTeX conversion failed: {latex_expr}</mtext></math>'

    def fix(self, content: str) -> Tuple[str, int]:
        """Convert $$...$$ blocks to MathML."""
        pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)

        def replace_match(match):
            latex_content = match.group(1)
            return self._latex_to_mathml(latex_content, display_mode="block")

        new_content, count = re.subn(pattern, replace_match, content)
        self.logger.debug(f"Converted {count} $$...$$ blocks to MathML")
        return new_content, count


class InlineMathToMathMLFixer(FixerStrategy):
    """Convert $x$ inline LaTeX to <math display="inline">...</math> MathML."""

    @property
    def name(self): return "inline_math_to_mathml"

    @property
    def category(self): return FixCategory.MATH_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        """Convert $...$ inline math to MathML, avoiding code blocks."""
        lines = content.split('\n')
        in_code_block = False
        fixes = 0

        for i, line in enumerate(lines):
            if CODE_FENCE_PATTERN.match(line):
                in_code_block = not in_code_block
                continue

            if in_code_block or self.is_protected(line):
                continue

            # Match $...$ but not $$
            pattern = re.compile(r'(?<!\$)\$(?!\$)([^\$]+?)\$(?!\$)')

            def replace_inline(match):
                latex = match.group(1)
                mathml = LaTeXToMathMLFixer()._latex_to_mathml(latex, display_mode="inline")
                return mathml

            new_line, count = re.subn(pattern, replace_inline, line)
            if count > 0:
                lines[i] = new_line
                fixes += count

        return '\n'.join(lines), fixes


class ParenMathToMathMLFixer(FixerStrategy):
    """Convert \(...\) LaTeX to <math display="inline">...</math> MathML (for tables)."""

    @property
    def name(self): return "paren_math_to_mathml"

    @property
    def category(self): return FixCategory.MATH_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        """Convert \(...\) to inline MathML."""
        pattern = re.compile(r'\\\((.*?)\\\)', re.DOTALL)

        def replace_match(match):
            latex_content = match.group(1)
            return LaTeXToMathMLFixer()._latex_to_mathml(latex_content, display_mode="inline")

        new_content, count = re.subn(pattern, replace_match, content)
        self.logger.debug(f"Converted {count} \\(...\\) blocks to inline MathML")
        return new_content, count


class MathCodeFenceToMathMLFixer(FixerStrategy):
    """Convert ```math fenced blocks → MathML."""

    @property
    def name(self): return "math_code_fence_to_mathml"

    @property
    def category(self): return FixCategory.MATH_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        pattern = re.compile(r'```math\s*\n(.*?)\n```', re.DOTALL)
        matches = pattern.findall(content)
        fixes = 0
        for block in matches:
            latex_expr = block.strip()
            mathml = LaTeXToMathMLFixer()._latex_to_mathml(latex_expr)
            content = content.replace(f"```math\n{block}\n```", mathml)
            fixes += 1
        return content, fixes


class ListFormattingFixer(FixerStrategy):
    """Preserve identical list formatting logic as original."""
    @property
    def name(self): return "list_formatting"
    @property
    def category(self): return FixCategory.LIST_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        """Add blank line before lists, remove blank lines between list items."""
        lines = content.split("\n")
        result, fixes = [], 0
        prev_was_list = False

        for i, line in enumerate(lines):
            is_list = bool(re.match(r'^\s*[-*+]\s+', line))

            # Add blank line before list start
            if is_list and not prev_was_list and i > 0 and lines[i-1].strip() != '':
                result.append('')
                fixes += 1

            # Remove blank line between list items
            if is_list and prev_was_list and i > 0 and lines[i-1].strip() == '':
                result.pop()  # Remove the blank line we just added
                fixes += 1

            result.append(line)
            prev_was_list = is_list

        return "\n".join(result), fixes


class BoldFormattingFixer(FixerStrategy):
    """Remove excessive bold markers (same as original)."""
    @property
    def name(self): return "bold_formatting"
    @property
    def category(self): return FixCategory.BOLD_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        original = content
        content = re.sub(r'\*\*([^*]+)\*\*(:)', r'\1\2', content)
        content = re.sub(r'^\*\*([^*]+)\*\*', r'\1', content, flags=re.MULTILINE)
        return content, 1 if content != original else 0


class EmptyMathMLBlockFixer(FixerStrategy):
    """Remove stray or empty <math> tags."""
    @property
    def name(self): return "remove_empty_mathml"
    @property
    def category(self): return FixCategory.SYNTAX

    def fix(self, content: str) -> Tuple[str, int]:
        original = content
        content = re.sub(r'<math[^>]*>\s*</math>', '', content)
        return content, 1 if content != original else 0


# ============================================================================
# MAIN FIXER ORCHESTRATOR
# ============================================================================

class MarkdownFixer:
    ALL_FIXERS = [
        LaTeXToMathMLFixer,           # $$...$$ → block MathML
        InlineMathToMathMLFixer,      # $...$ → inline MathML
        ParenMathToMathMLFixer,       # \(...\) → inline MathML
        MathCodeFenceToMathMLFixer,   # ```math → MathML
        ListFormattingFixer,
        BoldFormattingFixer,
        EmptyMathMLBlockFixer,
    ]

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.fixers = [fixer(self.logger) for fixer in self.ALL_FIXERS]

    def get_fixers_by_category(self, category: FixCategory):
        return self.fixers if category == FixCategory.ALL else [
            f for f in self.fixers if f.category == category
        ]

    def get_fixer_by_name(self, name: str):
        for f in self.fixers:
            if f.name == name:
                return f
        return None

    def fix_content(self, content: str, category=None, fixer_name=None):
        results = []
        selected_fixers = []
        if fixer_name:
            fx = self.get_fixer_by_name(fixer_name)
            selected_fixers = [fx] if fx else []
        elif category:
            selected_fixers = self.get_fixers_by_category(category)
        else:
            selected_fixers = self.fixers

        for fixer in selected_fixers:
            new_content, count = fixer.fix(content)
            results.append(FixResult(fixer.name, count, True))
            content = new_content
        return content, results

    def fix_file(self, filepath: Path, category=None, fixer_name=None, dry_run=False):
        try:
            content = filepath.read_text(encoding="utf-8")
            new_content, fix_results = self.fix_content(content, category, fixer_name)
            modified = new_content != content
            total = sum(r.fixes_applied for r in fix_results)
            if modified and not dry_run:
                filepath.write_text(new_content, encoding="utf-8")
            return FileFixResult(filepath, modified, fix_results, total)
        except Exception as e:
            return FileFixResult(filepath, False, error=str(e))


# ============================================================================
# CLI
# ============================================================================

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def list_fixers():
    print("\nAvailable Fixers:")
    for cls in MarkdownFixer.ALL_FIXERS:
        fx = cls()
        print(f"  - {fx.name} ({fx.category.value})")


def main():
    parser = argparse.ArgumentParser(description="Markdown → MathML Fixer")
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--category", choices=[c.value for c in FixCategory])
    parser.add_argument("--fixer")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-fixers", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.list_fixers:
        list_fixers()
        return 0

    if not args.files:
        parser.print_help()
        return 1

    category = FixCategory(args.category) if args.category else None
    fixer = MarkdownFixer()

    for path in args.files:
        res = fixer.fix_file(path, category, args.fixer, args.dry_run)
        print(f"\n{path.name}: {'Modified' if res.modified else 'No changes'}")
        for r in res.fix_results:
            print(f"  {r.fixer_name}: {r.fixes_applied} fixes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
