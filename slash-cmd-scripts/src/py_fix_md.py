#!/usr/bin/env python3
"""
Comprehensive Markdown/LaTeX Fixer

This module provides automated fixes for all issues detected by py_improve_md.py.
Created to support agentic AI execution with professional best practices.

Architecture:
    - Strategy Pattern for pluggable fixers
    - Single Responsibility Principle for each fixer class
    - Comprehensive type hints and error handling
    - Structured logging for debugging

Usage:
    python3 py_fix_md.py <file>                    # Fix all issues
    python3 py_fix_md.py --category list_formatting <file>
    python3 py_fix_md.py --fixer wrap_aligned_blocks <file>
    python3 py_fix_md.py --dry-run <file>          # Preview changes
    python3 py_fix_md.py --list-fixers             # Show available fixers
"""

import re
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import argparse


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

class FixCategory(Enum):
    """Categories of fixes matching detector categories."""
    MATH_FORMATTING = "math_formatting"
    LIST_FORMATTING = "list_formatting"
    BOLD_FORMATTING = "bold_formatting"
    SYNTAX = "syntax"
    ALL = "all"


# Compile regex patterns at module level for performance
PATTERNS = {
    'list_marker': re.compile(r'^\s*([-*+]|\d+\.)\s+'),
    'code_fence': re.compile(r'^\s*```'),
    'math_delimiter': re.compile(r'^\s*\$\$'),
    'heading': re.compile(r'^\s*#{1,6}\s'),
    'unwrapped_aligned_begin': re.compile(r'^\\begin\{aligned\}'),
    'unwrapped_aligned_end': re.compile(r'\\end\{aligned\}'),
    'single_dollar': re.compile(r'(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)'),
    'unescaped_underscore_brace': re.compile(r'(?<!\\)_\{'),
    'unescaped_underscore_alnum': re.compile(r'(?<!\\)_([a-zA-Z0-9])'),
    'empty_math_block': re.compile(r'\$\$\n\s*\n\s*\$\$'),
    'consecutive_dollars': re.compile(r'\$\$\n\$\$'),
    'bold_with_colon': re.compile(r'\*\*([^*]+)\*\*(:)'),
}

# Protected contexts where fixes should not be applied
PROTECTED_LINE_PREFIXES = {'```', '$$', '|', 'http://', 'https://'}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FixResult:
    """Result of applying a single fixer."""
    fixer_name: str
    fixes_applied: int
    success: bool
    error_message: Optional[str] = None

    def __str__(self) -> str:
        if not self.success:
            return f"{self.fixer_name}: ERROR - {self.error_message}"
        return f"{self.fixer_name}: {self.fixes_applied} fixes"


@dataclass
class FileFixResult:
    """Result of fixing an entire file."""
    filepath: Path
    modified: bool
    fix_results: List[FixResult] = field(default_factory=list)
    total_fixes: int = 0
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"{self.filepath.name}: ERROR - {self.error}"
        if not self.modified:
            return f"{self.filepath.name}: No changes needed"
        return f"{self.filepath.name}: {self.total_fixes} total fixes"


# ============================================================================
# EXCEPTIONS
# ============================================================================

class FixerError(Exception):
    """Base exception for fixer errors."""
    pass


class InvalidContentError(FixerError):
    """Raised when content is malformed or invalid."""
    pass


class ProtectedContextError(FixerError):
    """Raised when attempting to modify protected context."""
    pass


# ============================================================================
# BASE FIXER STRATEGY
# ============================================================================

class FixerStrategy(ABC):
    """
    Abstract base class for all fixer strategies.

    Implements the Strategy Pattern for pluggable fix operations.
    Each concrete fixer must implement the fix() method.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._fixes_count = 0

    @abstractmethod
    def fix(self, content: str) -> Tuple[str, int]:
        """
        Apply fixes to content.

        Args:
            content: Markdown content to fix

        Returns:
            Tuple of (fixed_content, number_of_fixes_applied)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the fixer's name."""
        pass

    @property
    @abstractmethod
    def category(self) -> FixCategory:
        """Return the fixer's category."""
        pass

    def is_protected_line(self, line: str) -> bool:
        """Check if line is in a protected context."""
        stripped = line.strip()
        return any(stripped.startswith(prefix) for prefix in PROTECTED_LINE_PREFIXES)


# ============================================================================
# CONCRETE FIXER IMPLEMENTATIONS
# ============================================================================

class WrapAlignedBlocksFixer(FixerStrategy):
    """Ensure all \\begin{aligned} blocks have proper $$ wrappers."""

    @property
    def name(self) -> str:
        return "wrap_aligned_blocks"

    @property
    def category(self) -> FixCategory:
        return FixCategory.MATH_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        lines = content.split('\n')
        result = []
        fixes = 0
        i = 0
        in_math_block = False  # Track if we're inside $$...$$
        in_code_fence = False  # Track if we're inside ```...```

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Update code fence state (```math or ``` blocks)
            if stripped.startswith('```'):
                in_code_fence = not in_code_fence
                result.append(line)
                i += 1
                continue

            # Update math block state (only outside code fences)
            if not in_code_fence and stripped.startswith('$$') and not stripped.startswith('$$$'):
                in_math_block = not in_math_block
                result.append(line)
                i += 1
                continue

            # Only wrap if NOT already in a math block or code fence
            if (PATTERNS['unwrapped_aligned_begin'].match(stripped) and
                not in_math_block and not in_code_fence):
                # Check if previous line has $$
                needs_opening = (i == 0 or result[-1].strip() != '$$')

                if needs_opening:
                    result.append('$$')
                    fixes += 1
                    self.logger.debug(f"Added opening $$ before line {i+1}")

                result.append(line)
                i += 1

                # Find matching \\end{aligned}
                while i < len(lines):
                    result.append(lines[i])
                    if PATTERNS['unwrapped_aligned_end'].search(lines[i]):
                        # Check if next line needs $$
                        needs_closing = (i + 1 >= len(lines) or
                                       lines[i + 1].strip() != '$$')
                        if needs_closing:
                            result.append('$$')
                            fixes += 1
                            self.logger.debug(f"Added closing $$ after line {i+1}")
                        i += 1
                        break
                    i += 1
            else:
                result.append(line)
                i += 1

        return '\n'.join(result), fixes


class EscapeUnderscoresFixer(FixerStrategy):
    """Escape underscores in all math expressions."""

    @property
    def name(self) -> str:
        return "escape_underscores"

    @property
    def category(self) -> FixCategory:
        return FixCategory.MATH_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        lines = content.split('\n')
        result = []
        fixes = 0
        in_math = False
        in_code = False

        for line in lines:
            # Track code blocks - DON'T escape in code!
            if PATTERNS['code_fence'].match(line):
                in_code = not in_code
                result.append(line)
                continue

            if in_code:
                result.append(line)
                continue

            # Track math blocks
            if PATTERNS['math_delimiter'].match(line):
                in_math = not in_math

            if in_math or '$$' in line:
                original = line
                # Escape underscores before braces and alphanumerics
                line = PATTERNS['unescaped_underscore_brace'].sub(r'\\_\{', line)
                line = PATTERNS['unescaped_underscore_alnum'].sub(r'\\_\1', line)

                if line != original:
                    fixes += 1
                    self.logger.debug(f"Escaped underscores in math: {original[:50]}...")

            result.append(line)

        return '\n'.join(result), fixes


class ListFormattingFixer(FixerStrategy):
    """Fix list formatting: add blank lines before lists, remove between items."""

    @property
    def name(self) -> str:
        return "list_formatting"

    @property
    def category(self) -> FixCategory:
        return FixCategory.LIST_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        lines = content.split('\n')
        result = []
        fixes = 0

        for i, line in enumerate(lines):
            is_list = PATTERNS['list_marker'].match(line)

            # Fix 1: Add blank line before list if needed
            if is_list and i > 0:
                prev = lines[i - 1].strip()
                if prev and prev not in ['$$', '```', '']:
                    prev_is_list = PATTERNS['list_marker'].match(lines[i-1])
                    if not prev_is_list:
                        # Check if there's already a blank line
                        if i < 2 or lines[i - 1]:
                            result.append('')
                            fixes += 1
                            self.logger.debug(f"Added blank line before list at line {i+1}")

            # Fix 2: Remove blank line between consecutive list items
            if line.strip() == '' and i > 0 and i < len(lines) - 1:
                prev_is_list = PATTERNS['list_marker'].match(lines[i - 1])
                next_is_list = PATTERNS['list_marker'].match(lines[i + 1])

                if prev_is_list and next_is_list:
                    # Check indentation - only remove if same level
                    prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    next_indent = len(lines[i+1]) - len(lines[i+1].lstrip())

                    if prev_indent == next_indent:
                        fixes += 1
                        self.logger.debug(f"Removed blank line between list items at line {i+1}")
                        continue  # Skip this blank line

            result.append(line)

        return '\n'.join(result), fixes


class SingleDollarToDoubleFixer(FixerStrategy):
    """Convert single $ to $$ for all inline math."""

    @property
    def name(self) -> str:
        return "single_to_double_dollar"

    @property
    def category(self) -> FixCategory:
        return FixCategory.MATH_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        lines = content.split('\n')
        result = []
        fixes = 0
        in_code = False

        for line in lines:
            if PATTERNS['code_fence'].match(line.strip()):
                in_code = not in_code
                result.append(line)
                continue

            if in_code or self.is_protected_line(line):
                result.append(line)
                continue

            original = line
            # Replace single $ with $$ (but not already-doubled ones)
            line = PATTERNS['single_dollar'].sub(r'$$\1$$', line)

            if line != original:
                fixes += 1
                self.logger.debug(f"Converted $ to $$ in line: {original[:50]}...")

            result.append(line)

        return '\n'.join(result), fixes


class EmptyBlockFixer(FixerStrategy):
    """Remove empty $$ blocks and consecutive delimiters."""

    @property
    def name(self) -> str:
        return "remove_empty_blocks"

    @property
    def category(self) -> FixCategory:
        return FixCategory.SYNTAX

    def fix(self, content: str) -> Tuple[str, int]:
        original = content

        # Remove empty $$ blocks
        content = PATTERNS['empty_math_block'].sub('$$', content)
        # Remove consecutive $$ markers
        content = PATTERNS['consecutive_dollars'].sub('$$', content)

        fixes = original.count('$$') - content.count('$$')

        if fixes > 0:
            self.logger.debug(f"Removed {fixes} empty/duplicate math blocks")

        return content, max(0, fixes)


class BoldFormattingFixer(FixerStrategy):
    """Remove excessive structural bold formatting."""

    @property
    def name(self) -> str:
        return "bold_formatting"

    @property
    def category(self) -> FixCategory:
        return FixCategory.BOLD_FORMATTING

    def fix(self, content: str) -> Tuple[str, int]:
        lines = content.split('\n')
        result = []
        fixes = 0
        in_code = False
        in_math = False

        for line in lines:
            # Track blocks
            if PATTERNS['code_fence'].match(line):
                in_code = not in_code
            if PATTERNS['math_delimiter'].match(line):
                in_math = not in_math

            if in_code or in_math or self.is_protected_line(line):
                result.append(line)
                continue

            original = line

            # Pattern 1: **Label**: text -> Label: text
            line = PATTERNS['bold_with_colon'].sub(r'\1\2', line)

            # Pattern 2: Bold after list marker
            line = re.sub(r'^(\s*(?:-|\d+\.|\*)\s+)\*\*([^*]+)\*\*', r'\1\2', line)

            # Pattern 3: Bold at start of line
            line = re.sub(r'^\*\*([^*]+)\*\*', r'\1', line)

            if line != original:
                fixes += 1
                self.logger.debug(f"Removed excessive bold: {original[:50]}...")

            result.append(line)

        return '\n'.join(result), fixes


# ============================================================================
# MAIN FIXER ORCHESTRATOR
# ============================================================================

class MarkdownFixer:
    """
    Main orchestrator for applying fixes to markdown files.

    Uses dependency injection to manage fixer strategies and provides
    flexible fix application based on categories or specific fixers.
    """

    # Registry of all available fixers
    ALL_FIXERS = [
        WrapAlignedBlocksFixer,
        # EscapeUnderscoresFixer,  # DISABLED: Incorrectly escapes underscores in LaTeX math
        ListFormattingFixer,
        # SingleDollarToDoubleFixer,  # DISABLED: Creates invalid inline display math
        EmptyBlockFixer,
        BoldFormattingFixer,
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fixers: List[FixerStrategy] = []
        self._initialize_all_fixers()

    def _initialize_all_fixers(self) -> None:
        """Initialize all fixer instances."""
        for fixer_class in self.ALL_FIXERS:
            self.fixers.append(fixer_class(self.logger))

    def get_fixers_by_category(self, category: FixCategory) -> List[FixerStrategy]:
        """Get all fixers for a specific category."""
        if category == FixCategory.ALL:
            return self.fixers
        return [f for f in self.fixers if f.category == category]

    def get_fixer_by_name(self, name: str) -> Optional[FixerStrategy]:
        """Get a specific fixer by name."""
        for fixer in self.fixers:
            if fixer.name == name:
                return fixer
        return None

    def fix_content(
        self,
        content: str,
        category: Optional[FixCategory] = None,
        fixer_name: Optional[str] = None
    ) -> Tuple[str, List[FixResult]]:
        """
        Apply fixes to content.

        Args:
            content: Markdown content to fix
            category: Optional category to filter fixers
            fixer_name: Optional specific fixer name to use

        Returns:
            Tuple of (fixed_content, list_of_fix_results)
        """
        results = []

        # Determine which fixers to use
        if fixer_name:
            fixer = self.get_fixer_by_name(fixer_name)
            if not fixer:
                raise ValueError(f"Unknown fixer: {fixer_name}")
            fixers_to_use = [fixer]
        elif category:
            fixers_to_use = self.get_fixers_by_category(category)
        else:
            fixers_to_use = self.fixers

        # Apply fixers in sequence
        current_content = content
        for fixer in fixers_to_use:
            try:
                self.logger.info(f"Applying {fixer.name}...")
                new_content, fixes_count = fixer.fix(current_content)

                result = FixResult(
                    fixer_name=fixer.name,
                    fixes_applied=fixes_count,
                    success=True
                )
                results.append(result)

                current_content = new_content

            except Exception as e:
                self.logger.error(f"Error in {fixer.name}: {e}")
                result = FixResult(
                    fixer_name=fixer.name,
                    fixes_applied=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)

        return current_content, results

    def fix_file(
        self,
        filepath: Path,
        category: Optional[FixCategory] = None,
        fixer_name: Optional[str] = None,
        dry_run: bool = False
    ) -> FileFixResult:
        """
        Fix a single markdown file.

        Args:
            filepath: Path to markdown file
            category: Optional category filter
            fixer_name: Optional specific fixer
            dry_run: If True, don't write changes

        Returns:
            FileFixResult with details of changes
        """
        try:
            content = filepath.read_text(encoding='utf-8')
            original_content = content

            new_content, fix_results = self.fix_content(content, category, fixer_name)

            modified = (new_content != original_content)
            total_fixes = sum(r.fixes_applied for r in fix_results if r.success)

            if modified and not dry_run:
                filepath.write_text(new_content, encoding='utf-8')
                self.logger.info(f"Wrote changes to {filepath}")

            return FileFixResult(
                filepath=filepath,
                modified=modified,
                fix_results=fix_results,
                total_fixes=total_fixes
            )

        except Exception as e:
            self.logger.error(f"Error processing {filepath}: {e}")
            return FileFixResult(
                filepath=filepath,
                modified=False,
                error=str(e)
            )


# ============================================================================
# CLI AND MAIN FUNCTION
# ============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def list_available_fixers() -> None:
    """Print all available fixers."""
    print("\nAvailable Fixers:")
    print("=" * 70)

    by_category: Dict[FixCategory, List[str]] = {}

    for fixer_class in MarkdownFixer.ALL_FIXERS:
        fixer = fixer_class()
        category = fixer.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(fixer.name)

    for category, fixers in sorted(by_category.items(), key=lambda x: x[0].value):
        print(f"\n{category.value.upper()}:")
        for fixer in sorted(fixers):
            print(f"  - {fixer}")

    print("\n" + "=" * 70)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Markdown/LaTeX Fixer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'files',
        nargs='*',
        type=Path,
        help='Markdown files to fix'
    )
    parser.add_argument(
        '--category',
        type=str,
        choices=[c.value for c in FixCategory],
        help='Fix only specific category'
    )
    parser.add_argument(
        '--fixer',
        type=str,
        help='Use only specific fixer'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--list-fixers',
        action='store_true',
        help='List all available fixers'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.list_fixers:
        list_available_fixers()
        return 0

    if not args.files:
        parser.print_help()
        return 1

    # Convert category string to enum
    category = FixCategory(args.category) if args.category else None

    # Create fixer instance
    fixer = MarkdownFixer()

    # Process files
    results = []
    for filepath in args.files:
        if not filepath.exists():
            logging.error(f"File not found: {filepath}")
            continue

        result = fixer.fix_file(filepath, category, args.fixer, args.dry_run)
        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Fix Summary")
    print("=" * 70)

    for result in results:
        print(f"\n{result}")
        if result.modified:
            for fix_result in result.fix_results:
                if fix_result.fixes_applied > 0:
                    print(f"  {fix_result}")

    total_files_modified = sum(1 for r in results if r.modified)
    total_fixes = sum(r.total_fixes for r in results)

    print("\n" + "=" * 70)
    status = "would be modified" if args.dry_run else "modified"
    print(f"✓ {total_files_modified}/{len(results)} files {status}")
    print(f"✓ {total_fixes} total fixes applied")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
