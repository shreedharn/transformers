#!/usr/bin/env python3
"""
Markdown and MathJax Formatting Detector

This module is created to support agentic AI execution.
Provides comprehensive detection of markdown and MathJax formatting issues,
ensuring proper separation between prose and mathematical content.

Supports command-line usage with selective detector execution, allowing users
to run specific detectors or all detectors based on command-line arguments.
Includes comprehensive help and logging capabilities.
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple


# Module-level constants
DEFAULT_CONTEXT_LINES = 3
MIN_INDENTATION_FOR_CODE_BLOCK = 4
DEFAULT_FILE_ENCODING = 'utf-8'
LOGGER_NAME = 'markdown_math_detector'

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1


class DetectorCategory(Enum):
    """Categories for grouping detectors by functionality."""
    INLINE_MATH = "Inline Math Issues"
    DISPLAY_MATH = "Display Math Issues"
    LIST_FORMATTING = "List Formatting Issues"
    ALIGNMENT = "Math Alignment Issues"
    STRUCTURAL = "Structural Issues"
    SYNTAX = "Syntax Issues"


class IssueType(Enum):
    """Categories of markdown and MathJax formatting issues."""
    LIST_WITH_DISPLAY_MATH = "List-marker lines contain display math (forbidden)"
    HEADING_WITH_MATH = "Heading lines contain math (forbidden)"
    TABLE_WITH_DISPLAY_MATH = "Table rows contain display math (forbidden)"
    INLINE_MATH_IN_PARAGRAPHS = "Paragraphs with inline math (promote to block)"
    DISPLAY_MATH_WITH_LIST_MARKER = "Display math on same line as list marker"
    OVER_INDENTED_DISPLAY_MATH = "Display math indented 4+ spaces (code block)"
    ADJACENT_MATH_BLOCKS = "Adjacent math blocks (check consolidation)"
    MATH_TOKENS_IN_PROSE = "Math tokens leaking into prose"
    BOLD_MARKDOWN_IN_MATH = "Bold Markdown used inside math"
    BACKSLASHES_IN_ALIGNED = "Backslashes in aligned (standardize to newline)"
    MATH_MISSING_ALIGNED = "Display math blocks missing aligned"
    MATH_IN_BLOCKQUOTES = "Math in blockquotes"
    LIST_MISSING_BLANK_LINE = "List-adjacent math missing blank line"
    MIXED_LIST_ITEMS = "Mixed list items with math"
    INLINE_MATH_IN_LISTS = "List items contain inline math"
    INLINE_MATH_IN_HEADINGS = "Heading items contain inline math"
    UNPAIRED_MATH_DELIMITERS = "Odd number of $$ (unpaired)"
    MATH_WITH_TEXT_SAME_LINE = "$$ with non-empty text on same line"
    STRAY_BRACES_WITH_MATH = "Stray braces next to $$"
    END_ALIGNED_WITH_TRAILING_TEXT = "end{aligned} followed by $$ then text"
    MISMATCHED_ALIGNED_BLOCKS = "Mismatched begin/end aligned counts"
    END_BEFORE_BEGIN_ALIGNED = "end{aligned} before any begin{aligned}"
    BLANK_LINES_BETWEEN_LIST_ITEMS = ("Blank lines between list items "
                                      "(should only be before list start)")
    ESCAPED_UNDERSCORES_IN_CODE = "Escaped underscores in code blocks (breaks syntax)"
    INLINE_DISPLAY_MATH_IN_PROSE = ("Display math delimiters used inline in prose "
                                     "(should be blocks)")
    CONSECUTIVE_BOLD_WITHOUT_SPACING = "Consecutive bold text lines without proper spacing"
    TEXT_TO_LIST_MISSING_BLANK_LINE = "Missing blank line before list item after text"
    MATH_CODE_FENCE_BLOCKS = "```math code fence blocks (convert to MathJax)"
    TEXTSTYLE_MISSING_OPENING_DELIMITER = "{\textstyle without opening $$ delimiter"


@dataclass
class DetectionResult:
    """Represents a detected markdown/MathJax formatting issue."""
    line_number: int
    issue_type: IssueType
    content: str
    context_lines: List[str] = None
    description: str = ""

    def __post_init__(self) -> None:
        """Generate description if not provided."""
        if not self.description:
            self.description = self.issue_type.value
        if self.context_lines is None:
            self.context_lines = []


class LineContext(NamedTuple):
    """Context information for a line during processing."""
    number: int
    content: str
    is_blank: bool
    is_header: bool
    is_list_item: bool
    is_table_row: bool
    is_code_fence: bool
    is_blockquote: bool


# Compiled regex patterns for performance
PATTERNS = {
    # Basic structure patterns
    'list_marker': re.compile(r'^\s*([-*+]|[0-9]+\.)\s'),
    'header': re.compile(r'^\s*#{1,6}\s'),
    'table_row': re.compile(r'^\s*\|.*\|.*$'),
    'blockquote': re.compile(r'^\s*>'),
    'code_fence': re.compile(r'^\s*`{3,}'),
    'blank_line': re.compile(r'^\s*$'),

    # Math patterns
    'display_math': re.compile(r'\$\$'),
    'inline_math_paren': re.compile(r'\\\([^)]*\\\)'),
    'inline_math_dollar': re.compile(r'\$[^$]+\$'),
    'begin_aligned': re.compile(r'\\begin\{aligned\}'),
    'end_aligned': re.compile(r'\\end\{aligned\}'),

    # Specific issue patterns
    'list_with_display_math': re.compile(
        r'^\s*([-*+]|[0-9]+\.)\s.*(\$\$|\\begin\{aligned\})'
    ),
    'heading_with_math': re.compile(
        r'^\s*#{1,6}\s.*(\$\$|\\begin\{aligned\}|\\\(|\\\))'
    ),
    'table_with_display_math': re.compile(r'^\s*\|.*\$\$.*\|.*$'),
    'display_math_with_list': re.compile(r'^\s*([-*+]|[0-9]+\.)\s*\$\$'),
    'over_indented_math': re.compile(rf'^\s{{{MIN_INDENTATION_FOR_CODE_BLOCK},}}\$\$'),
    'math_tokens_in_prose': re.compile(
        r'(^[^$`].*)\b(O\([^)]+\)|\\alpha|\\beta|\\gamma|\\sigma|'
        r'\\mathbb\{|\\mathcal\{|\\frac\{|\\sum\b|\\prod\b|\\to\b|'
        r'\\rightarrow\b|\\mid\b|\\lvert|\\rvert)\b'
    ),
    'bold_in_math': re.compile(r'\$\$.*\*\*|\*\*.*\$\$'),
    'backslashes_in_aligned': re.compile(r'\\begin\{aligned\}.*\\\\|^.*\\\\\s*$'),
    'math_in_blockquotes': re.compile(r'^\s*>\s.*(\$\$|\\begin\{aligned\}|\\\(|\\\))'),
    'math_same_line': re.compile(r'^\s*\$\$.*\S|.*\S.*\$\$\s*$'),
    'stray_braces': re.compile(r'\$\$\s*\}|\{\s*\$\$'),
    'end_aligned_trailing': re.compile(r'\\end\{aligned\}.*\$\$.*\S'),
    'code_fence_start': re.compile(r'^\s*```'),
    'escaped_underscore_in_code': re.compile(r'\\_'),
    'inline_display_math_in_prose': re.compile(r'^(?!\s*\$\$\s*$).*\$\$[^$]+\$\$.*\S'),
    'consecutive_bold_lines': re.compile(r'^\*\*[^*]+\*\*:?\s*$'),
    'math_code_fence': re.compile(r'^\s*```math\s*$'),
}


class MarkdownMathDetector:
    """Main class for detecting markdown and MathJax formatting issues."""

    def __init__(self, filepath: Path) -> None:
        """
        Initialize detector with the markdown file to analyze.

        Args:
            filepath: Path to the markdown file to analyze
        """
        self.filepath = filepath
        self.lines: List[str] = []
        self.results: List[DetectionResult] = []

    def _load_file(self) -> None:
        """Load and prepare file content for analysis."""
        try:
            with open(self.filepath, 'r', encoding=DEFAULT_FILE_ENCODING) as file:
                self.lines = file.readlines()
        except (IOError, OSError) as e:
            raise FileNotFoundError(f"Cannot read file {self.filepath}: {e}") from e

    def _create_line_context(self, line_num: int, content: str) -> LineContext:
        """
        Create context information for a line.

        Args:
            line_num: Line number (1-based)
            content: Line content

        Returns:
            LineContext with analyzed line information
        """
        return LineContext(
            number=line_num,
            content=content,
            is_blank=bool(PATTERNS['blank_line'].match(content)),
            is_header=bool(PATTERNS['header'].match(content)),
            is_list_item=bool(PATTERNS['list_marker'].match(content)),
            is_table_row=bool(PATTERNS['table_row'].match(content)),
            is_code_fence=bool(PATTERNS['code_fence'].match(content)),
            is_blockquote=bool(PATTERNS['blockquote'].match(content))
        )

    def _get_context_lines(self, line_num: int,
                          context: int = DEFAULT_CONTEXT_LINES) -> List[str]:
        """Get context lines around a specific line number."""
        start = max(0, line_num - context - 1)
        end = min(len(self.lines), line_num + context)
        return [f"{i+1:4d}: {line.rstrip()}"
                for i, line in enumerate(self.lines[start:end], start)]

    def detect_list_with_display_math(self) -> List[DetectionResult]:
        """Detect list-marker lines that contain display math (forbidden)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['list_with_display_math'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.LIST_WITH_DISPLAY_MATH,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_heading_with_math(self) -> List[DetectionResult]:
        """Detect heading lines that contain math (forbidden in headings)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['heading_with_math'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.HEADING_WITH_MATH,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_table_with_display_math(self) -> List[DetectionResult]:
        """Detect table rows containing display math (forbidden; inline only if needed)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['table_with_display_math'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.TABLE_WITH_DISPLAY_MATH,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_inline_math_in_paragraphs(self) -> List[DetectionResult]:
        """Detect paragraphs with inline math delimiters (promote to block)."""
        issues = []
        in_code_block = False

        for i, line in enumerate(self.lines):
            if PATTERNS['code_fence'].match(line):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            # Check for inline math patterns
            if (PATTERNS['inline_math_paren'].search(line) or
                PATTERNS['inline_math_dollar'].search(line)):

                # Exclude if it's a list, header, or blockquote
                if not (PATTERNS['list_marker'].match(line) or
                       PATTERNS['header'].match(line) or
                       PATTERNS['blockquote'].match(line)):
                    issues.append(DetectionResult(
                        line_number=i + 1,
                        issue_type=IssueType.INLINE_MATH_IN_PARAGRAPHS,
                        content=line.strip(),
                        context_lines=self._get_context_lines(i + 1)
                    ))
        return issues

    def detect_display_math_with_list_marker(self) -> List[DetectionResult]:
        """Detect display math on the same line as list marker (hard fail)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['display_math_with_list'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.DISPLAY_MATH_WITH_LIST_MARKER,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_over_indented_display_math(self) -> List[DetectionResult]:
        """Detect display math indented 4+ spaces (likely rendered as code block)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['over_indented_math'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.OVER_INDENTED_DISPLAY_MATH,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_adjacent_math_blocks(self) -> List[DetectionResult]:
        """Detect adjacent $$ blocks for manual consolidation check."""
        issues = []
        for i, line in enumerate(self.lines):
            if re.match(r'^\s*\$\$\s*$', line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.ADJACENT_MATH_BLOCKS,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_math_tokens_in_prose(self) -> List[DetectionResult]:
        """Detect math tokens leaking into prose (broad net; review matches)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['math_tokens_in_prose'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.MATH_TOKENS_IN_PROSE,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_bold_markdown_in_math(self) -> List[DetectionResult]:
        """Detect bold Markdown used inside math (prefer \\mathbf/\\boldsymbol)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['bold_in_math'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.BOLD_MARKDOWN_IN_MATH,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_backslashes_in_aligned(self) -> List[DetectionResult]:
        """Detect backslashes in aligned (standardize to \\newline for renderer)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['backslashes_in_aligned'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.BACKSLASHES_IN_ALIGNED,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_math_missing_aligned(self) -> List[DetectionResult]:
        """Detect display math blocks possibly missing \\begin{aligned} nearby."""
        issues = []
        open_block = False
        start_line = 0
        has_aligned = False

        for i, line in enumerate(self.lines):
            if '$$' in line:
                if not open_block:
                    start_line = i + 1
                    open_block = True
                    has_aligned = False
                else:
                    if not has_aligned:
                        issues.append(DetectionResult(
                            line_number=start_line,
                            issue_type=IssueType.MATH_MISSING_ALIGNED,
                            content=f"Block starting near line {start_line} may lack aligned",
                            context_lines=self._get_context_lines(start_line)
                        ))
                    open_block = False

            if open_block and PATTERNS['begin_aligned'].search(line) and i - start_line + 1 <= 3:
                has_aligned = True

        return issues

    def detect_math_in_blockquotes(self) -> List[DetectionResult]:
        """Detect math in blockquotes (usually unwanted)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['math_in_blockquotes'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.MATH_IN_BLOCKQUOTES,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_list_missing_blank_line(self) -> List[DetectionResult]:
        """Detect list-adjacent math blocks missing the blank spacer line."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['list_marker'].match(line) and i + 1 < len(self.lines):
                next_line = self.lines[i + 1]
                if re.match(r'^\s*\$\$', next_line):
                    issues.append(DetectionResult(
                        line_number=i + 1,
                        issue_type=IssueType.LIST_MISSING_BLANK_LINE,
                        content=f"Missing blank line before $$ near line {i + 1}",
                        context_lines=self._get_context_lines(i + 1)
                    ))
        return issues

    def detect_inline_math_in_lists(self) -> List[DetectionResult]:
        """Detect list items that contain inline math (forbidden in bullets)."""
        issues = []
        in_code_block = False

        for i, line in enumerate(self.lines):
            if PATTERNS['code_fence'].match(line):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            if (PATTERNS['list_marker'].match(line) and
                (PATTERNS['inline_math_dollar'].search(line) or
                 PATTERNS['inline_math_paren'].search(line))):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.INLINE_MATH_IN_LISTS,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_inline_math_in_headings(self) -> List[DetectionResult]:
        """Detect heading items that contain inline math (forbidden)."""
        issues = []
        in_code_block = False

        for i, line in enumerate(self.lines):
            if PATTERNS['code_fence'].match(line):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            if (PATTERNS['header'].match(line) and
                (PATTERNS['inline_math_dollar'].search(line) or
                 PATTERNS['inline_math_paren'].search(line))):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.INLINE_MATH_IN_HEADINGS,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_unpaired_math_delimiters(self) -> List[DetectionResult]:
        """Detect odd number of $$ (unpaired)."""
        content = ''.join(self.lines)
        dollar_count = content.count('$$')

        if dollar_count % 2 != 0:
            return [DetectionResult(
                line_number=1,
                issue_type=IssueType.UNPAIRED_MATH_DELIMITERS,
                content=f"Unpaired $$ delimiters: found {dollar_count} (should be even)",
                context_lines=[]
            )]
        return []

    def detect_math_with_text_same_line(self) -> List[DetectionResult]:
        """Detect $$ with non-empty text on the same line."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['math_same_line'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.MATH_WITH_TEXT_SAME_LINE,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_stray_braces_with_math(self) -> List[DetectionResult]:
        """Detect stray brace next to $$ ($$} or {$$)."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['stray_braces'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.STRAY_BRACES_WITH_MATH,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_end_aligned_with_trailing_text(self) -> List[DetectionResult]:
        """Detect \\end{aligned} followed by $$ then trailing text."""
        issues = []
        for i, line in enumerate(self.lines):
            if PATTERNS['end_aligned_trailing'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.END_ALIGNED_WITH_TRAILING_TEXT,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_mismatched_aligned_blocks(self) -> List[DetectionResult]:
        """Detect mismatched begin/end aligned counts."""
        content = ''.join(self.lines)
        begin_count = len(PATTERNS['begin_aligned'].findall(content))
        end_count = len(PATTERNS['end_aligned'].findall(content))

        if begin_count != end_count:
            return [DetectionResult(
                line_number=1,
                issue_type=IssueType.MISMATCHED_ALIGNED_BLOCKS,
                content=f"Mismatched aligned blocks: {begin_count} begin, {end_count} end",
                context_lines=[]
            )]
        return []

    def detect_end_before_begin_aligned(self) -> List[DetectionResult]:
        """Detect \\end{aligned} before any \\begin{aligned}."""
        issues = []
        seen_begin = False

        for i, line in enumerate(self.lines):
            if PATTERNS['end_aligned'].search(line) and not seen_begin:
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.END_BEFORE_BEGIN_ALIGNED,
                    content="end before any begin",
                    context_lines=self._get_context_lines(i + 1)
                ))
            if PATTERNS['begin_aligned'].search(line):
                seen_begin = True

        return issues

    def detect_blank_lines_between_list_items(self) -> List[DetectionResult]:
        """Detect blank lines between list items (should only be before list start)."""
        issues = []
        in_list = False
        prev_was_list_item = False

        for i, line in enumerate(self.lines):
            is_list_item = bool(PATTERNS['list_marker'].match(line))
            is_blank = bool(PATTERNS['blank_line'].match(line))

            # Start of a list
            if is_list_item and not in_list:
                in_list = True
                prev_was_list_item = True
                continue

            # Inside a list
            if in_list:
                if is_blank and prev_was_list_item:
                    # Check if next line is another list item
                    if i + 1 < len(self.lines) and PATTERNS['list_marker'].match(self.lines[i + 1]):
                        issues.append(DetectionResult(
                            line_number=i + 1,
                            issue_type=IssueType.BLANK_LINES_BETWEEN_LIST_ITEMS,
                            content="Unnecessary blank line between list items",
                            context_lines=self._get_context_lines(i + 1)
                        ))
                elif is_list_item:
                    prev_was_list_item = True
                    continue
                elif not is_blank:
                    # End of list (non-blank, non-list line)
                    in_list = False
                    prev_was_list_item = False

            prev_was_list_item = is_list_item

        return issues

    def detect_escaped_underscores_in_code(self) -> List[DetectionResult]:
        """Detect escaped underscores in code blocks (breaks syntax)."""
        issues = []
        in_code_block = False

        for i, line in enumerate(self.lines):
            if PATTERNS['code_fence_start'].match(line):
                in_code_block = not in_code_block
                continue

            if in_code_block and PATTERNS['escaped_underscore_in_code'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.ESCAPED_UNDERSCORES_IN_CODE,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))

        return issues

    def detect_inline_display_math_in_prose(self) -> List[DetectionResult]:
        """Detect display math delimiters ($$) used inline within prose sentences."""
        issues = []
        in_code_block = False

        for i, line in enumerate(self.lines):
            if PATTERNS['code_fence_start'].match(line):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            # Skip actual display math blocks (lines with only $$)
            if re.match(r'^\s*\$\$\s*$', line):
                continue

            # Detect $$ used inline within prose
            if PATTERNS['inline_display_math_in_prose'].search(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.INLINE_DISPLAY_MATH_IN_PROSE,
                    content=line.strip(),
                    context_lines=self._get_context_lines(i + 1)
                ))

        return issues

    def detect_consecutive_bold_without_spacing(self) -> List[DetectionResult]:
        """Detect consecutive bold text lines without proper spacing."""
        issues = []
        prev_was_bold = False

        for i, line in enumerate(self.lines):
            is_bold_line = bool(PATTERNS['consecutive_bold_lines'].match(line))
            is_blank = bool(PATTERNS['blank_line'].match(line))

            if is_bold_line and prev_was_bold:
                # Check if there was no blank line between consecutive bold lines
                if i > 0 and not PATTERNS['blank_line'].match(self.lines[i - 1]):
                    issues.append(DetectionResult(
                        line_number=i + 1,
                        issue_type=IssueType.CONSECUTIVE_BOLD_WITHOUT_SPACING,
                        content="Missing blank line between bold text lines",
                        context_lines=self._get_context_lines(i + 1)
                    ))

            prev_was_bold = is_bold_line and not is_blank

        return issues

    def detect_text_to_list_missing_blank_line(self) -> List[DetectionResult]:
        """Detect missing blank line before list items that follow descriptive text."""
        issues = []
        in_code_block = False

        for i, line in enumerate(self.lines):
            # Track code blocks to skip them
            if PATTERNS['code_fence'].match(line):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            # Skip if not a list item
            if not PATTERNS['list_marker'].match(line):
                continue

            # Check previous line exists
            if i == 0:
                continue

            prev_line = self.lines[i - 1].strip()

            # If previous line is blank, this is properly formatted
            if not prev_line:
                continue

            # If previous line is also a list item, this is continuation (allowed)
            if PATTERNS['list_marker'].match(self.lines[i - 1]):
                continue

            # If previous line is a heading, section break, or code, skip
            if (PATTERNS['header'].match(self.lines[i - 1]) or
                prev_line.startswith('---') or
                PATTERNS['code_fence'].match(self.lines[i - 1])):
                continue

            # If we reach here, we have descriptive text directly followed by list item
            # This violates the blank line requirement
            if (prev_line.endswith('.') or prev_line.endswith(':') or
                prev_line.endswith(')') or prev_line.endswith('.')):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.TEXT_TO_LIST_MISSING_BLANK_LINE,
                    content=f"List item needs blank line after text: '{prev_line[:50]}...'",
                    context_lines=self._get_context_lines(i + 1)
                ))
        return issues

    def detect_math_code_fence_blocks(self) -> List[DetectionResult]:
        """Detect ```math code fence blocks that should be converted to MathJax."""
        issues = []
        in_math_fence = False
        start_line = 0

        for i, line in enumerate(self.lines):
            if PATTERNS['math_code_fence'].match(line):
                if not in_math_fence:
                    # Found opening ```math fence
                    in_math_fence = True
                    start_line = i + 1
                    issues.append(DetectionResult(
                        line_number=start_line,
                        issue_type=IssueType.MATH_CODE_FENCE_BLOCKS,
                        content="```math block should be converted to MathJax $$\\begin{aligned}",
                        context_lines=self._get_context_lines(start_line)
                    ))
                else:
                    # Found closing ``` fence
                    in_math_fence = False

        return issues

    def detect_textstyle_missing_opening_delimiter(self) -> List[DetectionResult]:
        """
        Detect {\textstyle blocks that appear after a closing $$ without an opening $$.

        Pattern detected:
        }            <-- Closing brace from previous block
        $$           <-- Closing delimiter (end of math block)
        {\textstyle  <-- ERROR: Missing opening $$ before this
        \begin{aligned}
        ...

        Correct pattern should be:
        }            <-- Closing brace
        $$           <-- Closing delimiter

        $$           <-- Opening delimiter for new block
        {\textstyle  <-- OK: Inside math block
        """
        issues = []

        for i in range(len(self.lines) - 1):
            line = self.lines[i].strip()

            # Check for closing $$ delimiter (preceded by })
            if line == '$$':
                # Look backward to find previous non-empty line
                prev_idx = i - 1
                while prev_idx >= 0 and not self.lines[prev_idx].strip():
                    prev_idx -= 1

                # If previous non-empty line is }, this $$ is CLOSING a block
                if prev_idx >= 0 and self.lines[prev_idx].strip() == '}':
                    # Look ahead to find the next non-empty line
                    j = i + 1
                    while j < len(self.lines) and not self.lines[j].strip():
                        j += 1

                    # Check if next non-empty line starts with {\textstyle
                    if j < len(self.lines):
                        next_line = self.lines[j].strip()
                        if next_line.startswith('{\textstyle') or next_line.startswith('{\\textstyle'):
                            issues.append(DetectionResult(
                                line_number=j + 1,
                                issue_type=IssueType.TEXTSTYLE_MISSING_OPENING_DELIMITER,
                                content="{\\textstyle appears after closing $$ without opening $$",
                                context_lines=self._get_context_lines(j + 1)
                            ))

        return issues

    def run_all_detectors(self) -> Dict[str, List[DetectionResult]]:
        """
        Run all detection methods and return results grouped by detector type.

        Returns:
            Dictionary mapping detector names to their results
        """
        self._load_file()

        # Group detectors by category for better organization
        detector_groups = {
            DetectorCategory.INLINE_MATH: [
                (self.detect_inline_math_in_paragraphs, "Paragraphs with inline math"),
                (self.detect_inline_math_in_lists, "Inline math in lists"),
                (self.detect_inline_math_in_headings, "Inline math in headings"),
                (self.detect_inline_display_math_in_prose,
                 "Display math delimiters used inline in prose"),
                (self.detect_math_tokens_in_prose, "Math tokens in prose"),
            ],
            DetectorCategory.DISPLAY_MATH: [
                (self.detect_list_with_display_math, "List-marker lines with display math"),
                (self.detect_heading_with_math, "Heading lines with math"),
                (self.detect_table_with_display_math, "Table rows with display math"),
                (self.detect_display_math_with_list_marker, "Display math with list marker"),
                (self.detect_over_indented_display_math, "Over-indented display math"),
                (self.detect_adjacent_math_blocks, "Adjacent math blocks"),
                (self.detect_math_with_text_same_line, "Math with text on same line"),
                (self.detect_math_in_blockquotes, "Math in blockquotes"),
            ],
            DetectorCategory.LIST_FORMATTING: [
                (self.detect_list_missing_blank_line, "List missing blank line before math"),
                (self.detect_blank_lines_between_list_items, "Blank lines between list items"),
                (self.detect_text_to_list_missing_blank_line,
                 "Missing blank line before list items after text"),
            ],
            DetectorCategory.ALIGNMENT: [
                (self.detect_backslashes_in_aligned, "Backslashes in aligned"),
                (self.detect_math_missing_aligned, "Math missing aligned"),
                (self.detect_end_aligned_with_trailing_text, "End aligned with trailing text"),
                (self.detect_mismatched_aligned_blocks, "Mismatched aligned blocks"),
                (self.detect_end_before_begin_aligned, "End before begin aligned"),
            ],
            DetectorCategory.STRUCTURAL: [
                (self.detect_consecutive_bold_without_spacing,
                 "Consecutive bold text without spacing"),
                (self.detect_escaped_underscores_in_code, "Escaped underscores in code blocks"),
            ],
            DetectorCategory.SYNTAX: [
                (self.detect_unpaired_math_delimiters, "Unpaired math delimiters"),
                (self.detect_stray_braces_with_math, "Stray braces with math"),
                (self.detect_bold_markdown_in_math, "Bold Markdown in math"),
                (self.detect_math_code_fence_blocks, "Math code fence blocks"),
                (self.detect_textstyle_missing_opening_delimiter,
                 "Textstyle blocks missing opening delimiter"),
            ],
        }

        results = {}
        logger = logging.getLogger(LOGGER_NAME)

        for category, detectors in detector_groups.items():
            logger.debug("Running %s detectors", category.value)
            for detector_func, description in detectors:
                try:
                    detector_results = detector_func()
                    results[description] = detector_results
                    logger.debug("Detector '%s': %d issues found", description,
                               len(detector_results))
                except Exception as e:
                    logger.error("Error in detector '%s': %s", description, e)
                    results[description] = []

        return results

    def get_detector_groups(self) -> Dict[DetectorCategory, List[tuple]]:
        """
        Get all detector groups organized by category.

        Returns:
            Dictionary mapping categories to detector function/description tuples
        """
        return {
            DetectorCategory.INLINE_MATH: [
                (self.detect_inline_math_in_paragraphs, "Paragraphs with inline math"),
                (self.detect_inline_math_in_lists, "Inline math in lists"),
                (self.detect_inline_math_in_headings, "Inline math in headings"),
                (self.detect_inline_display_math_in_prose,
                 "Display math delimiters used inline in prose"),
                (self.detect_math_tokens_in_prose, "Math tokens in prose"),
            ],
            DetectorCategory.DISPLAY_MATH: [
                (self.detect_list_with_display_math, "List-marker lines with display math"),
                (self.detect_heading_with_math, "Heading lines with math"),
                (self.detect_table_with_display_math, "Table rows with display math"),
                (self.detect_display_math_with_list_marker, "Display math with list marker"),
                (self.detect_over_indented_display_math, "Over-indented display math"),
                (self.detect_adjacent_math_blocks, "Adjacent math blocks"),
                (self.detect_math_with_text_same_line, "Math with text on same line"),
                (self.detect_math_in_blockquotes, "Math in blockquotes"),
            ],
            DetectorCategory.LIST_FORMATTING: [
                (self.detect_list_missing_blank_line, "List missing blank line before math"),
                (self.detect_blank_lines_between_list_items, "Blank lines between list items"),
                (self.detect_text_to_list_missing_blank_line,
                 "Missing blank line before list items after text"),
            ],
            DetectorCategory.ALIGNMENT: [
                (self.detect_backslashes_in_aligned, "Backslashes in aligned"),
                (self.detect_math_missing_aligned, "Math missing aligned"),
                (self.detect_end_aligned_with_trailing_text, "End aligned with trailing text"),
                (self.detect_mismatched_aligned_blocks, "Mismatched aligned blocks"),
                (self.detect_end_before_begin_aligned, "End before begin aligned"),
            ],
            DetectorCategory.STRUCTURAL: [
                (self.detect_consecutive_bold_without_spacing,
                 "Consecutive bold text without spacing"),
                (self.detect_escaped_underscores_in_code, "Escaped underscores in code blocks"),
            ],
            DetectorCategory.SYNTAX: [
                (self.detect_unpaired_math_delimiters, "Unpaired math delimiters"),
                (self.detect_stray_braces_with_math, "Stray braces with math"),
                (self.detect_bold_markdown_in_math, "Bold Markdown in math"),
                (self.detect_math_code_fence_blocks, "Math code fence blocks"),
                (self.detect_textstyle_missing_opening_delimiter,
                 "Textstyle blocks missing opening delimiter"),
            ],
        }

    def run_selected_detectors(self, selected_detectors: List[str]) -> Dict[str, List[DetectionResult]]:
        """
        Run only selected detectors.

        Args:
            selected_detectors: List of detector names to run

        Returns:
            Dictionary mapping detector names to their results
        """
        self._load_file()

        # Create mapping from description to detector function
        detector_groups = self.get_detector_groups()
        all_detectors = {}
        for detectors in detector_groups.values():
            for detector_func, description in detectors:
                all_detectors[description] = detector_func

        results = {}
        logger = logging.getLogger(LOGGER_NAME)

        for detector_name in selected_detectors:
            if detector_name in all_detectors:
                try:
                    detector_results = all_detectors[detector_name]()
                    results[detector_name] = detector_results
                    logger.debug("Detector '%s': %d issues found", detector_name,
                               len(detector_results))
                except Exception as e:
                    logger.error("Error in detector '%s': %s", detector_name, e)
                    results[detector_name] = []
            else:
                logger.warning("Unknown detector: %s", detector_name)
                results[detector_name] = []

        return results

    def list_all_detectors(self) -> List[str]:
        """
        Get list of all available detector names.

        Returns:
            List of detector names
        """
        detector_groups = self.get_detector_groups()
        all_detectors = []
        for detectors in detector_groups.values():
            all_detectors.extend([description for _, description in detectors])
        return sorted(all_detectors)

    def format_results_by_type(self) -> str:
        """
        Run all detectors and format results grouped by detection type.

        Returns:
            Formatted string with results segmented by detector type
        """
        results_by_type = self.run_all_detectors()
        output_sections = []
        total_issues = 0

        for description, results in results_by_type.items():
            section_lines = [
                "---",
                f"Detector: {description}",
                f"Issues found: {len(results)}",
                ""
            ]

            if results:
                for result in results:
                    section_lines.extend([
                        f"Line {result.line_number}: {result.description}",
                        f"  Content: {result.content}",
                        ""
                    ])
                    if result.context_lines:
                        section_lines.append("  Context:")
                        section_lines.extend(f"    {line}" for line in result.context_lines)
                        section_lines.append("")
                total_issues += len(results)
            else:
                section_lines.append("No issues found.")
                section_lines.append("")

            output_sections.extend(section_lines)

        # Add summary at the end
        summary = [
            "---",
            f"SUMMARY: {total_issues} total issues found across {len(self.lines)} lines",
            ""
        ]
        output_sections.extend(summary)

        return "\n".join(output_sections)


def get_category_mapping() -> Dict[str, DetectorCategory]:
    """
    Get mapping from category names to enum values.

    Returns:
        Dictionary mapping category names to DetectorCategory enums
    """
    return {
        'inline_math': DetectorCategory.INLINE_MATH,
        'display_math': DetectorCategory.DISPLAY_MATH,
        'list_formatting': DetectorCategory.LIST_FORMATTING,
        'alignment': DetectorCategory.ALIGNMENT,
        'structural': DetectorCategory.STRUCTURAL,
        'syntax': DetectorCategory.SYNTAX
    }


def get_detector_name_mapping(detector: MarkdownMathDetector) -> Dict[str, str]:
    """
    Get mapping from detector command names to display names.

    Args:
        detector: Detector instance to get available detectors from

    Returns:
        Dictionary mapping command names to display names
    """
    detector_groups = detector.get_detector_groups()
    name_mapping = {}

    for detectors in detector_groups.values():
        for _, description in detectors:
            # Convert display name to command name
            command_name = description.lower().replace(' ', '_').replace('-', '_')
            name_mapping[command_name] = description

    return name_mapping


def print_categories_help() -> None:
    """Print detailed help for all categories with examples."""
    categories = {
        'inline_math': {
            'name': 'Inline Math Issues',
            'count': 5,
            'description': 'Variables and equations mixed with prose text',
            'before': 'The formula x_1 = 5 shows that variables can be inline.',
            'after': 'The formula shows that variables can be mathematical:\n\n$$\\begin{aligned}\nx\\_1 = 5\n\\end{aligned}$$'
        },
        'display_math': {
            'name': 'Display Math Issues',
            'count': 8,
            'description': 'Math expressions in wrong locations',
            'before': '- $$x = 5$$ shows equation in list',
            'after': '- Mathematical relationship:\n\n  $$\\begin{aligned}\n  x = 5\n  \\end{aligned}$$'
        },
        'list_formatting': {
            'name': 'List Formatting Issues',
            'count': 3,
            'description': 'Missing blank lines before lists',
            'before': 'Text here.\n- List item immediately follows',
            'after': 'Text here.\n\n- List item with proper spacing'
        },
        'alignment': {
            'name': 'Math Alignment Issues',
            'count': 5,
            'description': 'Missing professional math structure',
            'before': '$$x = 5$$',
            'after': '$$\\begin{aligned}\nx = 5\n\\end{aligned}$$'
        },
        'structural': {
            'name': 'Structural Issues',
            'count': 2,
            'description': 'Bold text spacing and code block issues',
            'before': '**Bold text**\n**Another bold** without spacing',
            'after': '**Bold text**\n\n**Another bold** with proper spacing'
        },
        'syntax': {
            'name': 'Syntax Issues',
            'count': 3,
            'description': 'Unpaired delimiters and stray characters',
            'before': '{$$equation} with stray brace',
            'after': '$$equation$$ with clean syntax'
        }
    }

    print("Detector Categories:\n")
    for cmd_name, info in categories.items():
        print(f"{cmd_name} ({info['count']} detectors)")
        print(f"  {info['description']}")
        print(f"  Before: {info['before']}")
        print(f"  After:  {info['after']}")
        print()


def print_detectors_help(detector: MarkdownMathDetector) -> None:
    """Print detailed help for all detectors with examples."""
    examples = {
        'list_marker_lines_with_display_math': {
            'before': '- $$x = 5$$ equation in list marker',
            'after': '- Mathematical relationship:\n\n  $$\\begin{aligned}\n  x = 5\n  \\end{aligned}$$'
        },
        'heading_lines_with_math': {
            'before': '# Section with $$math$$ in title',
            'after': '# Section Title\n\n$$\\begin{aligned}\nmath\n\\end{aligned}$$'
        },
        'paragraphs_with_inline_math': {
            'before': 'The variable x_1 represents the input.',
            'after': 'The variable represents the input:\n\n$$\\begin{aligned}\nx\\_1\n\\end{aligned}$$'
        },
        'missing_blank_line_before_list_items_after_text': {
            'before': 'Text ending with period.\n- List item follows immediately',
            'after': 'Text ending with period.\n\n- List item with proper spacing'
        },
        'unpaired_math_delimiters': {
            'before': 'Text with $$ equation missing closing delimiter',
            'after': 'Text with proper delimiters:\n\n$$\\begin{aligned}\nequation\n\\end{aligned}$$'
        }
    }

    detector_groups = detector.get_detector_groups()
    print("Available Detectors:\n")

    for category, detectors in detector_groups.items():
        print(f"{category.value}:")
        for _, description in detectors:
            cmd_name = description.lower().replace(' ', '_').replace('-', '_')
            print(f"  {cmd_name}")
            if cmd_name in examples:
                ex = examples[cmd_name]
                print(f"    Before: {ex['before']}")
                print(f"    After:  {ex['after']}")
            print()


def print_specific_detector_help(detector_name: str, detector: MarkdownMathDetector) -> None:
    """Print help for a specific detector or category with examples."""
    # First check if it's a category
    category_mapping = get_category_mapping()
    if detector_name in category_mapping:
        print_category_help(detector_name)
        return

    # Check if it's a specific detector
    name_mapping = get_detector_name_mapping(detector)
    if detector_name not in name_mapping:
        print(f"Unknown detector or category: {detector_name}")
        print("Use --list-categories or --list-detectors to see available options.")
        return

    display_name = name_mapping[detector_name]

    # Specific examples for each detector
    examples = {
        'inline_math_in_paragraphs': {
            'description': 'Finds mathematical variables and expressions mixed within prose paragraphs',
            'before': 'The variable x_1 = 5 and equation y = mx + b should be in display blocks.',
            'after': 'The variables and equation should be in display blocks:\n\n$$\\begin{aligned}\nx\\_1 &= 5 \\newline\ny &= mx + b\n\\end{aligned}$$',
            'fixes': 'Separates math from prose, promotes to professional LaTeX blocks'
        },
        'list_marker_lines_with_display_math': {
            'description': 'Finds mathematical expressions directly in list markers (forbidden pattern)',
            'before': '- $$x = 5$$ shows the equation\n* $$y = 2$$ another equation',
            'after': '- Mathematical relationship:\n\n  $$\\begin{aligned}\n  x = 5\n  \\end{aligned}$$\n\n* Another equation:\n\n  $$\\begin{aligned}\n  y = 2\n  \\end{aligned}$$',
            'fixes': 'Moves math out of list markers, adds proper spacing and structure'
        },
        'missing_blank_line_before_list_items_after_text': {
            'description': 'Finds list items that immediately follow descriptive text without blank line',
            'before': 'Here are the key points.\n- First item\n- Second item',
            'after': 'Here are the key points.\n\n- First item\n- Second item',
            'fixes': 'Adds required blank line between prose text and list items for proper markdown formatting'
        }
    }

    if detector_name in examples:
        ex = examples[detector_name]
        print(f"Detector: {display_name}")
        print(f"Description: {ex['description']}")
        print(f"Fixes: {ex['fixes']}")
        print(f"\nBefore:\n{ex['before']}")
        print(f"\nAfter:\n{ex['after']}")
    else:
        print(f"Detector: {display_name}")
        print("No specific example available for this detector.")


def print_category_help(category_name: str) -> None:
    """Print help for a specific category with examples."""
    category_examples = {
        'inline_math': {
            'description': 'Detectors that find mathematical variables and equations mixed with prose text',
            'before': 'The formula x_1 = 5 shows that variables can be inline.',
            'after': 'The formula shows that variables can be mathematical:\n\n$$\\begin{aligned}\nx\\_1 = 5\n\\end{aligned}$$',
            'fixes': 'Separates math from prose, promotes to professional LaTeX display blocks'
        },
        'display_math': {
            'description': 'Detectors that find math expressions in wrong locations like lists or headings',
            'before': '- $$x = 5$$ shows equation in list',
            'after': '- Mathematical relationship:\n\n  $$\\begin{aligned}\n  x = 5\n  \\end{aligned}$$',
            'fixes': 'Moves math to proper locations with correct spacing and structure'
        },
        'list_formatting': {
            'description': 'Detectors that find missing blank lines before lists and improper list spacing',
            'before': 'Text here.\n- List item immediately follows',
            'after': 'Text here.\n\n- List item with proper spacing',
            'fixes': 'Adds required blank lines for proper markdown list formatting'
        },
        'alignment': {
            'description': 'Detectors that find math expressions missing professional aligned structure',
            'before': '$$x = 5$$',
            'after': '$$\\begin{aligned}\nx = 5\n\\end{aligned}$$',
            'fixes': 'Wraps math in aligned blocks for professional presentation'
        },
        'structural': {
            'description': 'Detectors that find structural formatting issues like bold text spacing',
            'before': '**Bold text**\n**Another bold** without spacing',
            'after': '**Bold text**\n\n**Another bold** with proper spacing',
            'fixes': 'Adds proper spacing between structural elements'
        },
        'syntax': {
            'description': 'Detectors that find syntax errors like unpaired delimiters and stray characters',
            'before': '{$$equation} with stray brace',
            'after': '$$equation$$ with clean syntax',
            'fixes': 'Removes stray characters and fixes delimiter pairing'
        }
    }

    if category_name in category_examples:
        ex = category_examples[category_name]
        print(f"Category: {category_name}")
        print(f"Description: {ex['description']}")
        print(f"Fixes: {ex['fixes']}")
        print(f"\nBefore:\n{ex['before']}")
        print(f"\nAfter:\n{ex['after']}")
    else:
        print(f"Category: {category_name}")
        print("No specific example available for this category.")


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Markdown and MathJax Formatting Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # We'll handle help manually
        epilog="""
Usage Patterns:
  %(prog)s document.md                           # Run all detectors
  %(prog)s --list-categories                     # Show detector categories
  %(prog)s --list-detectors                      # Show all detector names
  %(prog)s --category inline_math document.md    # Run category detectors
  %(prog)s --detector inline_math_in_paragraphs document.md  # Run specific detector
  %(prog)s --verbose document.md                 # Run with verbose logging

Help for Categories and Detectors:
  %(prog)s --list-categories --help              # Show category examples
  %(prog)s --list-detectors --help               # Show detector examples
  %(prog)s --category inline_math --help         # Show category help with examples
  %(prog)s --detector inline_math_in_paragraphs --help  # Show specific detector help
""")

    parser.add_argument('filepath', nargs='?', type=Path,
                       help='Path to the markdown file to analyze')

    parser.add_argument('--list-categories', action='store_true',
                       help='List all detector categories')

    parser.add_argument('--list-detectors', action='store_true',
                       help='List all available detector names')

    parser.add_argument('--category', type=str, metavar='NAME',
                       help='Run all detectors in a specific category')

    parser.add_argument('--detector', type=str, metavar='NAME',
                       help='Run a specific individual detector')

    parser.add_argument('--help', '-h', action='store_true',
                       help='Show this help message and exit, or show examples when used with --list-* or --category/--detector')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    return parser


def get_selected_detectors_by_category(category_name: str, detector: MarkdownMathDetector) -> List[str]:
    """
    Get all detectors in a specific category.

    Args:
        category_name: Name of detector category
        detector: Detector instance to get available detectors from

    Returns:
        List of detector names to run
    """
    category_mapping = get_category_mapping()

    if category_name in category_mapping:
        category = category_mapping[category_name]
        detector_groups = detector.get_detector_groups()
        return [desc for _, desc in detector_groups[category]]

    return []


def get_selected_detector_by_name(detector_name: str, detector: MarkdownMathDetector) -> List[str]:
    """
    Get a specific detector by name.

    Args:
        detector_name: Name of specific detector
        detector: Detector instance to get available detectors from

    Returns:
        List containing single detector name if found, empty list otherwise
    """
    name_mapping = get_detector_name_mapping(detector)

    if detector_name in name_mapping:
        return [name_mapping[detector_name]]

    return []


def main() -> int:
    """
    Main entry point for command-line usage.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(LOGGER_NAME)

    # Create dummy detector for list operations
    dummy_detector = MarkdownMathDetector(Path('dummy'))

    # Handle general --help (when no other options are provided)
    if args.help and not (args.list_categories or args.list_detectors or args.category or args.detector):
        parser.print_help()
        return EXIT_SUCCESS

    # Handle --list-categories
    if args.list_categories:
        if args.help:
            print_categories_help()
        else:
            categories = get_category_mapping()
            print("Available Categories:")
            for category_name in categories:
                print(f"  {category_name}")
            print("\nUse --list-categories --help for examples")
        return EXIT_SUCCESS

    # Handle --list-detectors
    if args.list_detectors:
        if args.help:
            print_detectors_help(dummy_detector)
        else:
            all_detectors = dummy_detector.list_all_detectors()
            name_mapping = get_detector_name_mapping(dummy_detector)
            reverse_mapping = {v: k for k, v in name_mapping.items()}

            print("Available Detectors:")
            for detector_name in all_detectors:
                cmd_name = reverse_mapping.get(detector_name, detector_name.lower().replace(' ', '_'))
                print(f"  {cmd_name}")
            print("\nUse --list-detectors --help for examples")
        return EXIT_SUCCESS

    # Handle --category with --help
    if args.category and args.help:
        print_category_help(args.category)
        return EXIT_SUCCESS

    # Handle --detector with --help
    if args.detector and args.help:
        print_specific_detector_help(args.detector, dummy_detector)
        return EXIT_SUCCESS

    # Validate filepath for actual detection
    if not args.filepath:
        if args.category or args.detector:
            parser.error("filepath is required when using --category or --detector without --help")
        else:
            parser.error("filepath is required")

    if not args.filepath.exists():
        logger.error("File %s does not exist", args.filepath)
        return EXIT_ERROR

    try:
        detector = MarkdownMathDetector(args.filepath)

        # Determine which detectors to run
        selected_detectors = []
        operation_name = "all detectors"

        if args.category:
            selected_detectors = get_selected_detectors_by_category(args.category, detector)
            if not selected_detectors:
                logger.error("Unknown category: %s", args.category)
                logger.info("Use --list-categories to see available categories")
                return EXIT_ERROR
            operation_name = f"category '{args.category}'"

        elif args.detector:
            selected_detectors = get_selected_detector_by_name(args.detector, detector)
            if not selected_detectors:
                logger.error("Unknown detector: %s", args.detector)
                logger.info("Use --list-detectors to see available detectors")
                return EXIT_ERROR
            operation_name = f"detector '{args.detector}'"

        # Run detectors
        if selected_detectors:
            logger.info("Running %d detectors for %s", len(selected_detectors), operation_name)
            results = detector.run_selected_detectors(selected_detectors)
        else:
            # Run all detectors (default behavior)
            logger.info("Running all detectors")
            results = detector.run_all_detectors()

        # Format and print results
        output_sections = []
        total_issues = 0

        for description, issues in results.items():
            section_lines = [
                "---",
                f"Detector: {description}",
                f"Issues found: {len(issues)}",
                ""
            ]

            if issues:
                for result in issues:
                    section_lines.extend([
                        f"Line {result.line_number}: {result.description}",
                        f"  Content: {result.content}",
                        ""
                    ])
                    if result.context_lines:
                        section_lines.append("  Context:")
                        section_lines.extend(f"    {line}" for line in result.context_lines)
                        section_lines.append("")
                total_issues += len(issues)
            else:
                section_lines.append("No issues found.")
                section_lines.append("")

            output_sections.extend(section_lines)

        # Add summary
        if detector.lines:  # Only if file was loaded
            summary = [
                "---",
                f"SUMMARY: {total_issues} total issues found across {len(detector.lines)} lines",
                ""
            ]
            output_sections.extend(summary)

        print("\n".join(output_sections))
        return EXIT_SUCCESS

    except Exception as e:
        logger.error("Error: %s", e)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
