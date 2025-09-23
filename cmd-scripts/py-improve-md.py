#!/usr/bin/env python3
"""
Markdown and MathJax Formatting Detector

This module is created to support agentic AI execution.
Provides comprehensive detection of markdown and MathJax formatting issues,
ensuring proper separation between prose and mathematical content.
"""

import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional, Pattern, Dict, Tuple


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
    BLANK_LINES_BETWEEN_LIST_ITEMS = "Blank lines between list items (should only be before list start)"
    ESCAPED_UNDERSCORES_IN_CODE = "Escaped underscores in code blocks (breaks syntax)"
    INLINE_DISPLAY_MATH_IN_PROSE = "Display math delimiters used inline in prose (should be blocks)"
    CONSECUTIVE_BOLD_WITHOUT_SPACING = "Consecutive bold text lines without proper spacing"
    TEXT_TO_LIST_MISSING_BLANK_LINE = "Missing blank line before list item after text"


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
    'list_with_display_math': re.compile(r'^\s*([-*+]|[0-9]+\.)\s.*(\$\$|\\begin\{aligned\})'),
    'heading_with_math': re.compile(r'^\s*#{1,6}\s.*(\$\$|\\begin\{aligned\}|\\\(|\\\))'),
    'table_with_display_math': re.compile(r'^\s*\|.*\$\$.*\|.*$'),
    'display_math_with_list': re.compile(r'^\s*([-*+]|[0-9]+\.)\s*\$\$'),
    'over_indented_math': re.compile(r'^\s{4,}\$\$'),
    'math_tokens_in_prose': re.compile(r'(^[^$`].*)\b(O\([^)]+\)|\\alpha|\\beta|\\gamma|\\sigma|\\mathbb\{|\\mathcal\{|\\frac\{|\\sum\b|\\prod\b|\\to\b|\\rightarrow\b|\\mid\b|\\lvert|\\rvert)\b'),
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
            with open(self.filepath, 'r', encoding='utf-8') as file:
                self.lines = file.readlines()
        except (IOError, OSError) as e:
            raise FileNotFoundError(f"Cannot read file {self.filepath}: {e}")

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

    def _get_context_lines(self, line_num: int, context: int = 3) -> List[str]:
        """Get context lines around a specific line number."""
        start = max(0, line_num - context - 1)
        end = min(len(self.lines), line_num + context)
        return [f"{i+1:4d}: {line.rstrip()}" for i, line in enumerate(self.lines[start:end], start)]

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
        code_block_start = 0

        for i, line in enumerate(self.lines):
            if PATTERNS['code_fence_start'].match(line):
                if not in_code_block:
                    in_code_block = True
                    code_block_start = i
                else:
                    in_code_block = False
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

    def run_all_detectors(self) -> Dict[str, List[DetectionResult]]:
        """
        Run all detection methods and return results grouped by detector type.

        Returns:
            Dictionary mapping detector names to their results
        """
        self._load_file()

        detectors = [
            (self.detect_list_with_display_math, "List-marker lines with display math"),
            (self.detect_heading_with_math, "Heading lines with math"),
            (self.detect_table_with_display_math, "Table rows with display math"),
            (self.detect_inline_math_in_paragraphs, "Paragraphs with inline math"),
            (self.detect_display_math_with_list_marker, "Display math with list marker"),
            (self.detect_over_indented_display_math, "Over-indented display math"),
            (self.detect_adjacent_math_blocks, "Adjacent math blocks"),
            (self.detect_math_tokens_in_prose, "Math tokens in prose"),
            (self.detect_bold_markdown_in_math, "Bold Markdown in math"),
            (self.detect_backslashes_in_aligned, "Backslashes in aligned"),
            (self.detect_math_missing_aligned, "Math missing aligned"),
            (self.detect_math_in_blockquotes, "Math in blockquotes"),
            (self.detect_list_missing_blank_line, "List missing blank line before math"),
            (self.detect_inline_math_in_lists, "Inline math in lists"),
            (self.detect_inline_math_in_headings, "Inline math in headings"),
            (self.detect_unpaired_math_delimiters, "Unpaired math delimiters"),
            (self.detect_math_with_text_same_line, "Math with text on same line"),
            (self.detect_stray_braces_with_math, "Stray braces with math"),
            (self.detect_end_aligned_with_trailing_text, "End aligned with trailing text"),
            (self.detect_mismatched_aligned_blocks, "Mismatched aligned blocks"),
            (self.detect_end_before_begin_aligned, "End before begin aligned"),
            (self.detect_blank_lines_between_list_items, "Blank lines between list items"),
            (self.detect_escaped_underscores_in_code, "Escaped underscores in code blocks"),
            (self.detect_inline_display_math_in_prose, "Display math delimiters used inline in prose"),
            (self.detect_consecutive_bold_without_spacing, "Consecutive bold text without spacing"),
            (self.detect_text_to_list_missing_blank_line, "Missing blank line before list items after text"),
        ]

        results = {}
        for detector_func, description in detectors:
            try:
                detector_results = detector_func()
                results[description] = detector_results
            except Exception as e:
                print(f"Error in detector '{description}': {e}", file=sys.stderr)
                results[description] = []

        return results

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


def main() -> int:
    """
    Main entry point for command-line usage.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if len(sys.argv) != 2:
        print("Usage: python py-simplify-md.py <markdown_file>", file=sys.stderr)
        return 1

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist", file=sys.stderr)
        return 1

    try:
        detector = MarkdownMathDetector(filepath)
        print(detector.format_results_by_type())
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())