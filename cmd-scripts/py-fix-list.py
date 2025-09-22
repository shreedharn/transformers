#!/usr/bin/env python3
"""
Markdown List Formatting Detector

This module is created to support agentic AI execution.
Provides comprehensive detection of markdown list formatting issues,
particularly missing blank lines before list items.
"""

import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional, Pattern, Protocol


class IssueType(Enum):
    """Categories of markdown list formatting issues."""
    MISSING_BLANK_LINE_GENERAL = "List needs blank line above"
    MISSING_BLANK_LINE_AFTER_COLON = "Missing blank line after colon"
    MISSING_BLANK_LINE_AFTER_BOLD = "Missing blank line after bold text"
    MISSING_BLANK_LINE_AFTER_PARENTHESES = "Missing blank line after parentheses"
    MISSING_BLANK_LINE_AFTER_CODE = "Missing blank line after inline code/math"
    NUMBERED_LIST_SEPARATION = "Numbered list missing blank line separator"
    BULLETED_LIST_SEPARATION = "Bulleted list missing blank line separator"
    UNESCAPED_UNDERSCORE_LATEX = "Unescaped underscore in LaTeX block"

    # LaTeX formatting issues
    BULLET_CONTAINS_LATEX = "Markdown bullet contains LaTeX expression"
    MIXED_LATEX_NOTATION = "Mixed LaTeX notation (should use consistent format)"
    CONSOLIDATABLE_LATEX_BLOCKS = "Consecutive LaTeX blocks should be consolidated"
    BOUNDARY_VIOLATION = "LaTeX consolidation boundary violation"
    BOLD_IN_LATEX_MARKDOWN_STYLE = "Bold text using **markdown** instead of \\mathbf{} in LaTeX"
    MISSING_TEXT_WRAPPER = "Descriptive text in LaTeX missing \\text{} wrapper"
    MISSING_MATHBF_WRAPPER = "Mathematical expression missing \\mathbf{} wrapper"
    MARKDOWN_BULLET_IN_MATH = "Markdown bullet (-) in mathematical context (should use \\bullet)"
    INCONSISTENT_SPACING = "Inconsistent spacing in LaTeX (should use \\quad)"
    STRUCTURAL_INCONSISTENCY = "LaTeX structural inconsistency"


@dataclass
class DetectionResult:
    """Represents a detected markdown list formatting issue."""
    line_number: int
    issue_type: IssueType
    previous_line: str
    current_line: str
    description: str = ""

    def __post_init__(self) -> None:
        """Generate description if not provided."""
        if not self.description:
            self.description = self.issue_type.value


class LineContext(NamedTuple):
    """Context information for a line during processing."""
    number: int
    content: str
    is_blank: bool
    is_header: bool
    is_list_item: bool


# Compiled regex patterns for performance
PATTERNS = {
    'list_item': re.compile(r'^\s*([-*+]|[0-9]+\.)\s'),
    'numbered_list': re.compile(r'^\s*[0-9]+\.\s'),
    'bulleted_list': re.compile(r'^\s*[-*+]\s'),
    'header': re.compile(r'^#{1,6}\s'),
    'blank_line': re.compile(r'^\s*$'),
    'colon_ending': re.compile(r':$'),
    'bold_text_ending': re.compile(r'\*\*.*\*\*:?\s*$'),
    'parentheses_ending': re.compile(r'\):?$'),
    'code_math_ending': re.compile(r'(`[^`]+`|\$[^$]+\$):?$'),
    # LaTeX block patterns
    'latex_block_start': re.compile(r'\$\$'),
    'latex_aligned_start': re.compile(r'\\begin\{aligned\}'),
    'latex_aligned_end': re.compile(r'\\end\{aligned\}'),
    'latex_inline': re.compile(r'\$[^$]+\$'),
    # Underscore patterns - detect unescaped underscores in LaTeX contexts
    'unescaped_underscore': re.compile(r'(?<!\\)_'),
    'latex_subscript': re.compile(r'(?<!\\)_\{[^}]+\}'),

    # LaTeX formatting patterns
    'bullet_with_latex': re.compile(r'^\s*[-*+]\s+.*(\$\$|\$[^$]+\$|\\begin\{aligned\})'),
    'standalone_dollar_expr': re.compile(r'^\s*\$\$[^$]+\$\$\s*$'),
    'standalone_aligned': re.compile(r'^\s*\\begin\{aligned\}.*\\end\{aligned\}\s*$'),
    'consecutive_latex_blocks': re.compile(r'(\$\$|\\\w+\{)'),
    'markdown_bold_in_latex': re.compile(r'\*\*[^*]+\*\*'),
    'text_without_wrapper': re.compile(r'[a-zA-Z]+[^\\$]*(?!\\text)'),
    'math_without_mathbf': re.compile(r'(?<!\\mathbf\{)[a-zA-Z]+(?![^{]*\})'),
    'markdown_bullet_symbol': re.compile(r'(?<!\\)-(?!\s*\d)'),
    'missing_quad_spacing': re.compile(r'[&:]\s*[^\\]'),
    'inconsistent_newline': re.compile(r'\\\\(?!\s*\\newline)'),
}


class MarkdownListDetector:
    """Main class for detecting markdown list formatting issues."""

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
        stripped = content.strip()
        return LineContext(
            number=line_num,
            content=content,
            is_blank=bool(PATTERNS['blank_line'].match(content)),
            is_header=bool(PATTERNS['header'].match(content)),
            is_list_item=bool(PATTERNS['list_item'].match(content))
        )

    def _is_valid_list_issue(self, current: LineContext, previous: LineContext) -> bool:
        """
        Check if a list item represents a valid formatting issue.

        Args:
            current: Current line context
            previous: Previous line context

        Returns:
            True if this represents a formatting issue
        """
        if not current.is_list_item:
            return False

        # Allow lists after headers and blank lines
        if previous.is_blank or previous.is_header or previous.is_list_item:
            return False

        return True

    def detect_general_list_issues(self) -> List[DetectionResult]:
        """
        Detect general list formatting issues (primary detector).

        Returns:
            List of detected issues
        """
        issues = []

        for i in range(1, len(self.lines)):
            current = self._create_line_context(i + 1, self.lines[i])
            previous = self._create_line_context(i, self.lines[i - 1])

            if self._is_valid_list_issue(current, previous):
                issues.append(DetectionResult(
                    line_number=current.number,
                    issue_type=IssueType.MISSING_BLANK_LINE_GENERAL,
                    previous_line=previous.content.strip(),
                    current_line=current.content.strip()
                ))

        return issues

    def detect_lists_after_colons(self) -> List[DetectionResult]:
        """
        Detect lists immediately following colons without blank lines.

        Returns:
            List of detected issues
        """
        issues = []

        for i in range(1, len(self.lines)):
            current = self._create_line_context(i + 1, self.lines[i])
            previous = self._create_line_context(i, self.lines[i - 1])

            if (current.is_list_item and
                not previous.is_list_item and  # Don't flag consecutive list items
                PATTERNS['colon_ending'].search(previous.content.strip())):
                issues.append(DetectionResult(
                    line_number=current.number,
                    issue_type=IssueType.MISSING_BLANK_LINE_AFTER_COLON,
                    previous_line=previous.content.strip(),
                    current_line=current.content.strip()
                ))

        return issues

    def detect_lists_after_bold_text(self) -> List[DetectionResult]:
        """
        Detect lists immediately following bold/italic markdown without blank lines.

        Returns:
            List of detected issues
        """
        issues = []

        for i in range(1, len(self.lines)):
            current = self._create_line_context(i + 1, self.lines[i])
            previous = self._create_line_context(i, self.lines[i - 1])

            if (current.is_list_item and
                not previous.is_list_item and  # Don't flag consecutive list items
                PATTERNS['bold_text_ending'].search(previous.content.strip())):
                issues.append(DetectionResult(
                    line_number=current.number,
                    issue_type=IssueType.MISSING_BLANK_LINE_AFTER_BOLD,
                    previous_line=previous.content.strip(),
                    current_line=current.content.strip()
                ))

        return issues

    def detect_lists_after_parentheses(self) -> List[DetectionResult]:
        """
        Detect lists immediately following parenthetical statements without blank lines.

        Returns:
            List of detected issues
        """
        issues = []

        for i in range(1, len(self.lines)):
            current = self._create_line_context(i + 1, self.lines[i])
            previous = self._create_line_context(i, self.lines[i - 1])

            if (current.is_list_item and
                not previous.is_list_item and  # Don't flag consecutive list items
                PATTERNS['parentheses_ending'].search(previous.content.strip())):
                issues.append(DetectionResult(
                    line_number=current.number,
                    issue_type=IssueType.MISSING_BLANK_LINE_AFTER_PARENTHESES,
                    previous_line=previous.content.strip(),
                    current_line=current.content.strip()
                ))

        return issues

    def detect_lists_after_code_math(self) -> List[DetectionResult]:
        """
        Detect lists immediately following inline code or math expressions without blank lines.

        Returns:
            List of detected issues
        """
        issues = []

        for i in range(1, len(self.lines)):
            current = self._create_line_context(i + 1, self.lines[i])
            previous = self._create_line_context(i, self.lines[i - 1])

            if (current.is_list_item and
                not previous.is_list_item and  # Don't flag consecutive list items
                PATTERNS['code_math_ending'].search(previous.content.strip())):
                issues.append(DetectionResult(
                    line_number=current.number,
                    issue_type=IssueType.MISSING_BLANK_LINE_AFTER_CODE,
                    previous_line=previous.content.strip(),
                    current_line=current.content.strip()
                ))

        return issues

    def detect_numbered_list_issues(self) -> List[DetectionResult]:
        """
        Detect numbered lists missing blank line separators.

        Returns:
            List of detected issues
        """
        issues = []

        for i in range(1, len(self.lines)):
            current = self._create_line_context(i + 1, self.lines[i])
            previous = self._create_line_context(i, self.lines[i - 1])

            if (PATTERNS['numbered_list'].match(current.content) and
                not previous.is_blank and not previous.is_header and
                not PATTERNS['numbered_list'].match(previous.content)):
                issues.append(DetectionResult(
                    line_number=current.number,
                    issue_type=IssueType.NUMBERED_LIST_SEPARATION,
                    previous_line=previous.content.strip(),
                    current_line=current.content.strip()
                ))

        return issues

    def detect_bulleted_list_issues(self) -> List[DetectionResult]:
        """
        Detect bulleted lists missing blank line separators.

        Returns:
            List of detected issues
        """
        issues = []

        for i in range(1, len(self.lines)):
            current = self._create_line_context(i + 1, self.lines[i])
            previous = self._create_line_context(i, self.lines[i - 1])

            if (PATTERNS['bulleted_list'].match(current.content) and
                not previous.is_blank and not previous.is_header and
                not PATTERNS['bulleted_list'].match(previous.content)):
                issues.append(DetectionResult(
                    line_number=current.number,
                    issue_type=IssueType.BULLETED_LIST_SEPARATION,
                    previous_line=previous.content.strip(),
                    current_line=current.content.strip()
                ))

        return issues

    def detect_unescaped_underscores_in_latex(self) -> List[DetectionResult]:
        """
        Detect unescaped underscores in LaTeX mathematical expressions.

        This is critical for Markdown compatibility as raw underscores in LaTeX
        can be interpreted as Markdown italics, breaking the math rendering.

        Returns:
            List of detected issues with unescaped underscores
        """
        issues = []

        # Track whether we're inside different types of blocks
        in_latex_block = False
        in_aligned_block = False
        in_code_block = False
        latex_block_start_line = 0

        # Patterns for code blocks
        code_block_start = re.compile(r'^```')

        for i, line in enumerate(self.lines):
            line_num = i + 1
            stripped_line = line.strip()

            # Track code block boundaries (ignore everything inside)
            if code_block_start.match(stripped_line):
                in_code_block = not in_code_block
                continue

            # Skip everything inside code blocks
            if in_code_block:
                continue

            # Track LaTeX block boundaries - only double $$, not single $
            double_dollar_matches = list(re.finditer(r'\$\$', line))
            if double_dollar_matches:
                # Toggle in_latex_block state for each $$ found
                for _ in double_dollar_matches:
                    if not in_latex_block:
                        in_latex_block = True
                        latex_block_start_line = line_num
                    else:
                        in_latex_block = False

            if PATTERNS['latex_aligned_start'].search(line):
                in_aligned_block = True
                if not in_latex_block:
                    latex_block_start_line = line_num

            if PATTERNS['latex_aligned_end'].search(line):
                in_aligned_block = False

            # Check for unescaped underscores in LaTeX contexts
            if (in_latex_block or in_aligned_block) and stripped_line:
                # Skip lines that only contain $$ markers
                if stripped_line == '$$':
                    continue

                # Find all unescaped underscores
                underscore_matches = list(PATTERNS['unescaped_underscore'].finditer(stripped_line))

                if underscore_matches:
                    # Create detailed issue description with examples
                    sample_problematic = []

                    # Extract specific problematic patterns by finding underscore-based subscripts
                    subscript_matches = PATTERNS['latex_subscript'].findall(stripped_line)
                    if subscript_matches:
                        sample_problematic.extend(subscript_matches[:3])
                    else:
                        # If no subscript patterns, just show the immediate context around underscores
                        for match in underscore_matches[:3]:
                            start = max(0, match.start() - 5)
                            end = min(len(stripped_line), match.end() + 10)
                            context = stripped_line[start:end]
                            sample_problematic.append(f"...{context}...")

                    description = f"Found {len(underscore_matches)} unescaped underscore(s) in LaTeX block"
                    if sample_problematic:
                        description += f". Examples: {', '.join(sample_problematic[:3])}"

                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.UNESCAPED_UNDERSCORE_LATEX,
                        previous_line=f"LaTeX block started at line {latex_block_start_line}",
                        current_line=stripped_line,
                        description=description
                    ))

            # Check inline LaTeX expressions (single $ pairs) only when not in block mode
            elif not (in_latex_block or in_aligned_block):
                # Use a more precise regex to match proper inline LaTeX: $content$
                inline_pattern = re.compile(r'\$([^$]+)\$')
                inline_matches = inline_pattern.finditer(stripped_line)

                for match in inline_matches:
                    latex_content = match.group()
                    inner_content = match.group(1)  # Content between the $

                    # Only flag if it contains mathematical content (has backslash, braces, or common math chars)
                    if (re.search(r'[\\{}^_]', inner_content) and
                        PATTERNS['unescaped_underscore'].search(inner_content)):
                        issues.append(DetectionResult(
                            line_number=line_num,
                            issue_type=IssueType.UNESCAPED_UNDERSCORE_LATEX,
                            previous_line="Inline LaTeX expression",
                            current_line=latex_content,
                            description=f"Unescaped underscore in inline LaTeX: {latex_content}"
                        ))

        return issues

    def detect_bullets_containing_latex(self) -> List[DetectionResult]:
        """
        Detect markdown bullets that contain LaTeX expressions (Rule 1).

        This violates the rule: "NEVER have markdown bullets containing MathJax expressions"

        Returns:
            List of detected issues with bullets containing LaTeX
        """
        issues = []

        for i, line in enumerate(self.lines):
            line_num = i + 1
            stripped_line = line.strip()

            if PATTERNS['bullet_with_latex'].match(stripped_line):
                # Extract the LaTeX content for the description
                latex_matches = re.findall(r'(\$\$[^$]*\$\$|\$[^$]+\$|\\begin\{aligned\}.*?\\end\{aligned\})', stripped_line)
                latex_content = ', '.join(latex_matches[:2]) if latex_matches else "LaTeX expression"

                issues.append(DetectionResult(
                    line_number=line_num,
                    issue_type=IssueType.BULLET_CONTAINS_LATEX,
                    previous_line="",
                    current_line=stripped_line,
                    description=f"Bullet contains LaTeX expression: {latex_content}"
                ))

        return issues

    def detect_mixed_latex_notation(self) -> List[DetectionResult]:
        """
        Detect mixed LaTeX notation that should be standardized.

        This detects cases where $$expression$$ and \\begin{aligned} are mixed
        instead of using consistent \\begin{aligned} format.

        Returns:
            List of detected issues with mixed notation
        """
        issues = []

        for i, line in enumerate(self.lines):
            line_num = i + 1
            stripped_line = line.strip()

            # Check for standalone $$expression$$ that should be converted to aligned blocks
            if (PATTERNS['standalone_dollar_expr'].match(stripped_line) and
                not re.search(r'\\begin\{aligned\}', stripped_line)):

                # Extract the expression content
                expr_match = re.search(r'\$\$([^$]+)\$\$', stripped_line)
                if expr_match:
                    expression = expr_match.group(1).strip()
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.MIXED_LATEX_NOTATION,
                        previous_line="",
                        current_line=stripped_line,
                        description=f"Should convert $$expression$$ to \\begin{{aligned}} format: {expression[:50]}"
                    ))

        return issues

    def detect_consolidatable_latex_blocks(self) -> List[DetectionResult]:
        """
        Detect consecutive LaTeX blocks that could be consolidated (Rule 2).

        Identifies multiple separate blocks that are adjacent and could be merged
        while respecting boundary rules.

        Returns:
            List of detected consolidation opportunities
        """
        issues = []

        # Track consecutive LaTeX blocks
        prev_was_latex = False
        prev_line_num = 0
        prev_line_content = ""

        for i, line in enumerate(self.lines):
            line_num = i + 1
            stripped_line = line.strip()

            # Check if current line contains LaTeX
            current_is_latex = bool(
                PATTERNS['latex_block_start'].search(line) or
                PATTERNS['latex_aligned_start'].search(line) or
                PATTERNS['standalone_aligned'].match(stripped_line)
            )

            # Check if this line ends a LaTeX block
            current_ends_latex = bool(
                PATTERNS['latex_block_start'].search(line) or  # $$ can end blocks
                PATTERNS['latex_aligned_end'].search(line)
            )

            # If we have consecutive LaTeX blocks separated only by blank lines
            if (prev_was_latex and current_is_latex and
                line_num - prev_line_num <= 2):  # Allow one blank line between

                # Check if there's intervening non-mathematical text (boundary rule)
                has_text_boundary = False
                for j in range(prev_line_num, line_num - 1):
                    if j < len(self.lines):
                        intervening_line = self.lines[j].strip()
                        if (intervening_line and
                            not PATTERNS['blank_line'].match(self.lines[j]) and
                            not PATTERNS['latex_block_start'].search(self.lines[j]) and
                            not PATTERNS['latex_aligned_end'].search(self.lines[j])):
                            has_text_boundary = True
                            break

                if not has_text_boundary:
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.CONSOLIDATABLE_LATEX_BLOCKS,
                        previous_line=prev_line_content,
                        current_line=stripped_line,
                        description=f"Consecutive LaTeX blocks on lines {prev_line_num}-{line_num} could be consolidated"
                    ))

            # Update tracking
            if current_ends_latex or current_is_latex:
                prev_was_latex = True
                prev_line_num = line_num
                prev_line_content = stripped_line
            elif stripped_line and not PATTERNS['blank_line'].match(line):
                prev_was_latex = False

        return issues

    def detect_typography_issues(self) -> List[DetectionResult]:
        """
        Detect typography standard violations (Rule 3).

        Checks for:
        - **bold** instead of \\mathbf{} in LaTeX contexts
        - Missing \\text{} wrappers for descriptive text
        - Markdown bullets (-) instead of \\bullet in math contexts

        Returns:
            List of detected typography issues
        """
        issues = []

        # Track if we're in LaTeX context
        in_latex_context = False
        in_code_block = False
        code_block_pattern = re.compile(r'^```')

        for i, line in enumerate(self.lines):
            line_num = i + 1
            stripped_line = line.strip()

            # Track code blocks
            if code_block_pattern.match(stripped_line):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            # Track LaTeX context more precisely
            double_dollar_matches = len(re.findall(r'\$\$', line))
            if double_dollar_matches % 2 == 1:  # Odd number toggles state
                in_latex_context = not in_latex_context
            elif PATTERNS['latex_aligned_start'].search(line):
                in_latex_context = True
            elif PATTERNS['latex_aligned_end'].search(line):
                in_latex_context = False

            # Check for typography issues in LaTeX context
            if in_latex_context and stripped_line:
                # Check for **bold** instead of \mathbf{}
                bold_matches = PATTERNS['markdown_bold_in_latex'].finditer(stripped_line)
                for match in bold_matches:
                    bold_text = match.group()
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.BOLD_IN_LATEX_MARKDOWN_STYLE,
                        previous_line="In LaTeX context",
                        current_line=stripped_line,
                        description=f"Use \\mathbf{{}} instead of {bold_text} in LaTeX"
                    ))

                # Check for markdown bullets in mathematical context (only in actual math lines)
                if (stripped_line.startswith('-') and
                    re.search(r'[=\\{}^_&]', stripped_line) and  # Contains clear math symbols
                    not stripped_line.startswith('- **') and  # Not regular markdown bullets
                    not re.search(r'[a-zA-Z]+ [a-zA-Z]+', stripped_line)):  # Not prose
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.MARKDOWN_BULLET_IN_MATH,
                        previous_line="Mathematical context",
                        current_line=stripped_line,
                        description="Use \\bullet instead of markdown bullet (-) in mathematical context"
                    ))

        return issues

    def detect_lists_after_blockquotes(self) -> List[DetectionResult]:
        """Detect lists immediately following blockquotes without blank line."""
        issues = []

        for line_num, line in enumerate(self.lines[1:], 2):
            if re.match(r'^\s*([-*+]|[0-9]+\.)\s', line):
                prev_line = self.lines[line_num - 2].strip()
                if prev_line.startswith('>') and not prev_line.endswith('    '):
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.MISSING_BLANK_LINE_GENERAL,
                        previous_line=prev_line,
                        current_line=line.strip(),
                        description="List after blockquote needs blank line"
                    ))

        return issues

    def detect_lists_after_tables(self) -> List[DetectionResult]:
        """Detect lists immediately following table rows without blank line."""
        issues = []

        for line_num, line in enumerate(self.lines[1:], 2):
            if re.match(r'^\s*([-*+]|[0-9]+\.)\s', line):
                prev_line = self.lines[line_num - 2].strip()
                if re.match(r'^\s*\|.*\|\s*$', prev_line):
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.MISSING_BLANK_LINE_GENERAL,
                        previous_line=prev_line,
                        current_line=line.strip(),
                        description="List after table row needs blank line"
                    ))

        return issues

    def detect_lists_after_html(self) -> List[DetectionResult]:
        """Detect lists immediately following HTML blocks without blank line."""
        issues = []

        for line_num, line in enumerate(self.lines[1:], 2):
            if re.match(r'^\s*([-*+]|[0-9]+\.)\s', line):
                prev_line = self.lines[line_num - 2].strip()
                if re.match(r'^<[^>]+>.*$', prev_line):
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.MISSING_BLANK_LINE_GENERAL,
                        previous_line=prev_line,
                        current_line=line.strip(),
                        description="List after HTML block needs blank line"
                    ))

        return issues

    def detect_lists_after_code_fences(self) -> List[DetectionResult]:
        """Detect lists immediately following code fence blocks without blank line."""
        issues = []

        for line_num, line in enumerate(self.lines[1:], 2):
            if re.match(r'^\s*([-*+]|[0-9]+\.)\s', line):
                prev_line = self.lines[line_num - 2].strip()
                if re.match(r'^```.*$', prev_line):
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.MISSING_BLANK_LINE_GENERAL,
                        previous_line=prev_line,
                        current_line=line.strip(),
                        description="List after code fence needs blank line"
                    ))

        return issues

    def detect_lists_after_mathtext_blocks(self) -> List[DetectionResult]:
        """Detect lists immediately following MathJax display blocks without blank line."""
        issues = []

        for line_num, line in enumerate(self.lines[1:], 2):
            if re.match(r'^\s*([-*+]|[0-9]+\.)\s', line):
                prev_line = self.lines[line_num - 2].strip()
                if re.match(r'^\$\$.*\$\$$', prev_line):
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.MISSING_BLANK_LINE_GENERAL,
                        previous_line=prev_line,
                        current_line=line.strip(),
                        description="List after MathJax block needs blank line"
                    ))

        return issues

    def detect_inconsistent_bullet_markers(self) -> List[DetectionResult]:
        """Detect mixed bullet markers (-, *, +) within the same document."""
        issues = []
        bullet_markers = set()

        for line_num, line in enumerate(self.lines, 1):
            match = re.match(r'^\s*([-*+])\s', line)
            if match:
                bullet_markers.add(match.group(1))

        if len(bullet_markers) > 1:
            # Find first occurrence of each marker type
            for line_num, line in enumerate(self.lines, 1):
                match = re.match(r'^\s*([-*+])\s', line)
                if match:
                    issues.append(DetectionResult(
                        line_number=line_num,
                        issue_type=IssueType.MISSING_BLANK_LINE_GENERAL,
                        previous_line="",
                        current_line=line.strip(),
                        description=f"Mixed bullet markers found: {', '.join(sorted(bullet_markers))}. Consider using consistent markers."
                    ))
                    break  # Only report once per document

        return issues

    def run_all_detectors(self) -> List[DetectionResult]:
        """
        Run all detection methods and return combined results.

        Returns:
            List of all detected issues sorted by line number
        """
        self._load_file()

        all_issues = []
        detectors = [
            self.detect_general_list_issues,
            self.detect_lists_after_colons,
            self.detect_lists_after_bold_text,
            self.detect_lists_after_parentheses,
            self.detect_lists_after_code_math,
            self.detect_numbered_list_issues,
            self.detect_bulleted_list_issues,
            self.detect_unescaped_underscores_in_latex,
            self.detect_bullets_containing_latex,
            self.detect_mixed_latex_notation,
            self.detect_consolidatable_latex_blocks,
            self.detect_typography_issues,
            # Additional consolidated detectors from grep patterns
            self.detect_lists_after_blockquotes,
            self.detect_lists_after_tables,
            self.detect_lists_after_html,
            self.detect_lists_after_code_fences,
            self.detect_lists_after_mathtext_blocks,
            self.detect_inconsistent_bullet_markers,
        ]

        for detector in detectors:
            all_issues.extend(detector())

        # Remove duplicates and sort by line number
        unique_issues = {(issue.line_number, issue.issue_type): issue for issue in all_issues}
        return sorted(unique_issues.values(), key=lambda x: x.line_number)

    def format_results_by_type(self) -> str:
        """
        Run all detectors and format results grouped by detection type.

        Returns:
            Formatted string with results segmented by detector type
        """
        self._load_file()

        output_sections = []

        # Define detectors with their descriptions
        detectors = [
            (self.detect_general_list_issues, "Lists starting immediately after prose without blank line"),
            (self.detect_lists_after_colons, "Lists following colons without separation"),
            (self.detect_lists_after_bold_text, "Lists after bold/italic markdown without separation"),
            (self.detect_lists_after_parentheses, "Lists after parenthetical statements without blank line"),
            (self.detect_lists_after_code_math, "Lists after inline code or math expressions"),
            (self.detect_numbered_list_issues, "Numbered lists missing blank line separator"),
            (self.detect_bulleted_list_issues, "Bulleted lists missing blank line separator"),
            (self.detect_unescaped_underscores_in_latex, "Unescaped underscores in LaTeX blocks (critical for Markdown compatibility)"),
            (self.detect_bullets_containing_latex, "Markdown bullets containing LaTeX expressions (Rule 1 violation)"),
            (self.detect_mixed_latex_notation, "Mixed LaTeX notation - should use consistent format"),
            (self.detect_consolidatable_latex_blocks, "Consecutive LaTeX blocks that could be consolidated"),
            (self.detect_typography_issues, "Typography standard violations (bold, bullets, spacing)"),
        ]

        total_issues = 0

        for detector_func, description in detectors:
            results = detector_func()
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
                        f"  Previous: {result.previous_line}",
                        f"  Current:  {result.current_line}",
                        ""
                    ])
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
        print("Usage: python py-fix-list.py <markdown_file>", file=sys.stderr)
        return 1

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist", file=sys.stderr)
        return 1

    try:
        detector = MarkdownListDetector(filepath)
        print(detector.format_results_by_type())
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())