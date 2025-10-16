#!/usr/bin/env python3
"""
List Formatting Detector for Markdown

This module is created to support agentic AI execution.
Provides focused detection of list formatting issues in markdown files,
ensuring proper blank line spacing and bold text formatting.

Supports command-line usage with selective detector execution, allowing users
to run specific detectors or all list-related detectors based on command-line arguments.
Includes comprehensive help and logging capabilities.
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List

# Module-level constants
DEFAULT_CONTEXT_LINES = 8
DEFAULT_FILE_ENCODING = 'utf-8'
LOGGER_NAME = 'list_format_detector'

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1


class IssueType(Enum):
    """Categories of list formatting issues."""
    LIST_MISSING_BLANK_LINE_BEFORE = "List missing blank line before first item"
    BLANK_LINES_BETWEEN_LIST_ITEMS = "Blank lines between list items (should only be before list start)"
    CONSECUTIVE_BOLD_WITHOUT_SPACING = "Consecutive bold text lines without proper spacing"
    TICK_ICON_IN_LIST = "Tick icon (✅) used instead of normal list marker"
    EXCESSIVE_BLANK_LINES = "Excessive consecutive blank lines (more than 1)"


@dataclass
class DetectionResult:
    """Represents a detected list formatting issue."""
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


# Compiled regex patterns for performance
PATTERNS = {
    'list_marker': re.compile(r'^\s*([-*+]|[0-9]+\.)\s'),
    'blank_line': re.compile(r'^\s*$'),
    'consecutive_bold_lines': re.compile(r'^\*\*[^*]+\*\*:?\s*$'),
    'tick_icon': re.compile(r'^\s*✅\s'),
}


class ListFormatDetector:
    """Main class for detecting list formatting issues in markdown."""

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

    def _get_context_lines(self, line_num: int,
                          context: int = DEFAULT_CONTEXT_LINES) -> List[str]:
        """Get context lines around a specific line number."""
        start = max(0, line_num - context - 1)
        end = min(len(self.lines), line_num + context)
        return [f"{i+1:4d}: {line.rstrip()}"
                for i, line in enumerate(self.lines[start:end], start)]

    def detect_list_missing_blank_line_before(self) -> List[DetectionResult]:
        """Detect lists that don't have a blank line before the first item."""
        issues = []

        for i, line in enumerate(self.lines):
            is_list_item = bool(PATTERNS['list_marker'].match(line))

            if is_list_item and i > 0:
                prev_line = self.lines[i - 1]
                is_prev_blank = bool(PATTERNS['blank_line'].match(prev_line))
                is_prev_list = bool(PATTERNS['list_marker'].match(prev_line))

                # First item of a new list (not part of existing list)
                if not is_prev_blank and not is_prev_list:
                    # Make sure previous line is not empty and contains actual content
                    if prev_line.strip():
                        issues.append(DetectionResult(
                            line_number=i + 1,
                            issue_type=IssueType.LIST_MISSING_BLANK_LINE_BEFORE,
                            content=f"List starts without blank line (previous line: {prev_line.strip()[:50]}...)",
                            context_lines=self._get_context_lines(i + 1)
                        ))

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

    def detect_tick_icons(self) -> List[DetectionResult]:
        """Detect tick icons (✅) used instead of normal list markers."""
        issues = []

        for i, line in enumerate(self.lines):
            if PATTERNS['tick_icon'].match(line):
                issues.append(DetectionResult(
                    line_number=i + 1,
                    issue_type=IssueType.TICK_ICON_IN_LIST,
                    content=f"Tick icon found: {line.strip()[:60]}...",
                    context_lines=self._get_context_lines(i + 1)
                ))

        return issues

    def detect_excessive_blank_lines(self) -> List[DetectionResult]:
        """Detect excessive consecutive blank lines (more than 1)."""
        issues = []
        blank_count = 0
        blank_start_line = -1

        for i, line in enumerate(self.lines):
            is_blank = bool(PATTERNS['blank_line'].match(line))

            if is_blank:
                if blank_count == 0:
                    blank_start_line = i
                blank_count += 1
            else:
                # End of blank sequence
                if blank_count > 1:
                    issues.append(DetectionResult(
                        line_number=blank_start_line + 2,  # Report the second blank line
                        issue_type=IssueType.EXCESSIVE_BLANK_LINES,
                        content=f"{blank_count} consecutive blank lines (should be max 1)",
                        context_lines=self._get_context_lines(blank_start_line + 2)
                    ))
                blank_count = 0

        # Handle trailing blank lines
        if blank_count > 1:
            issues.append(DetectionResult(
                line_number=blank_start_line + 2,
                issue_type=IssueType.EXCESSIVE_BLANK_LINES,
                content=f"{blank_count} consecutive blank lines at end (should be max 1)",
                context_lines=self._get_context_lines(blank_start_line + 2)
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
            (self.detect_list_missing_blank_line_before, "List missing blank line before first item"),
            (self.detect_blank_lines_between_list_items, "Blank lines between list items"),
            (self.detect_consecutive_bold_without_spacing, "Consecutive bold text without spacing"),
            (self.detect_tick_icons, "Tick icons in lists"),
            (self.detect_excessive_blank_lines, "Excessive blank lines"),
        ]

        results = {}
        logger = logging.getLogger(LOGGER_NAME)

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
        all_detectors = {
            "List missing blank line before first item": self.detect_list_missing_blank_line_before,
            "Blank lines between list items": self.detect_blank_lines_between_list_items,
            "Consecutive bold text without spacing": self.detect_consecutive_bold_without_spacing,
            "Tick icons in lists": self.detect_tick_icons,
            "Excessive blank lines": self.detect_excessive_blank_lines,
        }

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
        return [
            "List missing blank line before first item",
            "Blank lines between list items",
            "Consecutive bold text without spacing",
            "Tick icons in lists",
            "Excessive blank lines",
        ]


def get_detector_name_mapping() -> Dict[str, str]:
    """
    Get mapping from detector command names to display names.

    Returns:
        Dictionary mapping command names to display names
    """
    return {
        'list_missing_blank_line_before': 'List missing blank line before first item',
        'blank_lines_between_list_items': 'Blank lines between list items',
        'consecutive_bold_without_spacing': 'Consecutive bold text without spacing',
        'tick_icons': 'Tick icons in lists',
        'excessive_blank_lines': 'Excessive blank lines',
    }


def print_detectors_help() -> None:
    """Print detailed help for all detectors with examples."""
    examples = {
        'list_missing_blank_line_before': {
            'before': 'Some text here\n- First item\n- Second item',
            'after': 'Some text here\n\n- First item\n- Second item'
        },
        'blank_lines_between_list_items': {
            'before': '- First item\n\n- Second item',
            'after': '- First item\n- Second item'
        },
        'consecutive_bold_without_spacing': {
            'before': '**Bold text 1**\n**Bold text 2**',
            'after': '**Bold text 1**\n\n**Bold text 2**'
        },
        'tick_icons': {
            'before': '✅ Item with tick icon\n✅ Another item',
            'after': '- Item with tick icon\n- Another item'
        },
        'excessive_blank_lines': {
            'before': 'Paragraph 1\n\n\n\nParagraph 2',
            'after': 'Paragraph 1\n\nParagraph 2'
        }
    }

    print("Available Detectors:\n")
    name_mapping = get_detector_name_mapping()

    for cmd_name, display_name in name_mapping.items():
        print(f"  {cmd_name}")
        print(f"    Description: {display_name}")
        if cmd_name in examples:
            ex = examples[cmd_name]
            print(f"    Before: {ex['before']}")
            print(f"    After:  {ex['after']}")
        print()


def print_specific_detector_help(detector_name: str) -> None:
    """Print help for a specific detector with examples."""
    name_mapping = get_detector_name_mapping()
    if detector_name not in name_mapping:
        print(f"Unknown detector: {detector_name}")
        print("Use --list-detectors to see available options.")
        return

    display_name = name_mapping[detector_name]

    examples = {
        'list_missing_blank_line_before': {
            'description': 'Finds lists that start without a blank line before the first item',
            'before': 'Some paragraph text\n- First item\n- Second item',
            'after': 'Some paragraph text\n\n- First item\n- Second item',
            'fixes': 'Adds blank line before list start for proper formatting'
        },
        'blank_lines_between_list_items': {
            'description': 'Finds unnecessary blank lines between consecutive list items',
            'before': '- First item\n\n- Second item\n\n- Third item',
            'after': '- First item\n- Second item\n- Third item',
            'fixes': 'Removes blank lines between list items (blank lines should only be before list start)'
        },
        'consecutive_bold_without_spacing': {
            'description': 'Finds consecutive bold text lines without proper blank line spacing',
            'before': '**Section 1:**\n**Section 2:**',
            'after': '**Section 1:**\n\n**Section 2:**',
            'fixes': 'Adds blank line between consecutive bold text for better readability'
        },
        'tick_icons': {
            'description': 'Finds tick icons (✅) used instead of normal list markers',
            'before': '✅ First requirement\n✅ Second requirement',
            'after': '- First requirement\n- Second requirement',
            'fixes': 'Replaces tick icons with standard markdown list markers for proper rendering'
        },
        'excessive_blank_lines': {
            'description': 'Finds excessive consecutive blank lines (more than 1)',
            'before': 'Paragraph 1\n\n\n\nParagraph 2',
            'after': 'Paragraph 1\n\nParagraph 2',
            'fixes': 'Removes excessive blank lines, keeping maximum of 1 blank line between content'
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
        description='List Formatting Detector for Markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="""
Usage Patterns:
  %(prog)s document.md                           # Run all list format detectors
  %(prog)s --list-detectors                      # Show all detector names
  %(prog)s --detector list_missing_blank_line_before document.md  # Run specific detector
  %(prog)s --verbose document.md                 # Run with verbose logging

Help for Detectors:
  %(prog)s --list-detectors --help               # Show detector examples
  %(prog)s --detector list_missing_blank_line_before --help  # Show specific detector help
""")

    parser.add_argument('filepath', nargs='?', type=Path,
                       help='Path to the markdown file to analyze')

    parser.add_argument('--list-detectors', action='store_true',
                       help='List all available detector names')

    parser.add_argument('--detector', type=str, metavar='NAME',
                       help='Run a specific individual detector')

    parser.add_argument('--help', '-h', action='store_true',
                       help='Show this help message and exit, or show examples when used with --list-detectors or --detector')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    return parser


def get_selected_detector_by_name(detector_name: str) -> List[str]:
    """
    Get a specific detector by name.

    Args:
        detector_name: Name of specific detector

    Returns:
        List containing single detector name if found, empty list otherwise
    """
    name_mapping = get_detector_name_mapping()

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

    # Handle general --help (when no other options are provided)
    if args.help and not (args.list_detectors or args.detector):
        parser.print_help()
        return EXIT_SUCCESS

    # Handle --list-detectors
    if args.list_detectors:
        if args.help:
            print_detectors_help()
        else:
            name_mapping = get_detector_name_mapping()
            print("Available Detectors:")
            for cmd_name in name_mapping:
                print(f"  {cmd_name}")
            print("\nUse --list-detectors --help for examples")
        return EXIT_SUCCESS

    # Handle --detector with --help
    if args.detector and args.help:
        print_specific_detector_help(args.detector)
        return EXIT_SUCCESS

    # Validate filepath for actual detection
    if not args.filepath:
        if args.detector:
            parser.error("filepath is required when using --detector without --help")
        else:
            parser.error("filepath is required")

    if not args.filepath.exists():
        logger.error("File %s does not exist", args.filepath)
        return EXIT_ERROR

    try:
        detector = ListFormatDetector(args.filepath)

        # Determine which detectors to run
        selected_detectors = []
        operation_name = "all list format detectors"

        if args.detector:
            selected_detectors = get_selected_detector_by_name(args.detector)
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
            logger.info("Running all list format detectors")
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
