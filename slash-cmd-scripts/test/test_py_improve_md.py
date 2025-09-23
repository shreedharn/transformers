#!/usr/bin/env python3
"""
Comprehensive test suite for py-improve-md.py markdown detector.

This module provides exhaustive testing for all detector categories and individual
detectors, ensuring robust validation of markdown and LaTeX formatting detection.

Test Categories:
- Unit tests for all individual detector methods
- Integration tests for category-based detection
- CLI argument parsing and help system validation
- Edge cases and error handling
- Performance benchmarks for large documents
"""

import sys
import unittest
import tempfile
import io
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from py_improve_md import (
    MarkdownMathDetector,
    DetectorCategory,
    IssueType,
    DetectionResult,
    LineContext,
    create_argument_parser,
    get_category_mapping,
    get_detector_name_mapping,
    get_selected_detectors_by_category,
    get_selected_detector_by_name,
    print_categories_help,
    print_detectors_help,
    print_category_help,
    print_specific_detector_help,
    main
)


class TestMarkdownMathDetector(unittest.TestCase):
    """Comprehensive test suite for MarkdownMathDetector class."""

    def setUp(self):
        """Set up test fixtures with temporary files."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.md"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        self.test_dir.rmdir()

    def create_test_file(self, content: str) -> MarkdownMathDetector:
        """Helper to create test file and return detector instance."""
        self.test_file.write_text(content, encoding='utf-8')
        return MarkdownMathDetector(self.test_file)

    def test_file_loading(self):
        """Test basic file loading and initialization."""
        content = "# Test Document\n\nSample content."
        detector = self.create_test_file(content)

        self.assertEqual(len(detector.lines), 3)
        self.assertEqual(detector.lines[0], "# Test Document")
        self.assertEqual(detector.lines[2], "Sample content.")

    def test_empty_file(self):
        """Test handling of empty files."""
        detector = self.create_test_file("")
        self.assertEqual(len(detector.lines), 0)

    def test_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with self.assertRaises(FileNotFoundError):
            MarkdownMathDetector(Path("nonexistent.md"))


class TestInlineMathDetectors(unittest.TestCase):
    """Test suite for inline math detection category (5 detectors)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.md"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        self.test_dir.rmdir()

    def create_detector(self, content: str) -> MarkdownMathDetector:
        """Helper to create detector with test content."""
        self.test_file.write_text(content, encoding='utf-8')
        return MarkdownMathDetector(self.test_file)

    def test_inline_math_in_paragraphs_detection(self):
        """Test detection of mathematical variables mixed in prose."""
        content = """# Test Document

The variable x_1 = 5 shows inline math in paragraph text.
Another paragraph with equation y = mx + b should be detected.

This paragraph has no math and should be ignored.
"""
        detector = self.create_detector(content)
        results = detector.detect_inline_math_in_paragraphs()

        self.assertGreater(len(results), 0)
        self.assertTrue(any("x_1" in result.content for result in results))

    def test_inline_math_in_lists_detection(self):
        """Test detection of inline math in list items (forbidden)."""
        content = """# Mathematical Lists

- Item with math: $x = 5$ should be detected
- Another item: $y = mx + b$
- Normal item without math

1. Numbered item with $alpha = 0.01$
2. Normal numbered item
"""
        detector = self.create_detector(content)
        results = detector.detect_inline_math_in_lists()

        self.assertGreater(len(results), 0)
        # Check that math in list items is detected
        math_in_list_detected = any("$x = 5$" in result.content for result in results)
        self.assertTrue(math_in_list_detected)

    def test_inline_math_in_headings_detection(self):
        """Test detection of math in headings (forbidden)."""
        content = """# Main Title

## Section with $x = 5$ in heading
### Another section with $$y = mx + b$$
#### Normal heading without math

Regular paragraph content.
"""
        detector = self.create_detector(content)
        results = detector.detect_inline_math_in_headings()

        self.assertGreater(len(results), 0)
        self.assertTrue(any("$x = 5$" in result.content for result in results))

    def test_display_math_delimiters_used_inline_detection(self):
        """Test detection of display math used incorrectly inline."""
        content = """# Document

This paragraph has $$x = 5$$ display math used inline in prose.
Another case with $$\\alpha + \\beta = \\gamma$$ in sentence.

$$
\\begin{aligned}
x &= 5 \\newline
y &= 10
\\end{aligned}
$$

This is correct usage above.
"""
        detector = self.create_detector(content)
        results = detector.detect_inline_display_math_in_prose()

        self.assertGreater(len(results), 0)
        inline_display_detected = any("$$x = 5$$" in result.content for result in results)
        self.assertTrue(inline_display_detected)

    def test_math_tokens_in_prose_detection(self):
        """Test detection of mathematical tokens scattered in text."""
        content = """# Analysis

The function parameters W, b, and activation σ are important.
We use variables like α, β, γ in the equations.
The complexity is O(n²) for this algorithm.

Normal text without mathematical symbols.
"""
        detector = self.create_detector(content)
        results = detector.detect_math_tokens_in_prose()

        self.assertGreater(len(results), 0)


class TestDisplayMathDetectors(unittest.TestCase):
    """Test suite for display math detection category (8 detectors)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.md"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        self.test_dir.rmdir()

    def create_detector(self, content: str) -> MarkdownMathDetector:
        """Helper to create detector with test content."""
        self.test_file.write_text(content, encoding='utf-8')
        return MarkdownMathDetector(self.test_file)

    def test_list_marker_lines_with_display_math_detection(self):
        """Test detection of display math directly in list markers."""
        content = """# Mathematical Lists

- $$x = 5$$ equation directly in bullet
* $$y = mx + b$$ another equation in bullet
- Normal bullet point

1. $$\\alpha = 0.01$$ in numbered item
2. Normal numbered item
"""
        detector = self.create_detector(content)
        results = detector.detect_list_with_display_math()

        self.assertGreater(len(results), 0)
        self.assertTrue(any("$$x = 5$$" in result.content for result in results))

    def test_heading_lines_with_math_detection(self):
        """Test detection of math expressions in heading lines."""
        content = """# Main Document

## Section $$x = 5$$ with math
### Analysis of $$f(x) = x^2$$
#### Normal heading

Content here.
"""
        detector = self.create_detector(content)
        results = detector.detect_heading_with_math()

        self.assertGreater(len(results), 0)

    def test_table_rows_with_display_math_detection(self):
        """Test detection of display math in table rows."""
        content = """# Tables

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Normal   | $$x = 5$$ | Value   |
| Data     | $$y = 10$$ | More   |
| Regular  | Content   | Here    |
"""
        detector = self.create_detector(content)
        results = detector.detect_table_with_display_math()

        self.assertGreater(len(results), 0)

    def test_display_math_with_list_marker_detection(self):
        """Test detection of list markers mixed with display math."""
        content = """# Mixed Content

$$
- x = 5
$$

$$
\\begin{aligned}
1. y &= mx + b \\newline
2. z &= a + b
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_display_math_with_list_marker()

        self.assertGreater(len(results), 0)

    def test_over_indented_display_math_detection(self):
        """Test detection of display math with excessive indentation."""
        content = """# Indentation Issues

Normal paragraph.

    $$
    \\begin{aligned}
    x = 5
    \\end{aligned}
    $$

        $$y = 10$$

Correct indentation:

$$
\\begin{aligned}
z = 15
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_over_indented_display_math()

        self.assertGreater(len(results), 0)

    def test_adjacent_math_blocks_detection(self):
        """Test detection of adjacent math blocks that should be consolidated."""
        content = """# Adjacent Blocks

$$
\\begin{aligned}
x = 5
\\end{aligned}
$$

$$
\\begin{aligned}
y = 10
\\end{aligned}
$$

This text breaks the adjacency.

$$
\\begin{aligned}
z = 15
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_adjacent_math_blocks()

        self.assertGreater(len(results), 0)

    def test_math_with_text_on_same_line_detection(self):
        """Test detection of math expressions with text on same line."""
        content = """# Same Line Issues

$$x = 5$$ This text is on same line as math
Here is text $$y = 10$$ more text on same line

$$
\\begin{aligned}
z = 15
\\end{aligned}
$$ Text after closing delimiter

Proper usage:

$$
\\begin{aligned}
w = 20
\\end{aligned}
$$

Text on separate line.
"""
        detector = self.create_detector(content)
        results = detector.detect_math_with_text_on_same_line()

        self.assertGreater(len(results), 0)

    def test_math_in_blockquotes_detection(self):
        """Test detection of mathematical expressions in blockquotes."""
        content = """# Blockquotes

> This is a quote with $$x = 5$$ math inside
> Another line with $$y = mx + b$$

> Normal blockquote without math

> Multiple line quote
> with $$\\alpha + \\beta = \\gamma$$ math
> on second line
"""
        detector = self.create_detector(content)
        results = detector.detect_math_in_blockquotes()

        self.assertGreater(len(results), 0)


class TestListFormattingDetectors(unittest.TestCase):
    """Test suite for list formatting detection category (3 detectors)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.md"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        self.test_dir.rmdir()

    def create_detector(self, content: str) -> MarkdownMathDetector:
        """Helper to create detector with test content."""
        self.test_file.write_text(content, encoding='utf-8')
        return MarkdownMathDetector(self.test_file)

    def test_list_missing_blank_line_before_math_detection(self):
        """Test detection of lists missing blank line before math blocks."""
        content = """# List Formatting

- Item one
- Item two
$$
\\begin{aligned}
x = 5
\\end{aligned}
$$

Correct usage:

- Item one
- Item two

$$
\\begin{aligned}
y = 10
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_list_missing_blank_line()

        self.assertGreater(len(results), 0)

    def test_blank_lines_between_list_items_detection(self):
        """Test detection of incorrect blank lines between list items."""
        content = """# List Spacing

Correct list:
- Item one
- Item two
- Item three

Incorrect list with blank lines:
- Item one

- Item two

- Item three

Mixed spacing:
1. First item
2. Second item

3. Third item with blank line before
"""
        detector = self.create_detector(content)
        results = detector.detect_blank_lines_between_list_items()

        self.assertGreater(len(results), 0)

    def test_missing_blank_line_before_list_items_after_text_detection(self):
        """Test detection of lists immediately following text without blank line."""
        content = """# List Separation

Text paragraph ending with period.
- List item immediately follows (wrong)
- Another item

Correct usage:

Text paragraph ending with colon:

- List item with proper spacing
- Another item

**Bold text followed by list:**
1. Missing blank line
2. Another item

**Correct bold text:**

1. Proper blank line
2. Another item
"""
        detector = self.create_detector(content)
        results = detector.detect_text_to_list_missing_blank_line()

        self.assertGreater(len(results), 0)


class TestAlignmentDetectors(unittest.TestCase):
    """Test suite for alignment detection category (5 detectors)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.md"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        self.test_dir.rmdir()

    def create_detector(self, content: str) -> MarkdownMathDetector:
        """Helper to create detector with test content."""
        self.test_file.write_text(content, encoding='utf-8')
        return MarkdownMathDetector(self.test_file)

    def test_backslashes_in_aligned_detection(self):
        """Test detection of incorrect backslashes in aligned blocks."""
        content = """# Alignment Issues

$$
\\begin{aligned}
x &= 5 \\\\
y &= 10 \\\\
z &= 15
\\end{aligned}
$$

Correct usage:

$$
\\begin{aligned}
x &= 5 \\newline
y &= 10 \\newline
z &= 15
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_backslashes_in_aligned()

        self.assertGreater(len(results), 0)

    def test_math_missing_aligned_detection(self):
        """Test detection of math blocks missing aligned structure."""
        content = """# Missing Aligned Structure

$$x = 5$$

$$
x = 5
$$

$$y = mx + b$$

Correct usage:

$$
\\begin{aligned}
x = 5
\\end{aligned}
$$

$$
\\begin{aligned}
y = mx + b
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_math_missing_aligned()

        self.assertGreater(len(results), 0)

    def test_end_aligned_with_trailing_text_detection(self):
        """Test detection of \\end{aligned} with text on same line."""
        content = """# End Aligned Issues

$$
\\begin{aligned}
x &= 5 \\newline
y &= 10
\\end{aligned} trailing text
$$

$$
\\begin{aligned}
z &= 15
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_end_aligned_with_trailing_text()

        self.assertGreater(len(results), 0)

    def test_mismatched_aligned_blocks_detection(self):
        """Test detection of mismatched begin/end aligned blocks."""
        content = """# Mismatched Blocks

$$
\\begin{aligned}
x = 5
\\end{align}
$$

$$
\\begin{align}
y = 10
\\end{aligned}
$$

$$
\\begin{aligned}
z = 15
$$

Correct usage:

$$
\\begin{aligned}
w = 20
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_mismatched_aligned_blocks()

        self.assertGreater(len(results), 0)

    def test_end_before_begin_aligned_detection(self):
        """Test detection of \\end{aligned} appearing before \\begin{aligned}."""
        content = """# Order Issues

$$
\\end{aligned}
x = 5
\\begin{aligned}
$$

$$
y = 10
\\end{aligned}
\\begin{aligned}
z = 15
\\end{aligned}
$$

Correct order:

$$
\\begin{aligned}
w = 20
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_end_before_begin_aligned()

        self.assertGreater(len(results), 0)


class TestStructuralDetectors(unittest.TestCase):
    """Test suite for structural detection category (2 detectors)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.md"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        self.test_dir.rmdir()

    def create_detector(self, content: str) -> MarkdownMathDetector:
        """Helper to create detector with test content."""
        self.test_file.write_text(content, encoding='utf-8')
        return MarkdownMathDetector(self.test_file)

    def test_consecutive_bold_text_without_spacing_detection(self):
        """Test detection of consecutive bold text without proper spacing."""
        content = """# Bold Text Spacing

**First bold text**
**Second bold text** immediately follows

**Correct first bold**

**Correct second bold** with proper spacing

**Another issue** **Right next to each other**
"""
        detector = self.create_detector(content)
        results = detector.detect_consecutive_bold_text_without_spacing()

        self.assertGreater(len(results), 0)

    def test_escaped_underscores_in_code_blocks_detection(self):
        """Test detection of escaped underscores in code blocks (wrong context)."""
        content = """# Code Block Issues

```python
def my\\_function(param\\_name):
    return param\\_name * 2

class My\\_Class:
    def \\_init\\_\\_self):
        pass
```

Correct code:

```python
def my_function(param_name):
    return param_name * 2

class My_Class:
    def __init__(self):
        pass
```

Math context (escaping correct):

$$x\\_1 = 5$$
"""
        detector = self.create_detector(content)
        results = detector.detect_escaped_underscores_in_code_blocks()

        self.assertGreater(len(results), 0)


class TestSyntaxDetectors(unittest.TestCase):
    """Test suite for syntax detection category (3 detectors)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.md"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        self.test_dir.rmdir()

    def create_detector(self, content: str) -> MarkdownMathDetector:
        """Helper to create detector with test content."""
        self.test_file.write_text(content, encoding='utf-8')
        return MarkdownMathDetector(self.test_file)

    def test_unpaired_math_delimiters_detection(self):
        """Test detection of unpaired mathematical delimiters."""
        content = """# Unpaired Delimiters

Text with $$ unpaired opening delimiter

Text with unpaired closing $$ delimiter

$$
\\begin{aligned}
x = 5
\\end{aligned}

Missing closing delimiter.

Correct usage:

$$
\\begin{aligned}
y = 10
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_unpaired_math_delimiters()

        self.assertGreater(len(results), 0)

    def test_stray_braces_with_math_detection(self):
        """Test detection of stray braces around mathematical expressions."""
        content = """# Stray Braces

Text with {$$x = 5$$} stray braces around math
Another case {$$y = mx + b$$} with braces

{
$$
\\begin{aligned}
z = 15
\\end{aligned}
$$
}

Correct usage without braces:

$$
\\begin{aligned}
w = 20
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_stray_braces_with_math()

        self.assertGreater(len(results), 0)

    def test_bold_markdown_in_math_detection(self):
        """Test detection of bold markdown mixed within LaTeX expressions."""
        content = """# Bold in Math

$$
\\begin{aligned}
**x** &= 5 \\newline
y &= **bold text** + 10
\\end{aligned}
$$

$$**z** = 15$$

Correct usage:

$$
\\begin{aligned}
\\mathbf{x} &= 5 \\newline
y &= \\text{regular text} + 10
\\end{aligned}
$$
"""
        detector = self.create_detector(content)
        results = detector.detect_bold_markdown_in_math()

        self.assertGreater(len(results), 0)


class TestDetectorCategories(unittest.TestCase):
    """Test suite for detector categorization and grouping."""

    def test_category_mapping(self):
        """Test that category mapping returns correct enum values."""
        categories = get_category_mapping()

        self.assertIn('inline_math', categories)
        self.assertIn('display_math', categories)
        self.assertIn('list_formatting', categories)
        self.assertIn('alignment', categories)
        self.assertIn('structural', categories)
        self.assertIn('syntax', categories)

        self.assertEqual(categories['inline_math'], DetectorCategory.INLINE_MATH)
        self.assertEqual(categories['display_math'], DetectorCategory.DISPLAY_MATH)

    def test_detector_groups(self):
        """Test that detector groups are properly organized."""
        test_file = Path(tempfile.mktemp(suffix='.md'))
        test_file.write_text("# Test")

        try:
            detector = MarkdownMathDetector(test_file)
            groups = detector.get_detector_groups()

            # Verify all categories are present
            self.assertIn(DetectorCategory.INLINE_MATH, groups)
            self.assertIn(DetectorCategory.DISPLAY_MATH, groups)
            self.assertIn(DetectorCategory.LIST_FORMATTING, groups)
            self.assertIn(DetectorCategory.ALIGNMENT, groups)
            self.assertIn(DetectorCategory.STRUCTURAL, groups)
            self.assertIn(DetectorCategory.SYNTAX, groups)

            # Verify correct number of detectors per category
            self.assertEqual(len(groups[DetectorCategory.INLINE_MATH]), 5)
            self.assertEqual(len(groups[DetectorCategory.DISPLAY_MATH]), 8)
            self.assertEqual(len(groups[DetectorCategory.LIST_FORMATTING]), 3)
            self.assertEqual(len(groups[DetectorCategory.ALIGNMENT]), 5)
            self.assertEqual(len(groups[DetectorCategory.STRUCTURAL]), 2)
            self.assertEqual(len(groups[DetectorCategory.SYNTAX]), 3)

        finally:
            if test_file.exists():
                test_file.unlink()

    def test_selected_detectors_by_category(self):
        """Test category-based detector selection."""
        test_file = Path(tempfile.mktemp(suffix='.md'))
        test_file.write_text("# Test")

        try:
            detector = MarkdownMathDetector(test_file)

            inline_math_detectors = get_selected_detectors_by_category('inline_math', detector)
            self.assertEqual(len(inline_math_detectors), 5)

            display_math_detectors = get_selected_detectors_by_category('display_math', detector)
            self.assertEqual(len(display_math_detectors), 8)

            # Test invalid category
            invalid_detectors = get_selected_detectors_by_category('invalid_category', detector)
            self.assertEqual(len(invalid_detectors), 0)

        finally:
            if test_file.exists():
                test_file.unlink()

    def test_selected_detector_by_name(self):
        """Test individual detector selection by name."""
        test_file = Path(tempfile.mktemp(suffix='.md'))
        test_file.write_text("# Test")

        try:
            detector = MarkdownMathDetector(test_file)

            # Test valid detector name
            detectors = get_selected_detector_by_name('missing_blank_line_before_list_items_after_text', detector)
            self.assertEqual(len(detectors), 1)

            # Test invalid detector name
            invalid_detectors = get_selected_detector_by_name('invalid_detector', detector)
            self.assertEqual(len(invalid_detectors), 0)

        finally:
            if test_file.exists():
                test_file.unlink()


class TestCLIFunctionality(unittest.TestCase):
    """Test suite for command-line interface functionality."""

    def test_argument_parser_creation(self):
        """Test argument parser creation and configuration."""
        parser = create_argument_parser()

        # Test that parser accepts expected arguments
        args = parser.parse_args(['test.md'])
        self.assertEqual(str(args.filepath), 'test.md')

        args = parser.parse_args(['--list-categories'])
        self.assertTrue(args.list_categories)

        args = parser.parse_args(['--list-detectors'])
        self.assertTrue(args.list_detectors)

        args = parser.parse_args(['--category', 'inline_math', 'test.md'])
        self.assertEqual(args.category, 'inline_math')

        args = parser.parse_args(['--detector', 'test_detector', 'test.md'])
        self.assertEqual(args.detector, 'test_detector')

        args = parser.parse_args(['--verbose', 'test.md'])
        self.assertTrue(args.verbose)

    def test_help_functions(self):
        """Test help output functions."""
        # Test categories help
        with redirect_stdout(io.StringIO()) as output:
            print_categories_help()
        help_output = output.getvalue()
        self.assertIn('inline_math', help_output)
        self.assertIn('display_math', help_output)
        self.assertIn('Before:', help_output)
        self.assertIn('After:', help_output)

    def test_category_help_function(self):
        """Test individual category help function."""
        with redirect_stdout(io.StringIO()) as output:
            print_category_help('inline_math')
        help_output = output.getvalue()
        self.assertIn('Category: inline_math', help_output)
        self.assertIn('Description:', help_output)
        self.assertIn('Before:', help_output)
        self.assertIn('After:', help_output)

    @patch('sys.argv', ['py-improve-md.py', '--help'])
    def test_main_help(self):
        """Test main function with help argument."""
        with redirect_stdout(io.StringIO()) as output:
            try:
                main()
            except SystemExit:
                pass  # Help command exits normally

        help_output = output.getvalue()
        self.assertIn('usage:', help_output)
        self.assertIn('--list-categories', help_output)

    @patch('sys.argv', ['py-improve-md.py', '--list-categories'])
    def test_main_list_categories(self):
        """Test main function with list categories."""
        with redirect_stdout(io.StringIO()) as output:
            result = main()

        self.assertEqual(result, 0)
        categories_output = output.getvalue()
        self.assertIn('Available Categories:', categories_output)
        self.assertIn('inline_math', categories_output)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test suite for edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.md"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        self.test_dir.rmdir()

    def test_unicode_content_handling(self):
        """Test handling of Unicode mathematical symbols."""
        content = """# Unicode Math

Mathematical symbols: α, β, γ, δ, ε, ζ, η, θ
Greek letters: Α, Β, Γ, Δ, Ε, Ζ, Η, Θ
Special symbols: ∑, ∏, ∫, ∂, ∇, ∞, ±, ≤, ≥, ≠

The variable α should be detected in this sentence.
"""
        self.test_file.write_text(content, encoding='utf-8')
        detector = MarkdownMathDetector(self.test_file)

        # Should not raise encoding errors
        results = detector.run_all_detectors()
        self.assertIsInstance(results, dict)

    def test_very_long_lines_handling(self):
        """Test handling of very long lines."""
        long_line = "Very long mathematical expression with variables: " + " + ".join(f"x_{i}" for i in range(1000))
        content = f"# Long Lines\n\n{long_line}\n"

        self.test_file.write_text(content, encoding='utf-8')
        detector = MarkdownMathDetector(self.test_file)

        # Should handle long lines without issues
        results = detector.detect_math_tokens_in_prose()
        self.assertIsInstance(results, list)

    def test_deeply_nested_structures(self):
        """Test handling of deeply nested markdown structures."""
        content = """# Nested Structures

> Quote level 1
> > Quote level 2
> > > Quote level 3 with $$x = 5$$
> > > > Quote level 4
> > > > - List in deep quote
> > > > - Another item with $y = 10$

- List level 1
  - List level 2
    - List level 3 with $$z = 15$$
      - List level 4
"""
        self.test_file.write_text(content, encoding='utf-8')
        detector = MarkdownMathDetector(self.test_file)

        results = detector.run_all_detectors()
        self.assertIsInstance(results, dict)

    def test_malformed_latex_handling(self):
        """Test handling of malformed LaTeX expressions."""
        content = """# Malformed LaTeX

$$
\\begin{aligned}
x = 5
\\end{alignd}  # Typo in end tag
$$

$$
\\begin{align}
y = 10
\\end{aligned}  # Mismatched tags
$$

$$
\\begin{aligned
z = 15  # Missing closing brace
\\end{aligned}
$$
"""
        self.test_file.write_text(content, encoding='utf-8')
        detector = MarkdownMathDetector(self.test_file)

        # Should detect malformed structures without crashing
        results = detector.detect_mismatched_aligned_blocks()
        self.assertIsInstance(results, list)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test suite for performance benchmarks on large documents."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        for file in self.test_dir.glob("*.md"):
            file.unlink()
        self.test_dir.rmdir()

    def test_large_document_performance(self):
        """Test performance on large documents (10,000+ lines)."""
        import time

        # Generate large document
        lines = []
        for i in range(10000):
            if i % 100 == 0:
                lines.append(f"## Section {i//100}")
            elif i % 10 == 0:
                lines.append(f"- List item {i} with variable x_{i}")
            elif i % 5 == 0:
                lines.append(f"Paragraph {i} with equation y_{i} = mx + b.")
            else:
                lines.append(f"Regular text line {i} without mathematical content.")

        large_file = self.test_dir / "large.md"
        large_file.write_text('\n'.join(lines), encoding='utf-8')

        # Time the detection process
        start_time = time.time()
        detector = MarkdownMathDetector(large_file)
        results = detector.run_all_detectors()
        end_time = time.time()

        # Performance assertions
        processing_time = end_time - start_time
        self.assertLess(processing_time, 30.0)  # Should complete within 30 seconds
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

    def test_regex_performance(self):
        """Test regex pattern performance on pathological cases."""
        import time

        # Create document with many mathematical expressions
        math_expressions = []
        for i in range(1000):
            math_expressions.append(f"Variable x_{i} with equation y_{i} = a_{i}*x + b_{i}")

        content = "# Performance Test\n\n" + "\n\n".join(math_expressions)
        perf_file = self.test_dir / "performance.md"
        perf_file.write_text(content, encoding='utf-8')

        # Time regex-heavy detector
        start_time = time.time()
        detector = MarkdownMathDetector(perf_file)
        results = detector.detect_math_tokens_in_prose()
        end_time = time.time()

        processing_time = end_time - start_time
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
        self.assertIsInstance(results, list)


if __name__ == '__main__':
    # Configure test runner with verbose output
    unittest.main(verbosity=2)