# Markdown Math Detector

A comprehensive Python tool for detecting and analyzing markdown formatting and LaTeX mathematical notation issues.

## Directory Structure

```
slash-cmd-scripts/
├── src/
│   └── py_improve_md.py    # Main detector implementation
├── test/
│   ├── __init__.py
│   ├── test_py_improve_md.py    # Comprehensive test suite
│   └── run_tests.py            # Test runner with category support
└── README.md
```

## Usage

### Run the detector

```bash
# Run all detectors on a document
python3 src/py_improve_md.py document.md

# Run specific category
python3 src/py_improve_md.py --category inline_math document.md

# Run specific detector
python3 src/py_improve_md.py --detector missing_blank_line_before_list_items_after_text document.md

# Get help
python3 src/py_improve_md.py --help
python3 src/py_improve_md.py --list-categories --help
python3 src/py_improve_md.py --category inline_math --help
```

## Testing

### Run all tests
```bash
cd slash-cmd-scripts
python3 test/run_tests.py
```

### Run specific test categories
```bash
# Run inline math detector tests
python3 test/run_tests.py --category inline_math

# Run CLI functionality tests
python3 test/run_tests.py --category cli

# Run performance benchmarks
python3 test/run_tests.py --category performance
```

### Available test categories
- `inline_math` - Tests for inline math detection (5 detectors)
- `display_math` - Tests for display math detection (8 detectors)
- `list_formatting` - Tests for list formatting detection (3 detectors)
- `alignment` - Tests for LaTeX alignment detection (5 detectors)
- `structural` - Tests for structural issues detection (2 detectors)
- `syntax` - Tests for syntax error detection (3 detectors)
- `cli` - Tests for command-line interface functionality
- `edge_cases` - Tests for edge cases and error handling
- `performance` - Performance benchmark tests

## Detector Categories

### 1. Inline Math (5 detectors)
- Variables and equations mixed with prose text
- Inline math in headings and lists (forbidden)
- Display math delimiters used incorrectly in prose
- Mathematical tokens scattered in descriptive text

### 2. Display Math (8 detectors)
- Math expressions in wrong locations (lists, headings, tables)
- Math blocks with incorrect positioning and indentation
- Adjacent math blocks needing consolidation
- Over-indented display math and improper alignment

### 3. List Formatting (3 detectors)
- Missing blank lines before lists after various content types
- Incorrect spacing between list items
- Lists appearing without proper separation

### 4. Alignment (5 detectors)
- Math expressions missing professional `\begin{aligned}` structure
- Mismatched and malformed aligned blocks
- Single-line equations without proper LaTeX wrapper
- Incorrect positioning of alignment statements

### 5. Structural (2 detectors)
- Bold text spacing issues and missing blank lines
- Escaped underscores in code blocks (breaks syntax)

### 6. Syntax (3 detectors)
- Unpaired mathematical delimiters and stray braces
- Bold markdown mixed within LaTeX expressions
- Context-specific underscore escaping violations

## Development

The test suite includes:
- **Unit tests** for all 26 individual detectors
- **Integration tests** for category-based detection
- **CLI functionality tests** with argument parsing validation
- **Edge case tests** including Unicode, long lines, nested structures
- **Performance benchmarks** for large documents (10,000+ lines)
- **Error handling tests** for malformed inputs

Run tests with verbose output:
```bash
python3 test/run_tests.py --verbose
```

Run specific detector category tests:
```bash
python3 test/run_tests.py --category inline_math --verbose
```