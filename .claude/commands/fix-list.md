---
argument-hint: [filename]
description: Fixes markdown files by ensuring there is a blank line before every markdown list
allowed-tools: Read, Edit, MultiEdit, Bash,Bash(grep:*), Bash(sed:*), Bash(awk:*)
---

Fixes markdown files by ensuring there is a blank line before every markdown list (both bulleted and numbered lists). This improves readability and ensures proper markdown rendering.

## What it fixes
- Adds missing blank lines before bulleted lists (`-`, `*`, `+`)
- Adds missing blank lines before numbered lists (`1.`, `2.`, etc.)
- Blank lines should not be added between bullets but before the first bullet and after the last bullet
- Ensures proper separation between prose/other content and list items
- Preserves existing formatting and content

## Examples

### Before (Incorrect)
where:
- Query matrix (what information to retrieve)
- Key matrix (what information is available)

### After (Corrected)
where:

- Query matrix (what information to retrieve)
- Key matrix (what information is available)

### Before (Incorrect)

**Implementation Details:**
1. **Parallel computation**: All heads computed simultaneously
2. **Linear projections**: Simple matrix multiplications


### After (Corrected)

**Implementation Details:**

1. **Parallel computation**: All heads computed simultaneously
2. **Linear projections**: Simple matrix multiplications


## Comprehensive List Formatting Detector

**Single unified Python detector** that runs detection patterns 

```bash
python3 cmd-scripts/py-fix-list.py "$1"
```

This comprehensive detector finds markdown list formatting issues and presents them in clearly segmented sections:

**Detection Categories:**
1. **Lists starting immediately after prose without blank line (primary issue)**
2. **Lists following colons without separation (common pattern)**
3. **Lists after bold/italic markdown without separation**
4. **Lists after parenthetical statements without blank line**
5. **Lists after inline code or math expressions**
6. **Numbered lists missing blank line separator**
7. **Bulleted lists missing blank line separator**
8. **Unescaped underscores in LaTeX blocks (critical for Markdown compatibility)**
9. **Markdown bullets containing LaTeX expressions (Rule 1 violation)**
10. **Mixed LaTeX notation - should use consistent format**
11. **Consecutive LaTeX blocks that could be consolidated**
12. **Typography standard violations (bold, bullets, spacing)**
13. **Lists after blockquotes missing separation**
14. **Lists following table rows without separation**
15. **Lists after HTML blocks without separation**
16. **Lists after code fence blocks**
17. **Lists after MathJax display blocks**
18. **Inconsistent bullet markers (mixed -, *, +)**

**Unified Detector Benefits:**
- **Single execution**: All patterns checked in one pass for maximum efficiency
- **Consistent output**: Grouped results by detection type with clear separators (`---`)
- **Complete context**: Shows line numbers, issue counts, and surrounding context
- **Performance optimized**: Compiled regex patterns and efficient algorithm
- **Maintainable**: All detection logic in one Python file vs scattered grep commands
- **Extensible**: Easy to add new detection patterns or modify existing ones

## Implementation

Execute the Python detector to analyze the markdown file:

```bash
python3 cmd-scripts/py-fix-list.py "$1"
```

This will provide a comprehensive report of all markdown list formatting issues organized by category, with line numbers and context for easy identification and fixing. After running the Python detector, perform a final AI scan to catch any edge cases or patterns the automated detection might miss.

## Detection Strategy

The unified Python detector uses optimized pattern matching to identify:

- **Primary patterns**: Non-blank lines followed immediately by list markers
- **Context awareness**: Distinguishes between acceptable cases (after headings) and problematic ones
- **LaTeX compatibility**: Critical underscore escaping and bullet/LaTeX separation issues
- **List continuation**: Preserves existing list structure by not flagging blanks between list items
- **Edge cases**: Handles special scenarios like nested lists, blockquotes, tables, code blocks, and MathJax
- **Typography standards**: Detects mixed notation, inconsistent formatting, and styling issues
