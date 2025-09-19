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

**Optimized Python-based detector** that runs all detection patterns in a single execution:

```bash
python3 cmd-scripts/py-fix-list.py "$1"
```

This comprehensive detector finds all the following issues and presents them in segmented sections:

**Detection Categories:**
1. **Lists starting immediately after prose without blank line (primary issue)**
2. **Lists following colons without separation (common pattern)**
3. **Lists after bold/italic markdown without separation**
4. **Lists after parenthetical statements without blank line**
5. **Lists after inline code or math expressions**
6. **Numbered lists missing blank line separator**
7. **Bulleted lists missing blank line separator**

The detector automatically:
- Analyzes the entire file in a single pass
- Groups results by detection type with clear separators (`---`)
- Shows line numbers, issue counts, and context for each problem
- Provides a summary with total issues found
- Handles all edge cases (headers, existing blank lines, nested lists)
- Uses optimized regex patterns compiled at startup for performance

## AI Verification Step

After running the Python detector, perform a final AI scan to catch any edge cases or patterns the automated detection might miss:

```bash
echo "=== PYTHON DETECTOR RESULTS ==="
python3 cmd-scripts/py-fix-list.py "$1"

echo ""
echo "=== AI VERIFICATION SCAN ==="
echo "Performing intelligent review of markdown list formatting..."
```

**Manual AI Review Guidelines:**
1. **Verify Python results** - Check if detected issues are legitimate formatting problems
2. **Scan for missed patterns** - Look for unusual list contexts not covered by regex patterns
3. **Check list consistency** - Ensure bullet style consistency throughout the document
4. **Validate context** - Confirm that adding blank lines would improve readability
5. **Edge case detection** - Find complex nested structures or unusual markdown combinations
6. **False positive filtering** - Identify any incorrectly flagged legitimate formatting

The AI verification ensures comprehensive coverage by combining automated pattern detection with intelligent contextual analysis of the original objective: "ensuring there is a blank line before every markdown list."

8. **Lists in blockquotes missing separation**
```bash
grep -nE -C20 '^>\s*[^-*+0-9\s]' "$1" | grep -A1 '^>\s*([-*+]|[0-9]+\.)\s'
```

9. **Lists following table rows without separation**
```bash
grep -nE -C20 '^\s*\|.*\|\s*$' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

10. **Lists after HTML blocks without separation**
```bash
grep -nE -C20 '^<[^>]+>.*$' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

11. **Lists after definition terms (term:)**
```bash
grep -nE -C20 '^[^:\n]*:$' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

12. **Lists after code fence blocks**
```bash
grep -nE -C20 '^```.*$' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

13. **Lists with inconsistent bullet markers (mixed -, *, +)**
```bash
grep -nE -C20 '^\s*[-*+]\s' "$1" | awk '/[-*+]/ {markers[substr($0,match($0,/[-*+]/),1)]++} END {if(length(markers)>1) print "Mixed bullet markers found"}'
```

14. **Lists after MathJax display blocks**
```bash
grep -nE -C20 '^\$\$.*\$\$$' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

15. **Detect proper list separation (should be preserved)**
```bash
grep -nE -C20 '^\s*$' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

16. **Lists after headings (usually acceptable)**
```bash
grep -nE -C20 '^#{1,6}\s+' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

17. **Multiple consecutive different list types without separation**
```bash
grep -nE -C20 '^\s*[0-9]+\.\s' "$1" | grep -A1 '^\s*[-*+]\s'
```

18. **Lists with improper nesting (indentation issues)**
```bash
grep -nE -C20 '^(  )+([-*+]|[0-9]+\.)\s' "$1"
```

19. **Lists at start of file (edge case check)**
```bash
head -5 "$1" | grep -nE -C20 '^\s*([-*+]|[0-9]+\.)\s'
```

20. **Lists after emphasis/strong text without separation**
```bash
grep -nE -C20 '(\*[^*]+\*|_[^_]+_):?$' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

21. **Quick scan for all list items to review context**
```bash
grep -nE -C20 '^\s*([-*+]|[0-9]+\.)\s' "$1"
```

22. **Detect text ending with periods followed by numbered lists**
```bash
grep -nE -C20 '\.$' "$1" | grep -A1 '^\s*[0-9]+\.\s'
```

23. **Find list items that might need blank lines above them**
```bash
python3 -c "
import re
with open('$1', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[1:], 2):
    if re.match(r'^\s*[-*+]\s', line):
        prev = lines[i-2].strip()
        if prev and not prev.startswith('#') and not re.match(r'^\s*[-*+]\s', lines[i-2]):
            print(f'Line {i}: List needs blank line above')
            print(f'  Previous: {prev}')
            print(f'  Current:  {line.strip()}')
            print()
"
```

24. **Lists after quoted text**
```bash
grep -nE -C20 '"[^"]*":?$' "$1" | grep -A1 '^\s*([-*+]|[0-9]+\.)\s'
```

25. **All potentially problematic list contexts**
```bash
grep -nE -C20 '^[^#\s\-\*\+0-9>|`].*[^:\s]$' "$1" | grep -B1 -A1 '^\s*([-*+]|[0-9]+\.)\s' | grep -E '^[0-9]+-[^-]'
```

## Command Implementation
This command will:
1. Read the specified markdown file
2. Identify lines that immediately precede list items without a blank line
3. Insert appropriate blank lines before list markers
4. Preserve all other formatting and content
5. Handle edge cases like lists at the beginning of files or after headings

### Detection Strategy
The command uses pattern matching to identify:
- **Primary pattern**: Non-blank lines followed immediately by list markers
- **Context awareness**: Distinguishes between acceptable cases (after headings) and problematic ones
- **List continuation**: Preserves existing list structure by not adding blanks between list items
- **Edge cases**: Handles special scenarios like nested lists, blockquotes, and code blocks