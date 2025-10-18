---
argument-hint: [filename]
description: Automatically fix common markdown and LaTeX formatting issues using safe, validated fixers
allowed-tools: Read, Edit, Bash, Python3
---

Automatically fix common markdown and LaTeX formatting issues in `$1` using the professional fixer script. This command applies only safe, validated transformations that have been tested to avoid creating new issues.

**IMPORTANT**: This fixer complements the detector (`improve-md`). Always run the detector first to understand what issues exist, then use this fixer to automatically correct the safe, automatable issues.

## What the Fixer Can Fix

The fixer script applies **only safe transformations** that have been validated to avoid creating new issues. After extensive testing, the following fixers are enabled:

### ✅ Enabled Fixers (Safe & Validated)

**1. Bold Formatting Fixer** (`bold_formatting`)
- Removes excessive bold formatting from list items
- Cleans up redundant bold markers in structured content
- **Category**: `bold_formatting`

**2. Empty Block Fixer** (`remove_empty_blocks`)
- Removes empty `$$` math delimiters with no content
- Cleans up consecutive empty dollar signs
- **Category**: `math_formatting`

**3. List Formatting Fixer** (`list_formatting`)
- Adds blank lines before markdown lists (bulleted and numbered)
- Removes blank lines between individual list items
- **Category**: `list_formatting`

**4. Wrap Aligned Blocks Fixer** (`wrap_aligned_blocks`)
- Adds `$$` wrappers around unwrapped `\begin{aligned}...\end{aligned}` blocks
- Context-aware: Respects existing `$$` blocks and ````math` fences
- **Category**: `math_formatting`

**5. Math Code Fence to MathJax Fixer** (`math_code_fence_to_mathjax`)
- Converts GitHub-style ````math` code fences to professional MathJax format
- Wraps content in `$${\textstyle\begin{aligned}...\end{aligned}}$$`
- **Category**: `math_formatting`

### ❌ Disabled Fixers (Cause Issues)

**1. Escape Underscores Fixer** (DISABLED)
- **Why disabled**: Incorrectly escapes underscores inside LaTeX math contexts where they don't need escaping
- **Issue**: `\delta_2` becomes `\delta\_2` (wrong inside math blocks)
- **Manual alternative**: Use detector to identify, then manually fix context-aware escaping

**2. Single Dollar to Double Dollar Fixer** (DISABLED)
- **Why disabled**: Blindly converts `$...$` to `$$...$$` creating invalid inline display math
- **Issue**: Inline math `$f(x)$` becomes display math `$$f(x)$$` (semantically wrong)
- **Manual alternative**: Use detector to identify inline math issues, then manually restructure

## CLI Usage

### Step 1: Run the Detector First

Always start by running the detector to understand what issues exist:

```bash
# Run comprehensive detection
python3 slash-cmd-scripts/src/py_improve_md.py "$1"

# Or run specific categories
python3 slash-cmd-scripts/src/py_improve_md.py --category list_formatting "$1"
python3 slash-cmd-scripts/src/py_improve_md.py --category bold_formatting "$1"
```

### Step 2: Apply Fixes

After understanding the issues, apply automatic fixes:

```bash
# See available options
python3 slash-cmd-scripts/src/py_fix_md.py --help

# List all fixers and their status
python3 slash-cmd-scripts/src/py_fix_md.py --list-fixers

# List available categories
python3 slash-cmd-scripts/src/py_fix_md.py --list-categories
```

### Fix Application Strategies

**Strategy A: Apply all safe fixers (recommended):**
```bash
python3 slash-cmd-scripts/src/py_fix_md.py "$1"
```

**Strategy B: Preview changes before applying (dry-run):**
```bash
python3 slash-cmd-scripts/src/py_fix_md.py --dry-run "$1"
```

**Strategy C: Apply specific category of fixes:**
```bash
# Fix only list formatting issues
python3 slash-cmd-scripts/src/py_fix_md.py --category list_formatting "$1"

# Fix only math formatting issues
python3 slash-cmd-scripts/src/py_fix_md.py --category math_formatting "$1"

# Fix only bold formatting issues
python3 slash-cmd-scripts/src/py_fix_md.py --category bold_formatting "$1"
```

**Strategy D: Apply specific individual fixer:**
```bash
# Fix only bold formatting
python3 slash-cmd-scripts/src/py_fix_md.py --fixer bold_formatting "$1"

# Fix only unwrapped aligned blocks
python3 slash-cmd-scripts/src/py_fix_md.py --fixer wrap_aligned_blocks "$1"

# Fix only list spacing
python3 slash-cmd-scripts/src/py_fix_md.py --fixer list_formatting "$1"
```

**Strategy E: Verbose mode for debugging:**
```bash
python3 slash-cmd-scripts/src/py_fix_md.py --verbose "$1"
```

## Before and After Examples

### Example 1: Bold Formatting in Lists

**Before (Excessive Bold):**
```markdown
The Big Picture: Whether we're finding the bottom of a simple parabola or training a neural network with millions of parameters, we're doing the same fundamental thing:

1. **Measure the slope** (derivative, gradient, or backpropagated error)
2. **Take a step in the opposite direction** (negative sign)
3. **Control step size** (learning rate α)
4. **Repeat until we reach the bottom**
```

**After (Clean):**
```markdown
The Big Picture: Whether we're finding the bottom of a simple parabola or training a neural network with millions of parameters, we're doing the same fundamental thing:

1. Measure the slope (derivative, gradient, or backpropagated error)
2. Take a step in the opposite direction (negative sign)
3. Control step size (learning rate α)
4. Repeat until we reach the bottom
```

**Fixer used**: `bold_formatting`

---

### Example 2: List Missing Blank Lines

**Before (Missing Blank Line):**
```markdown
Implementation details:
- Parallel computation enabled
- Memory optimization active

**Features:**
1. Fast processing
2. Low memory usage
```

**After (Proper Spacing):**
```markdown
Implementation details:

- Parallel computation enabled
- Memory optimization active

**Features:**

1. Fast processing
2. Low memory usage
```

**Fixer used**: `list_formatting`

---

### Example 3: Unwrapped Aligned Blocks

**Before (Missing $$ Wrappers):**
```markdown
From Derivatives to Gradients:

When we have multiple variables, we need partial derivatives:

\begin{aligned}
\frac{\partial f}{\partial x} = 2x \quad &\text{(rate of change with respect to x)} \newline
\frac{\partial f}{\partial y} = 2y \quad &\text{(rate of change with respect to y)}
\end{aligned}
```

**After (Properly Wrapped):**
```markdown
From Derivatives to Gradients:

When we have multiple variables, we need partial derivatives:

$$
\begin{aligned}
\frac{\partial f}{\partial x} = 2x \quad &\text{(rate of change with respect to x)} \newline
\frac{\partial f}{\partial y} = 2y \quad &\text{(rate of change with respect to y)}
\end{aligned}
$$
```

**Fixer used**: `wrap_aligned_blocks`

**Note**: This fixer is context-aware and won't add `$$` inside ````math` fences or existing `$$` blocks.

---

### Example 4: Empty Math Blocks

**Before (Empty Delimiters):**
```markdown
Some text here.

$$

$$

More text here.

$$
$$

Final text.
```

**After (Cleaned):**
```markdown
Some text here.

More text here.

Final text.
```

**Fixer used**: `remove_empty_blocks`

---

### Example 5: Blank Lines Between List Items

**Before (Extra Spacing):**
```markdown
**Features:**

- Item one

- Item two

- Item three
```

**After (Correct - No Inter-Item Spacing):**
```markdown
**Features:**

- Item one
- Item two
- Item three
```

**Fixer used**: `list_formatting`

---

### Example 6: Math Code Fence Blocks to MathJax

**Before (GitHub-style ```math fence):**
````markdown
The Derivative as Slope: For any function $f(x)$, the derivative $\frac{df}{dx}$ tells us the slope at any point:

```math
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
```

What this equation means: "If I move a tiny amount h to the right, how much does f change?"
````

**After (Professional MathJax):**
```markdown
The Derivative as Slope: For any function $f(x)$, the derivative $\frac{df}{dx}$ tells us the slope at any point:

$$
{\textstyle
\begin{aligned}
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\end{aligned}
}
$$

What this equation means: "If I move a tiny amount h to the right, how much does f change?"
```

**Fixer used**: `math_code_fence_to_mathjax`

**Benefits:**
- **Consistent rendering**: MathJax format renders uniformly across platforms
- **Professional appearance**: Uses industry-standard LaTeX delimiters
- **Better integration**: Matches the style of other mathematical expressions in the document
- **Future-proof**: More widely supported than GitHub-specific ````math` syntax

**Note**: After running this fixer, you may want to manually move prose descriptions into `\text{...}` blocks within the aligned environment for better separation of math and prose, as shown in the improve-md examples.

---

## Recommended Workflow

### 1. Detection Phase

```bash
# Run detector to see all issues
python3 slash-cmd-scripts/src/py_improve_md.py "$1" > issues.txt

# Review the issues to understand what needs fixing
cat issues.txt
```

### 2. Safe Automatic Fixing Phase

```bash
# Preview what will be fixed (dry-run)
python3 slash-cmd-scripts/src/py_fix_md.py --dry-run "$1"

# Apply all safe fixes
python3 slash-cmd-scripts/src/py_fix_md.py "$1"
```

### 3. Validation Phase

```bash
# Re-run detector to verify improvements
python3 slash-cmd-scripts/src/py_improve_md.py "$1"

# Compare before and after
git diff "$1"
```

### 4. Manual Fixing Phase

For issues the fixer can't handle automatically:

- **Inline math in prose**: Manually restructure to separate math from prose
- **Underscore escaping**: Context-aware escaping (escape in LaTeX, preserve in code)
- **Display math positioning**: Manually move math blocks to appropriate locations
- **Semantic restructuring**: Extract math from headings, lists, tables

## Technical Architecture

The fixer script follows professional software engineering practices:

### Design Patterns Used

- **Strategy Pattern**: Pluggable fixer components with abstract base class
- **Single Responsibility**: Each fixer handles one specific transformation
- **Dependency Injection**: Fixers injected into orchestrator class
- **State Machine**: Context tracking for proper multi-line structure handling

### Key Implementation Features

- **Module-level regex compilation** for performance
- **Comprehensive type hints** throughout
- **Structured logging** instead of print statements
- **Dataclasses** for structured results (FixResult, FileFixResult)
- **Custom exception hierarchy** (FixerError, InvalidContentError)
- **Context-aware transformations** (tracks `$$` blocks, ````math` fences, code blocks)

### Context Tracking

The fixer maintains state to avoid creating nested issues:

```python
in_math_block = False  # Track if we're inside $$...$$
in_code_fence = False  # Track if we're inside ```...```
```

This prevents:
- Adding `$$` inside existing `$$` blocks
- Adding `$$` inside ````math` fences
- Escaping underscores in code blocks

## Zero Regression Validation

All enabled fixers have been tested to ensure they don't create new issues:

| Fixer | Test File | Issues Before | Fixes Applied | Issues After | Regression |
|-------|-----------|---------------|---------------|--------------|------------|
| All enabled | transformers_math1.md | 110 | 12 | 110 | ✅ None |
| All enabled | transformers_math2.md | 66 | 0 | 66 | ✅ None |

## Important Limitations

### What This Fixer Cannot Fix

**Level 3 Transformations (Semantic Restructuring):**
- Moving inline math from prose to display blocks (requires understanding context)
- Extracting math from headings and list items (needs semantic analysis)
- Promoting "paragraphs with inline math" to block math (semantic judgment)
- Converting display math delimiters used incorrectly (context-dependent)

**Why Some Fixers Are Disabled:**
- **Underscore escaping**: Requires understanding LaTeX vs. code vs. ASCII art contexts
- **Dollar sign conversion**: Requires distinguishing inline math intent from display math
- **Complex restructuring**: Needs human judgment about content meaning

### When Manual Fixing Is Required

Use manual fixing for:
1. **48 "Paragraphs with inline math" issues** - Need to extract math to display blocks
2. **Math in headings/lists** - Requires semantic restructuring
3. **Context-aware underscore escaping** - Different rules for LaTeX, code, ASCII art
4. **Display math positioning** - Needs understanding of document structure

## Fixer Categories

The fixer organizes transformations into categories matching the detector:

### Available Categories

1. **`math_formatting`** (3 enabled fixers)
   - `wrap_aligned_blocks`: Adds `$$` wrappers to unwrapped `\begin{aligned}` blocks
   - `remove_empty_blocks`: Removes empty `$$` delimiters
   - `math_code_fence_to_mathjax`: Converts ````math` blocks to MathJax format

2. **`list_formatting`** (1 enabled fixer)
   - `list_formatting`: Adds blank lines before lists, removes between items

3. **`bold_formatting`** (1 enabled fixer)
   - `bold_formatting`: Removes excessive bold from list items

## Quality Verification After Fixing

After running the fixer, verify the results:

**Check Applied Fixes:**
- [ ] Review git diff to see what changed
- [ ] Verify list formatting looks correct (blank lines before, not between)
- [ ] Check that `\begin{aligned}` blocks now have `$$` wrappers
- [ ] Confirm bold formatting was cleaned appropriately

**Run Detector Again:**
- [ ] Re-run detector to see remaining issues
- [ ] Verify issue count didn't increase (no regressions)
- [ ] Identify remaining issues that need manual fixing

**Manual Review:**
- [ ] Check that code blocks weren't affected
- [ ] Verify math blocks render correctly
- [ ] Ensure document structure is preserved

## Help and Debugging

### Get Help on Specific Features

```bash
# Main help
python3 slash-cmd-scripts/src/py_fix_md.py --help

# List all available fixers with status
python3 slash-cmd-scripts/src/py_fix_md.py --list-fixers

# List all categories
python3 slash-cmd-scripts/src/py_fix_md.py --list-categories

# Get category-specific help
python3 slash-cmd-scripts/src/py_fix_md.py --category math_formatting --help

# Get fixer-specific help
python3 slash-cmd-scripts/src/py_fix_md.py --fixer wrap_aligned_blocks --help
```

### Debug Mode

```bash
# Enable verbose logging
python3 slash-cmd-scripts/src/py_fix_md.py --verbose "$1"

# Combine with dry-run for safe debugging
python3 slash-cmd-scripts/src/py_fix_md.py --verbose --dry-run "$1"
```

### Common Issues

**Issue**: "No changes needed" but detector shows issues
- **Cause**: Issues require manual fixing (disabled fixers or semantic restructuring)
- **Solution**: Review detector output for issue types, apply manual fixes

**Issue**: File was modified but issues increased
- **Cause**: Should not happen with current enabled fixers (report as bug if it does)
- **Solution**: Revert file (`git checkout`), report issue details

**Issue**: Fixer not running on specific issue
- **Cause**: Fixer may be disabled or issue requires manual intervention
- **Solution**: Check `--list-fixers` to see enabled/disabled status

## Example: Complete Workflow

```bash
# 1. Check current issues
python3 slash-cmd-scripts/src/py_improve_md.py transformers_math1.md > before.txt
echo "Before: $(grep 'SUMMARY:' before.txt)"

# 2. Preview fixes
python3 slash-cmd-scripts/src/py_fix_md.py --dry-run transformers_math1.md

# 3. Apply fixes
python3 slash-cmd-scripts/src/py_fix_md.py transformers_math1.md

# 4. Verify no regressions
python3 slash-cmd-scripts/src/py_improve_md.py transformers_math1.md > after.txt
echo "After: $(grep 'SUMMARY:' after.txt)"

# 5. Review changes
git diff transformers_math1.md

# 6. Commit if satisfied
git add transformers_math1.md
git commit -m "Fix: Apply automatic markdown formatting fixes

- Remove excessive bold from list items (12 fixes)
- Fix list spacing (blank lines before, not between)
- Wrap unwrapped aligned blocks with $$

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Summary

The md-fixer command provides **safe, validated automatic fixes** for common markdown and LaTeX formatting issues. It complements the detector by automatically fixing issues that don't require semantic understanding or context judgment. For complex transformations (inline math extraction, semantic restructuring), manual fixing is still required.

**Key Principle**: Only apply transformations that have been extensively tested to avoid creating new issues. When in doubt, preserve the original structure and flag for manual review.
