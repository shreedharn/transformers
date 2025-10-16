---
argument-hint: [filename]
description: Detect and fix markdown list formatting issues - blank line spacing before and between list items
allowed-tools: Read, Edit, MultiEdit, Bash, Bash(grep:*), Bash(sed:*), Bash(awk:*)
---

Improve list formatting in the markdown document `$1` by applying proper blank line spacing rules for professional markdown presentation.

**CORE PRINCIPLE**: Lists must have a blank line before the first item for proper separation, but NO blank lines between individual list items.

## List Formatting Rules

### 1. Blank Line Before Lists (Required)

**Rule**: Every list (bulleted or numbered) MUST have a blank line before the first item when the list follows any text content.

### 2. No Blank Lines Between List Items (Required)

**Rule**: Individual list items should be consecutive with NO blank lines between them.

### 3. Consecutive Bold Text Spacing

**Rule**: Consecutive lines of bold text should have blank line spacing between them.

## Python Detector Usage Instructions

### Step 1: Discover Available Options

```bash
# See main help and usage patterns
python3 slash-cmd-scripts/src/improve_list_md.py --help

# List all 3 available detector names
python3 slash-cmd-scripts/src/improve_list_md.py --list-detectors

# Get detailed examples for all detectors
python3 slash-cmd-scripts/src/improve_list_md.py --list-detectors --help
```

### Step 2: Choose Detection Strategy

**Strategy A: Run all detectors (comprehensive analysis):**
```bash
python3 slash-cmd-scripts/src/improve_list_md.py "$1"
```

**Strategy B: Run specific detector (targeted analysis):**
```bash
# Detect lists missing blank line before first item
python3 slash-cmd-scripts/src/improve_list_md.py --detector list_missing_blank_line_before "$1"

# Detect blank lines between list items
python3 slash-cmd-scripts/src/improve_list_md.py --detector blank_lines_between_list_items "$1"

# Detect consecutive bold text without spacing
python3 slash-cmd-scripts/src/improve_list_md.py --detector consecutive_bold_without_spacing "$1"

# Get help for specific detector
python3 slash-cmd-scripts/src/improve_list_md.py --detector list_missing_blank_line_before --help
```

**Strategy C: Debug mode with verbose logging:**
```bash
python3 slash-cmd-scripts/src/improve_list_md.py --verbose "$1"
```

### Detector Descriptions

**1. list_missing_blank_line_before:**

Finds lists that start without blank line before first item.

**What this detector finds:**
- Lists that start immediately after text paragraphs
- Lists following headings without blank line separation
- Lists after bold/italic text without proper spacing
- Any list that begins without blank line before first item
- Reports line number and shows context

**2. blank_lines_between_list_items:**

Finds unnecessary blank lines between consecutive list items.

**What this detector finds:**
- Blank lines separating consecutive list items
- Items that appear as separate lists instead of unified list
- Unnecessary spacing breaking list continuity
- Shows which items have improper spacing

**3. consecutive_bold_without_spacing:**

Finds consecutive bold text lines without blank line separation.

**What this detector finds:**
- Bold text lines appearing back-to-back without spacing
- Section headers without proper separation
- Multiple bold labels without blank lines
- Reports bold text formatting issues

## Formatting Examples

### Example 1: Missing Blank Line Before List

**Before (Incorrect - DETECTED):**
```markdown
These are the key features:
- Feature one with details
- Feature two with details
- Feature three with details
```

**After (Correct):**
```markdown
These are the key features:

- Feature one with details
- Feature two with details
- Feature three with details
```

### Example 2: Blank Lines Between List Items

**Before (Incorrect - DETECTED):**
```markdown
**Requirements:**

- First requirement

- Second requirement

- Third requirement
```

**After (Correct):**
```markdown
**Requirements:**

- First requirement
- Second requirement
- Third requirement
```

### Example 3: List After Bold Heading

**Before (Incorrect - DETECTED):**
```markdown
**Key Components:**
- Component A with description
- Component B with description
```

**After (Correct):**
```markdown
**Key Components:**

- Component A with description
- Component B with description
```

### Example 4: Numbered List After Text

**Before (Incorrect - DETECTED):**
```markdown
Follow these steps to complete the task:
1. First step with instructions
2. Second step with instructions
3. Third step with instructions
```

**After (Correct):**
```markdown
Follow these steps to complete the task:

1. First step with instructions
2. Second step with instructions
3. Third step with instructions
```

### Example 5: List After Paragraph

**Before (Incorrect - DETECTED):**
```markdown
This document describes the architecture components.
- Component A handles data processing
- Component B manages storage
- Component C provides the API
```

**After (Correct):**
```markdown
This document describes the architecture components.

- Component A handles data processing
- Component B manages storage
- Component C provides the API
```

### Example 6: Consecutive Bold Text

**Before (Incorrect - DETECTED):**
```markdown
**Section One:**
**Section Two:**
**Section Three:**
```

**After (Correct):**
```markdown
**Section One:**

**Section Two:**

**Section Three:**
```

### Example 7: Mixed Bulleted and Numbered Lists

**Before (Incorrect - DETECTED):**
```markdown
Overview of components:
- Component A
- Component B

Implementation steps:
1. Install dependencies
2. Configure settings
```

**After (Correct):**
```markdown
Overview of components:

- Component A
- Component B

Implementation steps:

1. Install dependencies
2. Configure settings
```

### Example 8: List After Bold Label

**Before (Incorrect - DETECTED):**
```markdown
**Available Options:**
- Option one for standard mode
- Option two for advanced mode
- Option three for expert mode
```

**After (Correct):**
```markdown
**Available Options:**

- Option one for standard mode
- Option two for advanced mode
- Option three for expert mode
```

### Example 9: Bulleted List with Blank Lines Between Items

**Before (Incorrect - DETECTED):**
```markdown
The system provides:

- Real-time processing

- Automated backups

- Security monitoring
```

**After (Correct):**
```markdown
The system provides:

- Real-time processing
- Automated backups
- Security monitoring
```

### Example 10: Text in Parentheses Followed by List

**Before (Incorrect - DETECTED):**
```markdown
The configuration includes (with default settings):
- Database connection settings
- Cache configuration
- Logging preferences
```

**After (Correct):**
```markdown
The configuration includes (with default settings):

- Database connection settings
- Cache configuration
- Logging preferences
```

## Quality Verification Checklist

After running the list formatting detector:

**Blank Line Before Lists:**
- [ ] All lists have blank line before first item
- [ ] Lists after text paragraphs are properly separated
- [ ] Lists after bold/italic text have blank line
- [ ] Lists after any content have proper spacing

**No Blank Lines Between Items:**
- [ ] No blank lines between consecutive list items
- [ ] List items appear as unified blocks
- [ ] All unnecessary spacing removed

**Bold Text Spacing:**
- [ ] Consecutive bold text has blank line separation
- [ ] Section headers are properly spaced
- [ ] Visual hierarchy is clear

**Overall List Quality:**
- [ ] Consistent formatting throughout document
- [ ] Proper visual presentation
- [ ] Clean list structure

## Common Patterns and Fixes

### Pattern 1: Text Paragraph Followed by List
```markdown
# Before (DETECTED)
Some descriptive text here
- List item one
- List item two

# After
Some descriptive text here

- List item one
- List item two
```

### Pattern 2: Bold Heading Followed by List
```markdown
# Before (DETECTED)
**Features:**
- Feature one
- Feature two

# After
**Features:**

- Feature one
- Feature two
```

### Pattern 3: Items with Blank Lines
```markdown
# Before (DETECTED)
- Item one

- Item two

- Item three

# After
- Item one
- Item two
- Item three
```

### Pattern 4: Colon Followed by List
```markdown
# Before (DETECTED)
Implementation requirements:
- Requirement A
- Requirement B

# After
Implementation requirements:

- Requirement A
- Requirement B
```

### Pattern 5: Multiple Lists in Sequence
```markdown
# Before (DETECTED)
First category:
- Item A
- Item B
Second category:
- Item X
- Item Y

# After
First category:

- Item A
- Item B

Second category:

- Item X
- Item Y
```

### Pattern 6: Numbered List After Description
```markdown
# Before (DETECTED)
Follow the installation process:
1. Download the package
2. Extract the files
3. Run the installer

# After
Follow the installation process:

1. Download the package
2. Extract the files
3. Run the installer
```

## Usage Guidelines

**Getting Started:**
- **First-time analysis**: Use `--help` to see main usage patterns
- **Explore detectors**: Use `--list-detectors` to see all 3 detector names
- **Get examples**: Add `--help` to any command for detailed examples
- **Debugging**: Use `--verbose` flag when troubleshooting

**Smart Detection Strategy:**
1. **Quick scan**: Run all detectors to find all list formatting issues
2. **Targeted fix**: Use specific detector if you know the issue type
3. **Verification**: Re-run after fixes to ensure all issues resolved

**Progressive Approach:**
- Start with all detectors to identify all problems
- Review reported line numbers and context
- Apply fixes systematically through the document
- Verify with final detection run

The improve-list command provides focused markdown list formatting analysis, ensuring proper blank line spacing for professional document presentation. This provides a comprehensive or targeted report of list formatting issues, organized by detector with line numbers and 8 lines of context for easy identification and fixing. After running the detector, perform a final AI scan to catch any edge cases or patterns the automated detection might miss.