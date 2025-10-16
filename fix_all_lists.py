#!/usr/bin/env python3
"""
Comprehensive markdown list formatter
Fixes:
1. Missing blank lines before lists
2. Blank lines between list items
3. Excessive blank lines (more than 1 consecutive)
4. Tick icons in lists (optional - preserves them by default)
"""

import re
import sys
from pathlib import Path


def fix_markdown_lists(content: str, fix_tick_icons: bool = False) -> str:
    """Fix all list formatting issues in markdown content."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        current_line = lines[i]

        # Get previous line (if exists)
        prev_line = lines[i-1] if i > 0 else ""
        # Get next line (if exists)
        next_line = lines[i+1] if i < len(lines) - 1 else ""

        # Fix 1: Excessive blank lines (more than 1 consecutive)
        # If current line is blank and previous line is also blank, skip this line
        if i > 0 and current_line == "" and prev_line == "" and i < len(lines) - 1:
            # Check if we already have 1 blank line
            if len(fixed_lines) > 0 and fixed_lines[-1] == "":
                i += 1
                continue

        # Fix 2: Missing blank line before list
        # If next line starts a list but current line is not blank and not empty
        if (i < len(lines) - 1 and
            next_line.strip() and
            (next_line.strip().startswith('- ') or
             next_line.strip().startswith('* ') or
             re.match(r'^\d+\.\s', next_line.strip()) or
             next_line.strip().startswith('✅ ') or
             next_line.strip().startswith('❌ '))):

            # Check if current line has content (not blank, not a heading)
            if (current_line.strip() and
                not current_line.strip().startswith('#') and
                not current_line.strip().startswith('```') and
                not current_line.strip().startswith('---') and
                current_line.strip() != '$$' and
                not current_line.strip().endswith('$$') and
                not re.match(r'^\\begin\{', current_line.strip()) and
                not re.match(r'^\\end\{', current_line.strip())):

                # Check if we're not already in a list
                if not (current_line.strip().startswith('- ') or
                        current_line.strip().startswith('* ') or
                        re.match(r'^\d+\.\s', current_line.strip())):

                    fixed_lines.append(current_line)
                    fixed_lines.append("")  # Add blank line before list
                    i += 1
                    continue

        # Fix 3: Blank lines between list items
        # If current line is blank, next line is a list item, and previous line is also a list item
        if (current_line == "" and
            i > 0 and i < len(lines) - 1):

            # Check if previous line is a list item
            prev_is_list = (prev_line.strip().startswith('- ') or
                           prev_line.strip().startswith('* ') or
                           re.match(r'^\d+\.\s', prev_line.strip()))

            # Check if next line is a list item (at same indentation level)
            next_is_list = (next_line.strip().startswith('- ') or
                           next_line.strip().startswith('* ') or
                           re.match(r'^\d+\.\s', next_line.strip()))

            # If both are list items, skip the blank line
            if prev_is_list and next_is_list:
                # Get indentation levels
                prev_indent = len(prev_line) - len(prev_line.lstrip())
                next_indent = len(next_line) - len(next_line.lstrip())

                # Only remove if same indentation (same level list items)
                if prev_indent == next_indent:
                    i += 1
                    continue

        # Add the line (possibly modified)
        fixed_lines.append(current_line)
        i += 1

    result = '\n'.join(fixed_lines)

    # Optional: Fix tick icons (convert to normal list markers)
    if fix_tick_icons:
        result = re.sub(r'^(\s*)✅\s+', r'\1- ', result, flags=re.MULTILINE)
        result = re.sub(r'^(\s*)❌\s+', r'\1- ', result, flags=re.MULTILINE)

    return result


def process_file(filepath: Path, fix_tick_icons: bool = False, dry_run: bool = False):
    """Process a single markdown file."""
    print(f"Processing {filepath.name}...", end=" ")

    try:
        content = filepath.read_text(encoding='utf-8')
        fixed_content = fix_markdown_lists(content, fix_tick_icons=fix_tick_icons)

        if content == fixed_content:
            print("✓ No changes needed")
            return False

        if dry_run:
            print(f"⚠️  Would fix (dry run)")
            return True
        else:
            filepath.write_text(fixed_content, encoding='utf-8')
            print("✓ Fixed!")
            return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Process all markdown files with issues."""

    # Files to process (from the analysis)
    files_to_fix = [
        ("history_quick_ref.md", False),
        ("knowledge_store.md", False),
        ("mlp_intro.md", False),  # Has tick icons but we preserve them
        ("rnn_intro.md", False),   # Has tick icons but we preserve them
        ("transformers_fundamentals.md", False),
        ("transformers_math2.md", False),
        ("pytorch_ref.md", False),  # Has tick icon but we preserve it
        ("glossary.md", False),
        ("nn_intro.md", False),
        ("transformers_math1.md", False),
    ]

    base_dir = Path(__file__).parent
    fixed_count = 0

    print("=" * 60)
    print("Markdown List Formatter")
    print("=" * 60)

    for filename, fix_ticks in files_to_fix:
        filepath = base_dir / filename
        if filepath.exists():
            if process_file(filepath, fix_tick_icons=fix_ticks):
                fixed_count += 1
        else:
            print(f"⚠️  {filename} not found")

    print("=" * 60)
    print(f"✓ Processed {len(files_to_fix)} files, fixed {fixed_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
