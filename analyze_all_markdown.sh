#!/bin/bash
# Comprehensive markdown analysis script for all files

echo "=========================================="
echo "Markdown Quality Analysis Report"
echo "Generated: $(date)"
echo "=========================================="
echo ""

# List of files to analyze (excluding LICENSE and CLAUDE)
files=(
    "README.md"
    "glossary.md"
    "history_quick_ref.md"
    "knowledge_store.md"
    "math_quick_ref.md"
    "mlp_intro.md"
    "nn_intro.md"
    "pytorch_ref.md"
    "rnn_intro.md"
    "transformers_advanced.md"
    "transformers_fundamentals.md"
    "transformers_math1.md"
    "transformers_math2.md"
)

# Summary statistics
total_files=0
files_with_issues=0
total_issues=0

# Create summary report
echo "SUMMARY BY FILE:"
echo "----------------------------------------"

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        ((total_files++))
        echo ""
        echo "File: $file"

        # Run detector and count issues
        output=$(python3 slash-cmd-scripts/src/py_improve_md.py "$file" 2>&1)

        # Extract issue counts
        issue_count=$(echo "$output" | grep -c "^Line [0-9]*:")

        if [ $issue_count -gt 0 ]; then
            ((files_with_issues++))
            ((total_issues += issue_count))
            echo "  Issues found: $issue_count"

            # Show category breakdown
            echo "  Categories:"
            echo "$output" | grep "^Detector:" | sort | uniq -c | sed 's/^/    /'
        else
            echo "  ✓ No issues found"
        fi
    fi
done

echo ""
echo "=========================================="
echo "OVERALL STATISTICS:"
echo "----------------------------------------"
echo "Total files analyzed: $total_files"
echo "Files with issues: $files_with_issues"
echo "Total issues found: $total_issues"
echo "=========================================="
echo ""
echo "To view detailed report for a specific file, run:"
echo "  python3 slash-cmd-scripts/src/py_improve_md.py <filename>"
echo ""
echo "To view issues by category, run:"
echo "  python3 slash-cmd-scripts/src/py_improve_md.py --category <category> <filename>"
echo ""
echo "Available categories: inline_math, display_math, list_formatting, alignment, structural, syntax"
