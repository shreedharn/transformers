---
name: improve-tutorial-md
description: Comprehensively improve or create an ML tutorial markdown document by combining automated fixing, list formatting, and ml-tutorial-writer standards into one unified workflow. Use when creating or polishing ML/DL tutorial markdown files.
argument-hint: [filename]
allowed-tools: Read, Edit, MultiEdit, Bash, Bash(grep:*), Bash(sed:*), Bash(awk:*), Bash(python3:*)
---

Comprehensively improve or create the markdown tutorial document `$ARGUMENTS` by combining automated fixing, list formatting, and ML tutorial writing standards into one unified workflow.

**SCOPE**: Apply this skill when creating a new ML tutorial OR polishing an existing one. It unifies the rules from `improve-md`, `improve-list`, and the `ml-tutorial-writer` agent into a single end-to-end pass.

---

## Phase 1: Automated Fixes (run first)

Run all safe automated fixers on the target file:

```bash
# Apply all automated fixes
python3 slash-cmd-scripts/src/py_fix_md.py "$ARGUMENTS"

# Then run the list fixer
python3 slash-cmd-scripts/src/improve_list_md.py "$ARGUMENTS"
```

Available fixer categories if you need targeted runs:

```bash
python3 slash-cmd-scripts/src/py_fix_md.py --category math_formatting "$ARGUMENTS"
python3 slash-cmd-scripts/src/py_fix_md.py --category list_formatting "$ARGUMENTS"
python3 slash-cmd-scripts/src/py_fix_md.py --category bold_formatting "$ARGUMENTS"
```

---

## Phase 2: Detect Remaining Issues

After automated fixes, run detectors to find issues requiring manual intervention:

```bash
# Full detection pass
python3 slash-cmd-scripts/src/py_improve_md.py "$ARGUMENTS"

# Targeted: math-prose mixing (most common in tutorials)
python3 slash-cmd-scripts/src/py_improve_md.py --category inline_math "$ARGUMENTS"

# Targeted: list formatting
python3 slash-cmd-scripts/src/improve_list_md.py --detector list_missing_blank_line_before "$ARGUMENTS"
python3 slash-cmd-scripts/src/improve_list_md.py --detector blank_lines_between_list_items "$ARGUMENTS"
```

---

## Phase 3: Manual Fixes — ML Tutorial Standards

Apply the following rules manually for issues the automated tools cannot resolve.

### UNIFIED PRINCIPLE

Apply the right format for the content type while maintaining strict separation between mathematical notation and prose text. ML tutorials in this repo use MathJax — never GitHub `math` code fences.

---

### A. Content Classification

**Mathematical Content → LaTeX display blocks:**
- Equations, variables, subscripts, formulas, parameter dimensions
- Any expression that mixes mathematical symbols with descriptive text

**Descriptive Content → Clean Markdown:**
- Architecture descriptions, training objectives, use cases
- Feature lists, capabilities, analogies, narrative explanations

**Critical Rule**: Never place mathematical symbols, variables, or notation (e.g., `x`, `W`, `\alpha`, `O(n)`, `→`, fractions) inside Markdown sentences, bullets, headings, or captions. Promote them to dedicated display blocks or rephrase using plain words.

---

### B. MathJax Rules (renderer-safe)

- **Display format**: `$$ \begin{aligned} … \end{aligned} $$` for all multi-line math
- **Single-line equations**: Always wrap in `\begin{aligned}`, never use bare `$$equation$$`
- **Inline variables**: Use `$$x\_1$$`, `$$h\_{t-1}$$` (double-dollar, escaped underscores)
- **Tables**: Use `$$` for all math in table cells (no display math in tables)
- **Line breaks inside blocks**: Use `\newline`, never `\\`
- **Labels**: Use `\text{description}` for short text labels within math blocks
- **Bold math**: `\mathbf{…}` for Latin, `\boldsymbol{…}` for Greek/symbols
- **Spacing**: `\,`, `\;`, `\quad`, `\qquad` inside math; no Markdown spacing inside blocks
- **Never use ````math` fences** — convert to `$${\textstyle\begin{aligned}…\end{aligned}}$$`

**Underscore escaping (context-critical):**

| Context | Rule | Example |
|---------|------|---------|
| Inside `\begin{aligned}…\end{aligned}` | Do NOT escape — use `_` directly | `x_1`, `h_{t-1}` |
| Inline math outside aligned blocks | MUST escape | `$$x\_1$$`, `$$h\_{t-1}$$` |
| Code blocks / ASCII art / Python | Never escape | `def my_func():` |

---

### C. List Formatting Rules

- **Blank line before every list** (bulleted and numbered) when it follows any content
- **No blank lines between list items** — only before the list starts
- **Consistent bullet markers** throughout (no mixing `-`, `*`, `+`)

**Correct:**
```markdown
These are the key components:

- Component A handles data processing
- Component B manages storage
- Component C provides the API
```

**Wrong:**
```markdown
These are the key components:
- Component A

- Component B

- Component C
```

---

### D. Tutorial Writing Standards (ml-tutorial-writer rules)

**Structure:**
- No table of contents
- Write at Freshman college level — accessible but not oversimplified
- Engaging narrative flow with natural transitions ("Now let's explore…")
- Express reasoning in paragraphs, not isolated bullet points
- Start with intuitive explanations before technical details
- Use analogies and metaphors for complex concepts
- Build progressively from fundamentals to advanced topics

**Visual Elements:**
- Use ASCII art / text diagrams where they add clarity
- Pay careful attention to spacing and alignment in diagrams

**Cross-references:**
- Review `README.md` and maintain cross-references
- Identify new terms for `glossary.md`

**Forbidden patterns:**
- Do NOT mix MathJax (`$$`, `\begin{aligned}`) on the same line as list markers, headings, or sentences
- Do NOT include math tokens in prose — promote to dedicated display blocks
- Do NOT use ````math` GitHub fences

---

## Phase 4: Verify

Run the full detector suite to confirm all issues are resolved:

```bash
python3 slash-cmd-scripts/src/py_improve_md.py "$ARGUMENTS"
python3 slash-cmd-scripts/src/improve_list_md.py "$ARGUMENTS"
```

Then perform a final manual scan for edge cases:

- [ ] No math symbols in Markdown prose, bullets, or headings
- [ ] All `\begin{aligned}` blocks use unescaped `_` (not `\_`)
- [ ] All inline `$$…$$` use escaped `\_`
- [ ] No code blocks with escaped underscores
- [ ] Blank line before every list, no blank lines between items
- [ ] No ````math` fences — all math uses MathJax `$$…$$` format
- [ ] Tutorial flows naturally with narrative prose, not choppy bullets
- [ ] Cross-references to `README.md` and `glossary.md` are current

---

## Quick Reference: Math-Prose Separation Patterns

**Before (wrong — math embedded in prose):**
```markdown
The attention mechanism computes Q, K, and V matrices where d_k = 512.
- Computing scores uses formula α_ij = softmax(q_i * k_j)
```

**After (correct — separated):**
```markdown
The attention mechanism computes query, key, and value matrices with specific dimensions:

$$
\begin{aligned}
Q, K, V &: \text{Query, key, and value matrices} \newline
d_k &= 512 \quad \text{(key dimension for computational efficiency)} \newline
\alpha_{ij} &= \text{softmax}(q_i \cdot k_j) \quad \text{(attention score)}
\end{aligned}
$$
```

**Before (wrong — descriptive content in LaTeX):**
```markdown
$$
{\textstyle
\begin{aligned}
\textbf{Structure} &: \text{Encoder (bidirectional) + Decoder (causal)} \newline
\textbf{Use cases} &: \text{Translation, summarization}
\end{aligned}
}
$$
```

**After (correct — descriptive content as Markdown):**
```markdown
**Structure**: Encoder (bidirectional) + Decoder (causal) with cross-attention
**Use cases**: Translation, summarization, structured tasks
```
