---
argument-hint: [filename]
description: Clean markdown file by separating markdown and MathJax elements properly
allowed-tools: Read, Edit, MultiEdit, Bash, Bash(grep:*), Bash(sed:*), Bash(awk:*)
---
### Task

Refactor `$1` so **all mathematics** appears in standalone MathJax display blocks and **all prose** remains pure Markdown. Move any math found in bullets, numbered items, tables, or paragraphs into a dedicated
`$$ \begin{aligned} … \end{aligned} $$` block placed immediately after the introducing text, with correct indentation and spacing. Where multiple adjacent math blocks belong to one idea, **consolidate** them; otherwise, keep them separate.

### Core Principle 

* **Never inline math with Markdown structures.** No `$$`/`\begin{aligned}` on the same line as a list marker, heading, table cell, or paragraph sentence.
* **One concept → one block.** Use `$$ … aligned … $$` with `\newline` for line breaks; keep short labels inside via `\text{…}`.
* **List safety.** Bullet/number → blank line → indented (2–3 spaces) display block; don’t indent 4+ spaces.
* **Boundary respect.** Consolidate only across **empty lines**; stop at any non-math content (headings, lists, bold labels, tables, quotes, code fences, HTML).
* **Tables stay inline.** Use `\(...\)` in tables; no display math.
* **No duplication.** Define variables once inside the nearest math block; avoid repeating in prose.
**No math tokens in prose.** Do not include mathematical symbols, variables, or notation (e.g., `x`, `W`, `\alpha`, `O(n)`, `→`, fractions) inside Markdown sentences, bullets, headings, or captions. Promote them to a dedicated display block (or use plain words).

### 1) Element Separation Rules

* **FORBID**: `^-|\*|\+|\d+\.` list lines that include `$$` or `\begin{aligned}`.
* **MOVE** any math in list text to a **separate** display block directly below (same indent, one blank line before).
* **GROUP** related equations into one block; keep variable definitions **once** with the equations.
* **KEEP** headings, feature lists, and narrative as pure Markdown.
* For tables, avoid display math; use inline `\(...\)` only.

### 2) MathJax Standardization (renderer-safe)

* Use `$$ … $$` for display math.
* Inside multi-line displays, use:

  ```tex
  \begin{aligned}
  … \newline
  …
  \end{aligned}
  ```

  (Use `\newline`, not `\\`.)
* Use `\text{…}` for **short labels** only (no long prose).
* Typography: `\mid`, `\lvert \cdot \rvert`, `\mathrm{…}` as needed.
* Bold math: `\mathbf{…}` (Latin) and `\boldsymbol{…}` (Greek/symbols).
* Spacing: use `\,`, `\;`, `\quad`, `\qquad` inside math; avoid Markdown spacing inside blocks.
* Replace Markdown `**bold**` inside math with `\mathbf{…}`/`\boldsymbol{…}`.
* Replace Markdown bullets inside math with `\(\bullet\)` **only** within math contexts.

### 3) List & Indentation Rules

* After a bullet/number introducing math, insert **one blank line**, then an **indented (2–3 spaces)** `$$ … $$` block to keep it within the list item.
* Do **not** indent 4+ spaces (that becomes a code block).
* For nested lists, indent the math block to the same nesting level as the item text.
* Never place `$$ … $$` on the same line as the list marker.

### 4) Block Consolidation with Boundary Respect

* **Goal:** Merge consecutive MathJax `$$\begin{aligned}…\end{aligned}$$` blocks **only** when they belong to one concept and are separated by **empty lines only**.
* **Consolidate iff** blocks are directly adjacent or separated only by blank lines.
* **Hard boundary (stop merging)** on any non-math, non-empty line or any new Markdown structure:
  headings (`^#{1,6}\s`), lists (`^\s*([-*+]|[0-9]+\.)\s`), blockquotes (`^\s*>`), tables (`^\s*\|`), code fences (` ``` `), HTML blocks (`^\s*<`), or plain text.
* **List safety:** never merge across list item boundaries.
* Inside the consolidated block: use `\newline` between lines; keep `&` alignment columns consistent; keep labels short via `\text{…}`.

### 5) Content Flow

* Lead-in sentence in Markdown → math block below.
* If prose + math mixing breaks layout, convert the small prose bit to a `\text{…}` line **inside** the block.
* Consolidate variable definitions in the nearest display block; avoid repetition in prose.

### 6) Implementation Strategy (CLI-friendly)

1. **Promote inline math** in paragraphs to standalone blocks below the lead-in.
2. **Normalize multi-line displays** to `\begin{aligned} … \end{aligned}`; replace any `\\` with `\newline`.
3. **Tables**: replace display math with inline `\(...\)` or plain text.
4. **Consolidate adjacent math blocks** per boundary rules.
5. **Whitespace**: one blank line before/after display blocks; strip trailing spaces.

## Comprehensive Markdown/MathJax Formatting Detector

**Optimized Python-based detector** that runs all detection patterns in a single execution:

```bash
python3 cmd-scripts/py-simplify-md.py "$1"
```

This comprehensive detector finds all markdown and MathJax formatting issues and presents them in segmented sections:

**Detection Categories:**
1. **List-marker lines with display math** (forbidden)
2. **Heading lines with math** (forbidden in headings)
3. **Table rows with display math** (forbidden; inline only if needed)
4. **Paragraphs with inline math** (promote to block)
5. **Display math with list marker** (hard fail)
6. **Over-indented display math** (likely rendered as code block)
7. **Adjacent math blocks** (manual consolidate check)
8. **Math tokens in prose** (broad net; review matches)
9. **Bold Markdown in math** (prefer \\mathbf/\\boldsymbol)
10. **Backslashes in aligned** (standardize to \\newline)
11. **Math missing aligned** (display math blocks possibly missing \\begin{aligned})
12. **Math in blockquotes** (usually unwanted)
13. **List missing blank line before math** (heuristic)
14. **Inline math in lists** (forbidden in bullets)
15. **Inline math in headings** (forbidden)
16. **Unpaired math delimiters** (odd number of $$)
17. **Math with text on same line** ($$ with non-empty text)
18. **Stray braces with math** ($$} or {$$)
19. **End aligned with trailing text** (\\end{aligned} followed by $$ then text)
20. **Mismatched aligned blocks** (unequal begin/end counts)
21. **End before begin aligned** (\\end{aligned} before any \\begin{aligned})

The detector automatically:
- Analyzes the entire file in a single pass
- Groups results by detection type with clear separators (`---`)
- Shows line numbers, issue counts, and context for each problem
- Provides a summary with total issues found
- Handles code blocks and other edge cases correctly
- Uses optimized regex patterns compiled at startup for performance

## AI Verification Step

After running the Python detector, perform a final AI scan to catch any edge cases or patterns the automated detection might miss:

```bash
echo "=== PYTHON DETECTOR RESULTS ==="
python3 cmd-scripts/py-simplify-md.py "$1"

echo ""
echo "=== AI VERIFICATION SCAN ==="
echo "Performing intelligent review of markdown and MathJax formatting..."
```

**Manual AI Review Guidelines:**
1. **Verify Python results** - Check if detected issues are legitimate formatting problems
2. **Scan for missed patterns** - Look for unusual math/markdown combinations not covered by patterns
3. **Check consolidation opportunities** - Identify adjacent math blocks that should be merged
4. **Validate context** - Confirm that proposed changes improve readability and rendering
5. **Edge case detection** - Find complex nested structures or unusual MathJax combinations
6. **False positive filtering** - Identify any incorrectly flagged legitimate formatting

The AI verification ensures comprehensive coverage by combining automated pattern detection with intelligent contextual analysis of the original objective: "separating markdown and MathJax elements properly."



### 7) Quality Checks

* [ ] No list line contains `$$` or `\begin{aligned}`.
* [ ] **No math symbols/notation appear in Markdown prose** (paragraphs, bullets, headings, captions, table text).
* [ ] All equations are in `$$ … \begin{aligned} … \end{aligned} … $$` blocks using `\newline`.
* [ ] List-adjacent equations are separate, **indented** blocks (2–3 spaces), with a blank line before.
* [ ] Adjacent math blocks consolidated only across **empty lines**; never across headings/lists/quotes/tables/code/HTML or any non-empty prose.
* [ ] No display math in tables (inline `\(...\)` only if absolutely necessary).
* [ ] Variable definitions appear once inside the nearest math block (no duplication in prose).
* [ ] Consistent typography (`\mathbf{…}` / `\boldsymbol{…}`, `\mathrm{…}`, `\mid`, `\lvert\cdot\rvert`) and spacing (`\,`, `\;`, `\quad`).
* [ ] File renders without list breaks, code-fence misclassification, or layout jitter.


---

## Output Format Examples

### Example 1: Mixed Notation → Standardized

**Before:**

```markdown
- **$$x W$$:**
- **$$+ b$$:** Bias allows shifting the activation threshold
- **\begin{aligned} \sigma(\cdot) \end{aligned}:** Non-linearity enables learning complex patterns

\begin{aligned} h^{(1)} &= \sigma^{(1)}(x W^{(1)} + b^{(1)}) \end{aligned}

\begin{aligned} h^{(2)} &= \sigma^{(2)}(h^{(1)} W^{(2)} + b^{(2)}) \end{aligned}
```

**After:**

```latex
$$
{\textstyle
\begin{aligned}
\mathbf{x W} \quad &: \text{Matrix multiplication combines input features with learned weights} \newline
\mathbf{+ b} \quad &: \text{Bias allows shifting the activation threshold} \newline
\mathbf{\sigma(\cdot)} \quad &: \text{Non-linearity enables learning complex patterns}
\end{aligned}
}
$$

$$
\begin{aligned}
h^{(1)} &= \sigma^{(1)}(x W^{(1)} + b^{(1)}) \newline
h^{(2)} &= \sigma^{(2)}(h^{(1)} W^{(2)} + b^{(2)})
\end{aligned}
$$
```

---

### Example 2: Parameter Counting

**Before:**

```markdown
- **Weights**: \begin{aligned} D_{in} \times D_{out} \end{aligned} parameters
- **Biases**: $$D_{out}$$ parameters
- **Total**: \begin{aligned} D_{in} \times D_{out} + D_{out} &= D_{out}(D_{in} + 1) \end{aligned}
```

**After:**

```latex
$$
\begin{aligned}
\text{Weights:} \quad &D_{in} \times D_{out} \text{ parameters} \newline
\text{Biases:} \quad &D_{out} \text{ parameters} \newline
\text{Total:} \quad &D_{in} \times D_{out} + D_{out} = D_{out}(D_{in} + 1)
\end{aligned}
$$
```

---

### Example 3: Activation Functions

**Before:**

```markdown
- **ReLU**: \begin{aligned} \sigma(z) &= \max(0, z) \end{aligned} - most popular, simple and effective
- **Sigmoid**: \begin{aligned} \sigma(z) &= \frac{1}{1 + e^{-z}} \end{aligned} - outputs between 0 and 1
- **Tanh**: \begin{aligned} \sigma(z) &= \tanh(z) \end{aligned} - outputs between -1 and 1
```

**After:**

```latex
$$
{\textstyle
\begin{aligned}
\text{ReLU:} \quad &\sigma(z) = \max(0, z) \quad \text{- most popular, simple and effective} \newline
\text{Sigmoid:} \quad &\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{- outputs between 0 and 1} \newline
\text{Tanh:} \quad &\sigma(z) = \tanh(z) \quad \text{- outputs between -1 and 1}
\end{aligned}
}
$$
```

---

### Example 4: Network Example

**Before:**

```markdown
- Input: 10 features
- Hidden 1: 20 neurons → \begin{aligned} 20 \times (10 + 1) &= 220 \end{aligned} parameters
- Hidden 2: 15 neurons → \begin{aligned} 15 \times (20 + 1) &= 315 \end{aligned} parameters
- Output: 1 neuron → \begin{aligned} 1 \times (15 + 1) &= 16 \end{aligned} parameters
- **Total**: 551 parameters
```

**After:**

```latex
$$
\begin{aligned}
\text{Input:} \quad &\text{10 features} \newline
\text{Hidden 1:} \quad &\text{20 neurons} \rightarrow 20 \times (10 + 1) = 220 \text{ parameters} \newline
\text{Hidden 2:} \quad &\text{15 neurons} \rightarrow 15 \times (20 + 1) = 315 \text{ parameters} \newline
\text{Output:} \quad &\text{1 neuron} \rightarrow 1 \times (15 + 1) = 16 \text{ parameters} \newline
\textbf{Total:} \quad &\textbf{551 parameters}
\end{aligned}
$$
```

---

### Example 5: Boundary Respect (Do **Not** Merge Across Sections)

**Before (Correct — keep separate):**

```markdown
\begin{aligned} h^{(1)} &= \sigma^{(1)}(x W^{(1)} + b^{(1)}) \end{aligned}

**Layer Naming Convention:**

\begin{aligned} y &= h^{(L-1)} W^{(L)} + b^{(L)} \end{aligned}
```

**After (Correct — kept separate):**

```latex
$$
\begin{aligned}
h^{(1)} &= \sigma^{(1)}(x W^{(1)} + b^{(1)})
\end{aligned}
$$

**Layer Naming Convention:**

$$
\begin{aligned}
y &= h^{(L-1)} W^{(L)} + b^{(L)}
\end{aligned}
$$
```

---
