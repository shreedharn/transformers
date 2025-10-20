
---
argument-hint: [filename]
description: Comprehensively improve markdown file by separating math from prose and converting all mathematical content to clean MathML markup
allowed-tools: Read, Edit, MultiEdit, Bash, Bash(grep:*), Bash(sed:*), Bash(awk:*)
---


Comprehensively improve the markdown document `$1` by applying **semantic separation** between prose and mathematical content, converting all equations and notation to **MathML**, and ensuring professional, accessible, and standards-compliant mathematical presentation.

---

## 🌐 UNIFIED PRINCIPLE

**Maintain perfect separation** between descriptive Markdown prose and MathML mathematics.
All formulas, expressions, and mathematical variables must exist **only inside proper `<math>` blocks**.

---

## 1. Content Classification and Separation

### A. Mathematical Content → MathML Blocks

* All content containing formulas, symbols, equations, variables, or fractions.
* Any section containing LaTeX remnants (`$$`, `\begin{aligned}`, `\frac{}`, etc.) must be **converted to valid MathML**.
* Parameter tables, model equations, and numerical definitions are converted to `<math>` markup.

### B. Descriptive Content → Clean Markdown

* Narrative sections, architecture explanations, design reasoning.
* Lists and bullets describing text, not math.
* Tables that define terms, shapes, or meanings — keep as Markdown tables with inline MathML.

### ⚠️ Critical Rule:

Never mix MathML inside sentences. Instead, separate prose and math visually and semantically.

---

## 2. Math-Prose Separation Rules

✅ **DO:**

* Keep prose text and `<math>` elements on separate lines.
* Use `<math display="block">` for display equations.
* Use `<math display="inline">` or inline `<math>` inside tables and compact descriptions.

❌ **DON’T:**

* Embed `<math>` tags inside list markers, headings, or paragraph text.
* Use LaTeX delimiters (`$`, `$$`, `\begin{aligned}`).

---

### Example — Before (Mixed LaTeX and prose)

```markdown
The weight matrix is initialized as:
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad (49)$$
```

### Example — After (Clean MathML)

```markdown
The weight matrix is initialized as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>W</mi>
  <mo>∼</mo>
  <mi mathvariant="script">N</mi>
  <mo stretchy="false">(</mo>
  <mn>0</mn>
  <mo>,</mo>
  <mfrac>
    <mn>2</mn>
    <mrow>
      <msub><mi>n</mi><mtext>in</mtext></msub>
      <mo>+</mo>
      <msub><mi>n</mi><mtext>out</mtext></msub>
    </mrow>
  </mfrac>
  <mo stretchy="false">)</mo>
  <mspace width="1em"/>
  <mo>(</mo><mn>49</mn><mo>)</mo>
</math>
```

---

## 3. MathML Typographic Standards

### General Rules

* **Block math:**
  `<math display="block"> … </math>`
* **Inline math (within tables only):**
  `<math display="inline"> … </math>`
* **Subscripts and superscripts:**
  Use `<msub>`, `<msup>`, and `<msubsup>`.
* **Fractions:**
  `<mfrac><numerator/><denominator/></mfrac>`
* **Greek and calligraphic letters:**
  Use Unicode symbols (`𝒩`, `𝛼`, `𝜃`) or `mathvariant="script"`, `mathvariant="bold"` where needed.
* **Descriptive text inside math:**
  Wrap with `<mtext>`.

### Alignment

* Replace `\begin{aligned}` blocks with `<mtable>` + `<mtr>` + `<mtd>`.
* Use `<mo>=</mo>` alignment in separate `<mtd>` elements for equation alignment.

#### Example — LaTeX Aligned → MathML Aligned

**Before:**

```latex
$$
\begin{aligned}
y &= W x + b \\
\hat{y} &= \text{softmax}(y)
\end{aligned}
$$
```

**After:**

```html
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable>
    <mtr>
      <mtd><mi>y</mi></mtd><mtd><mo>=</mo></mtd>
      <mtd><mi>W</mi><mi>x</mi><mo>+</mo><mi>b</mi></mtd>
    </mtr>
    <mtr>
      <mtd><mover><mi>y</mi><mo>^</mo></mover></mtd><mtd><mo>=</mo></mtd>
      <mtd><mi>softmax</mi><mo>(</mo><mi>y</mi><mo>)</mo></mtd>
    </mtr>
  </mtable>
</math>
```

---

## 4. List Formatting Rules (Unchanged)

Keep all Markdown list conventions identical to the LaTeX version:

* Blank line before every list.
* No blank lines between list items.
* Never embed `<math>` tags within list markers.

---

## 5. Table Math Rules (Inline MathML)

Inside tables, **never use block math**. Use inline `<math>` elements to preserve alignment.

### Example — Before (Correct LaTeX for Tables)

**Use `\(...\)` in tables** (per improve-md.md rules):
```markdown
| Term | Size | Meaning |
|------|------|----------|
| \(x_t\) | \([1, D_{in}]\) | Input vector |
| \(W\) | \([D_{in}, D_{out}]\) | Weight matrix |
```

**Note**: Never use `$$...$$` in table cells as it breaks table formatting.

### Example — After (Inline MathML)

```markdown
| Term | Size | Meaning |
|------|------|----------|
| <math display="inline"><mi>x</mi><msub><mi>t</mi></msub></math> | <math display="inline"><mfenced><mn>1</mn><mo>,</mo><msub><mi>D</mi><mtext>in</mtext></msub></mfenced></math> | Input vector |
| <math display="inline"><mi>W</mi></math> | <math display="inline"><mfenced><msub><mi>D</mi><mtext>in</mtext></msub><mo>,</mo><msub><mi>D</mi><mtext>out</mtext></msub></mfenced></math> | Weight matrix |
```

---

## 6. Automation and Fixers

### Automated Fixes

| Fixer                       | Description                                                        | Safe | Converts To                                         |
| --------------------------- | ------------------------------------------------------------------ | ---- | --------------------------------------------------- |
| `latex_to_mathml`           | Converts `$$...$$` block LaTeX → MathML using latex2mathml library | ✅    | `<math display="block">...</math>` |
| `inline_math_to_mathml`     | Converts `$x$` inline LaTeX → MathML | ✅    | `<math display="inline">...</math>` |
| `paren_math_to_mathml`      | Converts `\(...\)` inline LaTeX (tables) → MathML | ✅    | `<math display="inline">...</math>` |
| `math_code_fence_to_mathml` | Converts ````math` fences → MathML | ✅    | `<math display="block">...</math>` |
| `list_formatting`           | Adds blank lines before lists, removes between items | ✅    | Formatting only |
| `bold_formatting`           | Removes excessive bold markers | ✅    | Formatting only |
| `remove_empty_mathml`       | Removes empty `<math>` tags | ✅    | Cleanup only |

---

### Manual Fixes Required

1. **Math-Prose Separation** — must manually extract and move text out of `<math>`.
2. **Semantic conversion** — replace ambiguous LaTeX commands (`\text`, `\frac`) with proper MathML equivalents.
3. **Content Classification** — decide whether a block is textual or mathematical.

---

## 7. Manual Review Workflow

**IMPORTANT**: Always review AI-generated changes before finalizing. Follow this workflow:

### Step 1: Run Automated Fixer

```bash
# Apply all fixes
python3 slash-cmd-scripts/src/py_latex_to_mathml.py yourfile.md

# Or with specific fixer
python3 slash-cmd-scripts/src/py_latex_to_mathml.py --fixer latex_to_mathml yourfile.md

# Or dry-run to preview
python3 slash-cmd-scripts/src/py_latex_to_mathml.py --dry-run yourfile.md
```

### Step 2: Review Changes with Git Diff

```bash
# View all changes for manual review
git diff yourfile.md

# Review specific segments
git diff yourfile.md | grep -A 5 -B 5 "<math"
```

### Step 3: Manual Verification Checklist

- [ ] **MathML validity**: All `<math>` tags properly closed
- [ ] **Display attributes**: Block math has `display="block"`, inline has `display="inline"`
- [ ] **Table math**: Tables use `display="inline"` (not block)
- [ ] **No LaTeX remnants**: No `$$`, `$`, `\frac`, `\begin{aligned}` remaining
- [ ] **Proper encoding**: Greek letters and symbols use HTML entities (e.g., `&#x003B1;` for α)
- [ ] **List formatting**: Blank lines before lists, none between items
- [ ] **Rendering test**: View in browser/MkDocs to verify visual output

### Step 4: Fix Any Issues Found

If issues detected:
```bash
# Revert if needed
git checkout yourfile.md

# Apply targeted fixes
python3 slash-cmd-scripts/src/py_latex_to_mathml.py --fixer <specific-fixer> yourfile.md

# Or manual edit
code yourfile.md
```

### Step 5: Validate Final Output

```bash
# Final check - should show no issues
git diff yourfile.md | head -50

# Commit when satisfied
git add yourfile.md
git commit -m "Convert LaTeX to MathML in yourfile.md"
```

---

## 8. Example Transformations

### Mixed Markdown → MathML

**Before:**

```markdown
The total parameters are computed as:
$$D_{in} \times D_{out} + D_{out} = D_{out}(D_{in} + 1)$$
```

**After:**

```markdown
The total parameters are computed as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow>
    <msub><mi>D</mi><mtext>in</mtext></msub>
    <mo>×</mo>
    <msub><mi>D</mi><mtext>out</mtext></msub>
    <mo>+</mo>
    <msub><mi>D</mi><mtext>out</mtext></msub>
    <mo>=</mo>
    <msub><mi>D</mi><mtext>out</mtext></msub>
    <mfenced><msub><mi>D</mi><mtext>in</mtext></msub><mo>+</mo><mn>1</mn></mfenced>
  </mrow>
</math>
```

---

### Alignment Example (ReLU, Sigmoid, Tanh)

**Before (LaTeX):**

```latex
$$
\begin{aligned}
\text{ReLU:} &\ \sigma(z) = \max(0, z) \\
\text{Sigmoid:} &\ \sigma(z) = \frac{1}{1 + e^{-z}} \\
\text{Tanh:} &\ \sigma(z) = \tanh(z)
\end{aligned}
$$
```

**After (MathML):**

```html
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable>
    <mtr>
      <mtd><mtext>ReLU:</mtext></mtd>
      <mtd><mi>σ</mi><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><mi>z</mi><mo>)</mo></mtd>
    </mtr>
    <mtr>
      <mtd><mtext>Sigmoid:</mtext></mtd>
      <mtd><mi>σ</mi><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>-</mo><mi>z</mi></mrow></msup></mrow></mfrac></mtd>
    </mtr>
    <mtr>
      <mtd><mtext>Tanh:</mtext></mtd>
      <mtd><mi>σ</mi><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo><mi>tanh</mi><mo>(</mo><mi>z</mi><mo>)</mo></mtd>
    </mtr>
  </mtable>
</math>
```

---

## 8. Quality Verification Checklist (Updated)

✅ **Math-Prose Separation**

* No mathematical variables appear in Markdown prose.
* All math is wrapped in `<math>` elements.
* All lists and headings are free of MathML content.

✅ **MathML Formatting**

* Proper use of `<mfrac>`, `<msub>`, `<mtext>`, `<mi>`, `<mo>`.
* No leftover LaTeX delimiters or commands.
* `<mtable>` used for multi-line equations.
* `<math display="inline">` used for table cells.

✅ **Accessibility**

* Each MathML block renders natively in browsers with MathML Core support.
* Semantic tags (`<mtext>`, `<mi>`, `<mo>`) aid screen reader accessibility.


