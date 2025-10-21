
---
argument-hint: [filename]
description: Comprehensively improve markdown file by separating math from prose and converting all mathematical content to clean MathML markup
allowed-tools: Read, Edit, MultiEdit, Bash, Bash(grep:*), Bash(sed:*), Bash(awk:*)
---


Comprehensively improve the markdown document `$1` by applying **semantic separation** between prose and mathematical content, converting all equations and notation to **MathML**, and ensuring professional, accessible, and standards-compliant mathematical presentation and polished Markdown formatting for descriptive content.


---

## 🌐 UNIFIED PRINCIPLE

**Maintain perfect separation** between descriptive Markdown prose and MathML mathematics.
All formulas, expressions, and mathematical variables must exist **only inside proper `<math>` blocks**.

---

## 1. Content Classification and Separation

**BEFORE applying any transformations, classify content and separate elements:**

### A. Mathematical Content → MathML Blocks

* Content with mathematical symbols, equations, variables, formulas
* Expressions using LaTeX remnants (`$$`, `\begin{aligned}`, `\frac{}`, etc.) that must be **converted to valid MathML**
* Parameter descriptions with dimensions or calculations
* Any content mixing mathematical variables with descriptive text
* Parameter tables, model equations, and numerical definitions are converted to `<math>` markup

### B. Descriptive Content → Clean Markdown

* Architecture descriptions, use cases, training objectives
* Feature lists, capabilities, or narrative explanations
* Simple structured information without mathematical elements
* Plain text descriptions about functionality or characteristics
* Narrative sections, design reasoning
* Tables that define terms, shapes, or meanings — keep as Markdown tables with inline MathML

### ⚠️ Critical Separation Rule:

**Do not include mathematical symbols, variables, or notation (e.g., `x`, `W`, `α`, `O(n)`, `→`, fractions) inside Markdown sentences, bullets, headings, or captions. Promote them to dedicated MathML display blocks (or use plain words).**

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

## 4. List Formatting and Separation Rules

* **Blank line requirement:** Ensure blank lines before every markdown list (bulleted and numbered)
* **NO blank lines between list items:** Only add blank lines before the start of lists, never between individual list items
* **List safety:** Never place mathematical content directly in list markers. Never embed `<math>` tags within list markers.
* **Proper separation:** Lists after any content type require blank line separation:
  - After prose paragraphs, colons, bold/italic text
  - After parenthetical statements, inline code/math expressions
  - After blockquotes, table rows, HTML blocks, code fences
  - After MathML display blocks or other structural elements
* **Consistent markers:** Use consistent bullet markers throughout document (avoid mixing -, *, +)
* **Nested lists:** Maintain proper indentation and blank line structure

### List Formatting Examples:

**Before (Incorrect):**
```markdown
Implementation details:
- Parallel computation enabled
- Memory optimization active

**Features:**
1. Fast processing
2. Low memory usage
```

**After (Correct):**
```markdown
Implementation details:

- Parallel computation enabled
- Memory optimization active

**Features:**

1. Fast processing
2. Low memory usage
```

**CRITICAL: Avoid These Common Mistakes:**

❌ **Wrong - Blank lines between list items:**
```markdown
**Features:**

- Item one

- Item two

- Item three
```

✅ **Correct - No blank lines between items:**
```markdown
**Features:**

- Item one
- Item two
- Item three
```

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

## 5a. Content-Specific Formatting Rules

### Keep as Clean Markdown:

```markdown
### Encoder-Decoder: T5 Family

**Structure**: Encoder (bidirectional) + Decoder (causal) with cross-attention
**Training**: Various objectives (span corruption, translation, etc.)
**Use cases**: Translation, summarization, structured tasks
```

### Convert to Professional MathML:

```markdown
The attention mechanism computes query, key, and value matrices with specific dimensions:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable>
    <mtr>
      <mtd><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi></mtd>
      <mtd><mo>:</mo></mtd>
      <mtd><mtext>Query, key, and value matrices</mtext></mtd>
    </mtr>
    <mtr>
      <mtd><msub><mi>d</mi><mi>k</mi></msub><mo>=</mo><mn>512</mn></mtd>
      <mtd><mo>:</mo></mtd>
      <mtd><mtext>Key dimension for computational efficiency</mtext></mtd>
    </mtr>
    <mtr>
      <mtd colspan="3">
        <mtext>Attention</mtext><mo>(</mo><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi><mo>)</mo>
        <mo>=</mo>
        <mtext>softmax</mtext>
        <mfenced>
          <mfrac>
            <mrow><mi>Q</mi><msup><mi>K</mi><mi>T</mi></msup></mrow>
            <msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt>
          </mfrac>
        </mfenced>
        <mi>V</mi>
      </mtd>
    </mtr>
  </mtable>
</math>
```

---

## 6. Automation and Fixers

The improvement workflow uses two complementary tools:
1. **Automated Fixer** (`py_latex_to_mathml.py`) - Applies safe, deterministic LaTeX→MathML transformations
2. **Manual Intervention** - Required for semantic restructuring and context-aware decisions

### What Can Be Fixed Automatically

The automated fixer (`slash-cmd-scripts/src/py_latex_to_mathml.py`) provides **8 safe, enabled fixers**:

#### 1. **LaTeXToMathMLFixer** - Block Math Conversion
- **Fixes**: `$$...$$` block LaTeX expressions
- **Action**: Converts to `<math display="block">...</math>` using latex2mathml library
- **Category**: `math_formatting`
- **Safe**: ✅ Yes - Uses robust external library with fallback error handling
- **Example**:
  ```markdown
  # Before
  $$
  \frac{1}{1 + e^{-z}}
  $$

  # After
  <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
    <mfrac>
      <mn>1</mn>
      <mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>-</mo><mi>z</mi></mrow></msup></mrow>
    </mfrac>
  </math>
  ```

#### 2. **InlineMathToMathMLFixer** - Inline Math Conversion
- **Fixes**: `$x$` inline LaTeX expressions (avoids code blocks)
- **Action**: Converts to `<math display="inline">...</math>`
- **Category**: `math_formatting`
- **Safe**: ✅ Yes - Skips code blocks and protected contexts
- **Example**:
  ```markdown
  # Before
  The variable $x_1$ represents input.

  # After
  The variable <math display="inline"><msub><mi>x</mi><mn>1</mn></msub></math> represents input.
  ```

#### 3. **ParenMathToMathMLFixer** - Table Math Conversion
- **Fixes**: `\(...\)` inline LaTeX (commonly used in tables)
- **Action**: Converts to `<math display="inline">...</math>`
- **Category**: `math_formatting`
- **Safe**: ✅ Yes - Specifically designed for table contexts
- **Example**:
  ```markdown
  # Before
  | \(x_t\) | \([1, D_{in}]\) | Input vector |

  # After
  | <math display="inline"><msub><mi>x</mi><mi>t</mi></msub></math> | <math display="inline"><mfenced><mn>1</mn><mo>,</mo><msub><mi>D</mi><mtext>in</mtext></msub></mfenced></math> | Input vector |
  ```

#### 4. **MathCodeFenceToMathMLFixer** - GitHub Math Fence Conversion
- **Fixes**: GitHub-style ````math` code fences
- **Action**: Converts to `<math display="block">...</math>` with alignment support
- **Category**: `math_formatting`
- **Safe**: ✅ Yes - Preserves all math content
- **Example**:
  ````markdown
  # Before
  ```math
  x = y + z
  ```

  # After
  <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
    <mi>x</mi><mo>=</mo><mi>y</mi><mo>+</mo><mi>z</mi>
  </math>
  ````

#### 5. **ListFormattingFixer** - List Spacing
- **Fixes**: Missing blank lines before lists, excess blank lines between items
- **Action**: Adds blank line before list start, removes blank lines between consecutive items
- **Category**: `list_formatting`
- **Safe**: ✅ Yes - Preserves indentation levels and nested lists
- **Example**:
  ```markdown
  # Before
  Here are the features:
  - Feature one

  - Feature two

  # After
  Here are the features:

  - Feature one
  - Feature two
  ```

#### 6. **BoldFormattingFixer** - Structural Bold Removal
- **Fixes**: Excessive bold in specific patterns
- **Action**: Removes `**` from labels with colons, list markers, line starts
- **Category**: `bold_formatting`
- **Safe**: ⚠️  Partial - Only removes in specific structural patterns
- **Example**:
  ```markdown
  # Before
  **Section One:**
  - **Feature**: Description

  # After
  Section One:
  - Feature: Description
  ```

#### 7. **EmptyMathMLBlockFixer** - Cleanup
- **Fixes**: Empty `<math>` blocks and malformed tags
- **Action**: Removes empty blocks, cleans up conversion artifacts
- **Category**: `syntax`
- **Safe**: ✅ Yes - Only removes truly empty blocks
- **Example**:
  ```markdown
  # Before
  <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  </math>

  # After
  (removed)
  ```

#### 8. **AlignmentArtifactFixer** - Post-Conversion Cleanup
- **Fixes**: `<mi>&</mi>` and `<mi>\newline</mi>` artifacts from latex2mathml library
- **Action**: Removes LaTeX alignment markers (`&`) and newline commands (`\newline`) that were incorrectly treated as math identifiers
- **Category**: `syntax`
- **Safe**: ✅ Yes - Only removes known artifacts from automated conversion
- **Why Needed**: The latex2mathml library doesn't understand LaTeX `\begin{aligned}` environments properly, treating alignment markers as mathematical content
- **Example**:
  ```markdown
  # Before (from latex2mathml output)
  <math>
    <mi>&</mi><mtext>Calculate gradients</mtext><mi>\newline</mi>
    <mi>&</mi><mtext>Scale by factor</mtext>
  </math>

  # After (cleaned)
  <math>
    <mtext>Calculate gradients</mtext>
    <mtext>Scale by factor</mtext>
  </math>
  ```
- **Usage**: Run this fixer AFTER latex_to_mathml conversion as a cleanup pass:
  ```bash
  python3 slash-cmd-scripts/src/py_latex_to_mathml.py yourfile.md
  ```
  The fixer runs automatically as part of the default pipeline.

### What Requires Manual Fixes

The following issues **cannot be safely automated** and require human judgment:

#### 1. **Math-Prose Separation** ⚠️ MANUAL ONLY
- **Issue**: Mathematical notation embedded in prose sentences
- **Why Manual**: Requires semantic understanding and sentence restructuring
- **Solution**: Extract math from prose and promote to dedicated `<math display="block">` blocks

#### 2. **Content Classification** ⚠️ MANUAL ONLY
- **Issue**: Determining if content should be MathML or clean Markdown
- **Why Manual**: Requires understanding of semantic meaning
- **Solution**: Use MathML for mathematical content, Markdown for descriptive content

#### 3. **Semantic MathML Quality** ⚠️ MANUAL ONLY
- **Issue**: Automated conversion may produce valid but non-optimal MathML
- **Why Manual**: Library may not choose the best semantic tags for all contexts
- **Solution**: Review converted MathML for proper use of `<mtext>`, `<mi>`, `<mo>`, `<mn>`

### Automated Fixer Usage

**Basic Usage:**
```bash
# Fix all issues automatically
python3 slash-cmd-scripts/src/py_latex_to_mathml.py yourfile.md

# Preview changes without modifying file
python3 slash-cmd-scripts/src/py_latex_to_mathml.py --dry-run yourfile.md

# Use only specific fixer
python3 slash-cmd-scripts/src/py_latex_to_mathml.py --fixer latex_to_mathml yourfile.md
python3 slash-cmd-scripts/src/py_latex_to_mathml.py --fixer inline_math_to_mathml yourfile.md

# List all available fixers
python3 slash-cmd-scripts/src/py_latex_to_mathml.py --list-fixers
```

### Lessons Learned from Large-Scale Conversions

Based on converting large documents (800+ lines with 62 MathML blocks), here are key insights:

#### 1. **Hybrid Approach Works Best**
- **Automated conversion** handles ~80% of the work (simple equations, inline math, tables)
- **Manual refinement** needed for ~20% (aligned equations, parameter tables, descriptive blocks)
- **Post-processing cleanup** (AlignmentArtifactFixer) handles conversion artifacts

#### 2. **latex2mathml Library Limitations**
The `latex2mathml` library has known issues with:
- **Aligned environments**: Treats `&` and `\newline` as math identifiers instead of structural markers
- **Semantic quality**: May not choose optimal MathML tags (e.g., `<mi>log</mi>` vs `<mo>log</mo>`)
- **Complex tables**: Multi-line parameter descriptions need manual `<mtable>` restructuring

**Solution**: Use AlignmentArtifactFixer as automatic post-processor, then manually refine critical educational content.

#### 3. **Prioritize by Importance**
For documents with many equations:
1. **Manual conversion first** for critical pedagogical content (main formulas, key theorems)
2. **Automated batch conversion** for remaining standard equations
3. **Cleanup pass** to remove artifacts
4. **Selective manual fixes** for most visible issues only

#### 4. **When to Keep LaTeX**
Consider keeping LaTeX if:
- Document has >50 complex aligned equations
- Time constraints make manual refinement impractical
- Target audience uses MathJax-enabled platforms anyway
- Conversion quality trade-offs outweigh browser compatibility benefits

#### 5. **Validation Strategy**
Always follow this validation workflow:
```bash
# 1. Automated conversion
python3 slash-cmd-scripts/src/py_latex_to_mathml.py yourfile.md

# 2. Check what was fixed
git diff --stat yourfile.md

# 3. Verify no LaTeX remains
echo "LaTeX blocks: $(grep -c '^\$\$' yourfile.md)"
echo "Inline LaTeX: $(grep -c '\\(' yourfile.md)"
echo "Artifacts: $(grep -c '<mi>&</mi>' yourfile.md)"

# 4. Manual review of critical sections
git diff yourfile.md | grep -A 10 -B 2 "Section 9.1"

# 5. Test rendering in browser
# (Open in browser with MathML Core support)
```

#### 6. **Performance Metrics**
Typical conversion yields for an 800-line document:
- **Automated fixers**: 38-60 LaTeX blocks converted
- **Cleanup pass**: 80+ artifacts removed
- **Manual refinements**: 10-15 critical equations
- **Time**: 30-60 minutes for high-quality conversion
- **Success rate**: 100% LaTeX elimination, 95%+ semantic correctness

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

# Or edit with AI

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


