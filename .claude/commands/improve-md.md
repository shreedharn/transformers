---
argument-hint: [filename]
description: Comprehensively improve markdown file by separating math from prose and applying professional LaTeX typography
allowed-tools: Read, Edit, MultiEdit, Bash, Bash(grep:*), Bash(sed:*), Bash(awk:*)
---

Comprehensively improve the markdown document `$1` by applying content-appropriate formatting: clean math-prose separation, professional LaTeX typography for mathematical content, and polished Markdown formatting for descriptive content.

**UNIFIED PRINCIPLE**: Apply the right format for the content type while maintaining strict separation between mathematical notation and prose text.

## Core Improvement Strategy

### 1. Content Classification and Separation

**BEFORE applying any transformations, classify content and separate elements:**

#### Mathematical Content → Professional LaTeX Blocks:
- Content with mathematical symbols, equations, variables, formulas
- Expressions using `$$`, `\begin{aligned}`, mathematical notation
- Parameter descriptions with dimensions or calculations
- Any content mixing mathematical variables with descriptive text

#### Descriptive Content → Clean Markdown Formatting:
- Architecture descriptions, use cases, training objectives
- Feature lists, capabilities, or narrative explanations
- Simple structured information without mathematical elements
- Plain text descriptions about functionality or characteristics

#### Critical Separation Rule:
**Do not include mathematical symbols, variables, or notation (e.g., `x`, `W`, `\alpha`, `O(n)`, `→`, fractions) inside Markdown sentences, bullets, headings, or captions. Promote them to dedicated display blocks (or use plain words).**

### 2. Math-Prose Separation Rules

* **Never inline math with Markdown structures.** No `$$`/`\begin{aligned}` on the same line as list markers, headings, table cells, or paragraph sentences.
* **Extract and promote.** Move any math found in bullets, numbered items, tables, or paragraphs into dedicated `$$ \begin{aligned} … \end{aligned} $$` blocks.
* **One concept → one block.** Use consolidated LaTeX blocks with `\newline` for line breaks.
* **List safety.** Bullet/number → blank line → indented (2–3 spaces) display block; don't indent 4+ spaces.
* **Tables stay inline.** Use `\(...\)` in tables; no display math.
* **Rewrite when necessary.** Rephrase sentences that mix mathematical notation with prose.

### 3. Professional LaTeX Typography Standards

* **Display format:** Use `$$ \begin{aligned} … \end{aligned} $$` for all mathematical content
* **Single-line equations:** Always wrap in `\begin{aligned}` blocks, never use standalone `$$equation$$`
* **Mathematical notation:** Use `$$` for ALL mathematical expressions (variables, equations, subscripts), never single `$`
* **Table math notation:** Use `$$` for all mathematical expressions in tables
* **Inline math:** Use `$$` even for simple variables like `$$x\_1$$`, `$$h\_t$$`
* **Line breaks:** Use `\newline` (not `\\`) for line separation within blocks
* **Text labels:** Use `\text{description}` for descriptive text within math blocks
* **Bold mathematics:** Use `\mathbf{expression}` (Latin) and `\boldsymbol{expression}` (Greek/symbols)
* **Spacing:** Use `\,`, `\;`, `\quad`, `\qquad` for mathematical spacing
* **Alignment:** Use `&` characters for consistent positioning across lines
* **Underscore escaping:** Transform `_{subscript}` to `\_{subscript}` for Markdown compatibility
* **Context-aware escaping:** Escape underscores ONLY in LaTeX contexts (`\_`), NEVER in code blocks, ASCII art, or Python code where normal underscores (`_`) must be preserved

### 4. List Formatting and Separation Rules

* **Blank line requirement:** Ensure blank lines before every markdown list (bulleted and numbered)
* **NO blank lines between list items:** Only add blank lines before the start of lists, never between individual list items
* **List safety:** Never place mathematical content directly in list markers
* **Proper separation:** Lists after any content type require blank line separation:
  - After prose paragraphs, colons, bold/italic text
  - After parenthetical statements, inline code/math expressions
  - After blockquotes, table rows, HTML blocks, code fences
  - After MathJax display blocks or other structural elements
* **Consistent markers:** Use consistent bullet markers throughout document (avoid mixing -, *, +)
* **Nested lists:** Maintain proper indentation and blank line structure

#### List Formatting Examples:

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

❌ **Wrong - Escaped underscores in code:**
```python
def my\_function(param\_name):
    return param\_name * 2
```

✅ **Correct - Normal underscores in code:**
```python
def my_function(param_name):
    return param_name * 2
```

### 5. Block Consolidation with Boundary Respect

* **Consolidate** consecutive `$$\begin{aligned}...\end{aligned}$$` blocks **only** when they represent one concept and are separated by empty lines only
* **Stop consolidation** at any non-mathematical text boundary:
  - Headings (`^#{1,6}\s`), lists (`^\s*([-*+]|[0-9]+\.)\s`)
  - Blockquotes (`^\s*>`), tables (`^\s*\|`), code fences (` ``` `)
  - HTML blocks (`^\s*<`), or descriptive text
* **Maintain context:** Keep related mathematical expressions together while respecting logical boundaries

### 6. Content-Specific Formatting Rules

#### Keep as Clean Markdown:
```markdown
### Encoder-Decoder: T5 Family

**Structure**: Encoder (bidirectional) + Decoder (causal) with cross-attention
**Training**: Various objectives (span corruption, translation, etc.)
**Use cases**: Translation, summarization, structured tasks
```

#### Fix as Professional LaTeX:
```markdown
The attention mechanism computes query, key, and value matrices with specific dimensions:

$$
\begin{aligned}
Q, K, V &: \text{Query, key, and value matrices} \newline
d_k &= 512 : \text{Key dimension for computational efficiency} \newline
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V
\end{aligned}
$$
```

## Python Detector Usage Instructions

### Step 1: Discover Available Options

The detector provides comprehensive analysis with clear separation between categories and individual detectors. Start by exploring available options:

```bash
# See main help and usage patterns
python3 slash-cmd-scripts/src/py_improve_md.py --help

# List all 6 detector categories
python3 slash-cmd-scripts/src/py_improve_md.py --list-categories

# List all 26 individual detector names
python3 slash-cmd-scripts/src/py_improve_md.py --list-detectors

# Get detailed examples for categories
python3 slash-cmd-scripts/src/py_improve_md.py --list-categories --help

# Get detailed examples for detectors
python3 slash-cmd-scripts/src/py_improve_md.py --list-detectors --help
```

### Step 2: Choose Detection Strategy

**Strategy A: Run all detectors (comprehensive analysis):**
```bash
python3 slash-cmd-scripts/src/py_improve_md.py "$1"
```

**Strategy B: Run specific categories (targeted analysis):**
```bash
# Focus on inline math issues (variables mixed with prose)
python3 slash-cmd-scripts/src/py_improve_md.py --category inline_math "$1"

# Focus on list formatting problems
python3 slash-cmd-scripts/src/py_improve_md.py --category list_formatting "$1"

# Focus on display math positioning issues
python3 slash-cmd-scripts/src/py_improve_md.py --category display_math "$1"

# Focus on LaTeX alignment structure
python3 slash-cmd-scripts/src/py_improve_md.py --category alignment "$1"

# Get help for specific category
python3 slash-cmd-scripts/src/py_improve_md.py --category inline_math --help
```

**Strategy C: Run individual detectors (pinpoint analysis):**
```bash
# Target specific known issues
python3 slash-cmd-scripts/src/py_improve_md.py --detector missing_blank_line_before_list_items_after_text "$1"
python3 slash-cmd-scripts/src/py_improve_md.py --detector paragraphs_with_inline_math "$1"
python3 slash-cmd-scripts/src/py_improve_md.py --detector list_marker_lines_with_display_math "$1"

# Get help for specific detector
python3 slash-cmd-scripts/src/py_improve_md.py --detector missing_blank_line_before_list_items_after_text --help
```

**Strategy D: Debug mode with verbose logging:**
```bash
python3 slash-cmd-scripts/src/py_improve_md.py --verbose "$1"
```

This provides a comprehensive or targeted report of markdown formatting and LaTeX issues, organized by categories with line numbers and context for easy identification and fixing. After running the detector, perform a final AI scan to catch any edge cases or patterns the automated detection might miss.

### Usage Guidelines

**Getting Started:**
- **First-time analysis**: Use `--help` to see main usage patterns and available options
- **Explore categories**: Use `--list-categories` to see 6 main detector groups
- **Explore detectors**: Use `--list-detectors` to see all 26 individual detector names
- **Get examples**: Add `--help` to any `--list-*`, `--category`, or `--detector` command for detailed examples
- **Debugging**: Use `--verbose` flag when troubleshooting detection issues

**Smart Detection Strategy:**
1. **Quick scan**: Run `--category list_formatting` first (catches most common issues)
2. **Math-heavy documents**: Use `--category inline_math` and `--category alignment` for mathematical content
3. **Display math issues**: Use `--category display_math` for positioning problems
4. **Comprehensive check**: Run all detectors when thorough analysis is needed
5. **Follow-up**: Use specific `--detector` commands based on initial findings

**Progressive Approach:**
- Start with categories (`--category`) for broad issue types
- Drill down to specific detectors (`--detector`) for pinpoint fixes
- Use help system (`--help`) to understand what each option fixes before running

**Comprehensive Detection Categories:**

**1. Inline Math Category (5 detectors):**
- Mathematical variables and equations mixed with prose text
- Inline math in headings and list items (forbidden patterns)
- Display math delimiters used incorrectly in prose context
- Mathematical tokens scattered throughout descriptive text

**2. Display Math Category (8 detectors):**
- Mathematical expressions in wrong locations (lists, headings, tables)
- Math blocks with incorrect positioning and indentation
- Adjacent math blocks that need consolidation
- Over-indented display math and improper alignment

**3. List Formatting Category (3 detectors):**
- Missing blank lines before lists after text, math, and other content
- Incorrect spacing between list items (should have no blank lines)
- Lists appearing after various content types without proper separation

**4. Alignment Category (5 detectors):**
- Math expressions missing professional `\begin{aligned}` structure
- Mismatched and malformed aligned blocks
- Single-line equations without proper LaTeX wrapper
- Incorrect positioning of `\end{aligned}` statements

**5. Structural Category (2 detectors):**
- Bold text spacing issues and missing blank lines after headings
- Escaped underscores in code blocks (breaks syntax highlighting)

**6. Syntax Category (4 detectors):**
- Unpaired mathematical delimiters and stray braces
- Bold markdown mixed within LaTeX expressions
- Context-specific underscore escaping violations
- Math code fence blocks (````math` that should be MathJax)


## Quality Verification Checklist

After running the improvement detector:

**Math-Prose Separation:**
- [ ] **No mathematical symbols appear in Markdown prose** (paragraphs, bullets, headings, captions)
- [ ] **No list lines contain `$$` or `\begin{aligned}`**
- [ ] **Clean separation** between mathematical notation and prose text

**List Formatting:**
- [ ] **Blank lines before all lists** (bulleted and numbered)
- [ ] **NO blank lines between individual list items** (only before list start)
- [ ] **Consistent bullet markers** throughout document (no mixing -, *, +)
- [ ] **Proper separation after all content types** (prose, colons, bold text, blockquotes, tables, code blocks)
- [ ] **Blank lines after bold headings with colons** (e.g., **Section**:)
- [ ] **No mathematical content in list markers**

**LaTeX Typography:**
- [ ] **All equations use professional `$$ \begin{aligned} … \end{aligned} $$` blocks**
- [ ] **No single-line `$$equation$$` without aligned wrapper**
- [ ] **Use `$$` for ALL mathematical expressions, never single `$`**
- [ ] **Tables use `$$` for all math notation**
- [ ] **Mathematical content uses consistent LaTeX typography**
- [ ] **All underscores properly escaped** in LaTeX expressions (`\_{subscript}`)
- [ ] **All underscores properly escaped** in all math (`$$x\_1$$`, `$$h\_{t-1}$$`)
- [ ] **No escaped underscores in code blocks, ASCII art, or Python code** (preserve normal `_` syntax)
- [ ] **Proper mathematical formatting** (`\mathbf{}`, `\text{}`, spacing)

**Content Classification:**
- [ ] **Descriptive content uses clean Markdown formatting**
- [ ] **Mathematical content uses professional LaTeX blocks**
- [ ] **Multiple inline math converted to display blocks** for better readability
- [ ] **Appropriate content classification** applied throughout document
- [ ] **Block consolidation respects content boundaries**

## Comprehensive Examples

### Math-Prose Separation Examples:

**Before (Mixed):**
```markdown
The attention mechanism computes Q, K, and V matrices where d_k = 512 for efficiency.
- Computing attention scores uses formula α_ij = softmax(q_i * k_j)
```

**After (Separated):**
```markdown
The attention mechanism computes query, key, and value matrices with specific dimensions for efficiency:

$$
\begin{aligned}
Q, K, V &: \text{Query, key, and value matrices} \newline
d\_k &= 512 : \text{Key dimension for computational efficiency} \newline
\alpha\_{ij} &= \text{softmax}(q\_i \cdot k\_j) : \text{Attention score computation}
\end{aligned}
$$
```

### List Formatting Examples:

**Before (Incorrect):**
```markdown
Implementation details:
- Parallel computation enabled
- Memory optimization active

Text with (parenthetical content)
* Bullet after parentheses

> This is a blockquote
- List after blockquote

**Features:**
1. Fast processing using -O3 flag
2. Low memory with cache optimization
```

**After (Correct):**
```markdown
Implementation details:

- Parallel computation enabled
- Memory optimization active

Text with (parenthetical content)

- Bullet after parentheses

> This is a blockquote

- List after blockquote

**Features:**

1. Fast processing using optimization flags
2. Low memory with cache optimization
```

### Content Classification Examples:

**Before (Wrong Format):**
```markdown
$$
{\textstyle
\begin{aligned}
\textbf{Structure} &: \text{Encoder (bidirectional) + Decoder (causal)} \newline
\textbf{Training} &: \text{Various objectives (span corruption, translation)} \newline
\textbf{Use cases} &: \text{Translation, summarization, structured tasks}
\end{aligned}
}
$$
```

**After (Correct Format):**
```markdown
**Structure**: Encoder (bidirectional) + Decoder (causal) with cross-attention
**Training**: Various objectives (span corruption, translation, etc.)
**Use cases**: Translation, summarization, structured tasks
```

### Extended List Formatting Examples:

**Example 1 - Basic List Separation (from fix-list.md):**

**Before (Incorrect):**
```markdown
where:
- Query matrix (what information to retrieve)
- Key matrix (what information is available)
```

**After (Correct):**
```markdown
where:

- Query matrix (what information to retrieve)
- Key matrix (what information is available)
```

**Example 2 - Numbered Lists (from fix-list.md):**

**Before (Incorrect):**
```markdown
**Implementation Details:**
1. **Parallel computation**: All heads computed simultaneously
2. **Linear projections**: Simple matrix multiplications
```

**After (Correct):**
```markdown
**Implementation Details:**

1. **Parallel computation**: All heads computed simultaneously
2. **Linear projections**: Simple matrix multiplications
```

### Math-Markdown Separation Examples (from simplify-md.md):

**Example 1: Mixed Notation → Standardized**

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

**Example 2: Parameter Counting**

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

**Example 3: Activation Functions**

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

**Example 4: Network Example**

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

**Example 5: Boundary Respect (Do NOT Merge)**

**Before (Correct - Should NOT be consolidated):**
```markdown
\begin{aligned} h^{(1)} &= \sigma^{(1)}(x W^{(1)} + b^{(1)}) \end{aligned}

**Layer Naming Convention:**

\begin{aligned} y &= h^{(L-1)} W^{(L)} + b^{(L)} \end{aligned}
```

**After (Correct - Keep separate):**
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

### LaTeX Typography Examples:

**Example 1: Basic Typography Issues**

**Before (Typography Issues):**
```markdown
$$
\begin{aligned}
W_1 &= **Weight matrix** \newline
d_model &= 768 \text{ dimensions}
\end{aligned}
$$
```

**After (Professional Typography):**
```markdown
$$
\begin{aligned}
W\_1 &= \mathbf{Weight matrix} \newline
d\_{\text{model}} &= 768 \text{ dimensions}
\end{aligned}
$$
```

**Example 2: Underscore Escaping (Critical Fix from fix-latex.md)**

**Before (Broken - Markdown interprets underscores as italics):**
```markdown
$$\mathcal{L}_{CLM} = -\sum_{t=1}^{n-1} \log P(x_{t+1} | x_1, \ldots, x_t)$$

The loss function $\mathcal{L}_{CLM}$ uses subscripts $x_{t+1}$ and $x_t$.
```

**After (Fixed - Escaped underscores):**
```latex
$$\mathcal{L}\_{CLM} = -\sum\_{t=1}^{n-1} \log P(x\_{t+1} | x\_1, \ldots, x\_t)$$

The loss function $\mathcal{L}\_{CLM}$ uses subscripts $x\_{t+1}$ and $x\_t$.
```

**Critical patterns to escape:**
- `_{subscript}` → `\_{subscript}`
- `x_{t+1}` → `x\_{t+1}`
- `W^{(l)}_{ij}` → `W^{(l)}\_{ij}`
- `\mathcal{L}_{CLM}` → `\mathcal{L}\_{CLM}`
- `h^{(l)}_{t}` → `h^{(l)}\_{t}`

### Critical Edge Cases (Common Missed Issues):

**Edge Case 1: Single-Line LaTeX Without Aligned Wrapper**

**Before (Incorrect - Missing professional structure):**
```markdown
$$h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)$$
```

**After (Correct - Professional aligned block):**
```markdown
$$
\begin{aligned}
h\_t = \tanh(x\_t W\_{xh} + h\_{t-1} W\_{hh} + b\_h)
\end{aligned}
$$
```

**Edge Case 2: Single $ in Tables (Should be $$)**

**Before (Incorrect - Single $ in table):**
```markdown
| Term | Size | Meaning |
|------|------|---------|
| $x_t$ | $[1, E]$ | Current input |
| $h_{t-1}$ | $[1, H]$ | Past memory |
```

**After (Correct - $$ for proper display math):**
```markdown
| Term | Size | Meaning |
|------|------|---------|
| $$x\_t$$ | $$[1, E]$$ | Current input |
| $$h\_{t-1}$$ | $$[1, H]$$ | Past memory |
```

**Edge Case 3: Escaped Underscores in Non-LaTeX Context**

**Before (Incorrect - Escaped underscores in ASCII art):**
```markdown
```
Past Memory    Current Input
    h\_{t-1}  +      x\_t
       ↓              ↓
   h\_{t-1} W\_{hh} + x\_t W\_{xh} + b\_h
```
```

**After (Correct - Unescaped in code blocks):**
```markdown
```
Past Memory    Current Input
    h_{t-1}  +      x_t
       ↓              ↓
   h_{t-1} W_{hh} + x_t W_{xh} + b_h
```
```

**Edge Case 4: Unescaped Underscores in Inline Math**

**Before (Incorrect - Unescaped underscores in inline math):**
```markdown
**Input:** $x_1 = [0.5, 0.2]$, **Memory:** $h_0 = [0.0, 0.0]$

The hidden state $h_{t-1}$ carries memory forward.

RNNs produce a sequence ($h_1, h_2, h_3, ..., h_T$) over time.
```

**After (Correct - $$ with escaped underscores in all math contexts):**
```markdown
**Input:** $$x\_1 = [0.5, 0.2]$$, **Memory:** $$h\_0 = [0.0, 0.0]$$

The hidden state $$h\_{t-1}$$ carries memory forward.

RNNs produce a sequence ($$h\_1$$, $$h\_2$$, $$h\_3$$, ..., $$h\_T$$) over time.
```

**Edge Case 5: Single $ Should Be $$ for All Math**

**Before (Incorrect - Single $ for mathematical variables):**
```markdown
**Key Insight:** Each hidden state $h_t$ encodes information up to time $t$.

The variable $x_1$ represents the input vector.
```

**After (Correct - $$ for all mathematical expressions):**
```markdown
**Key Insight:** Each hidden state $$h\_t$$ encodes information up to time $$t$$.

The variable $$x\_1$$ represents the input vector.
```

**Edge Case 6: Bold Text with Colon Missing Blank Line**

**Before (Incorrect - No blank line after bold heading with colon):**
```markdown
**RNN Limitations**:
- Vanishing gradients limited long-range dependencies
- Sequential processing prevented parallelization
```

**After (Correct - Blank line after bold heading):**
```markdown
**RNN Limitations**:

- Vanishing gradients limited long-range dependencies
- Sequential processing prevented parallelization
```

**Edge Case 7: Multiple Inline Math Should Be Display Block**

**Before (Incorrect - Multiple inline math in prose):**
```markdown
Initially, $$W\_{xh}$$, $$W\_{hh}$$, and $$b\_h$$ are random numbers.
```

**After (Correct - Display block for multiple variables):**
```markdown
Initially, the following parameters are random numbers:

$$
\begin{aligned}
W\_{xh}, W\_{hh}, b\_h \quad \text{(Input weights, hidden weights, and bias)}
\end{aligned}
$$
```

**Edge Case 8: Mathematical Variables Without Dollar Signs (Unwrapped Math)**

**Before (Incorrect - Mathematical variables unwrapped in parentheses):**
```markdown
**Key Point:** The same weights are used at every time step, but gradients flow back through the entire sequence (W_{xh}, W_{hh} are reused at each time step)
```

**After (Correct - Mathematical variables in proper display block):**
```markdown
**Key Point:** The same weights are used at every time step, but gradients flow back through the entire sequence:

$$
\begin{aligned}
\text{Shared weights:} \quad W\_{xh}, W\_{hh} \quad \text{(reused at each time step)}
\end{aligned}
$$
```

**Edge Case 9: Unnecessary Comma Between Mathematical Expressions**

**Before (Incorrect - Comma between separate labeled math expressions):**
```markdown
**Input:** $$x\_1 = [0.5, 0.2]$$, **Memory:** $$h\_0 = [0.0, 0.0]$$
```

**After (Correct - No comma between separate labeled expressions):**
```markdown
**Input:** $$x\_1 = [0.5, 0.2]$$ **Memory:** $$h\_0 = [0.0, 0.0]$$
```

**Edge Case 10: Math Code Fence Blocks (````math`)**

**Before (Incorrect - GitHub-style ```math fence):**
````markdown
The Derivative as Slope: For any function $f(x)$, the derivative $\frac{df}{dx}$ tells us the slope at any point:

```math
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
```
````

**After (Correct - Professional MathJax with prose in text blocks):**
```markdown
The Derivative as Slope:

$$
{\textstyle
\begin{aligned}
\text{For any function } f(x), \text{ the derivative } \frac{df}{dx} \text{ tells us the slope at any point:} \newline
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\end{aligned}
}
$$
```

**Why this transformation is important:**

1. **Separates prose from math**: Descriptive text is moved into `\text{...}` blocks within the aligned environment
2. **Professional LaTeX format**: Uses `$${\textstyle\begin{aligned}...\end{aligned}}$$` instead of GitHub's ````math` fence
3. **Better rendering**: MathJax renders more consistently across platforms than ````math` blocks
4. **Unified style**: Maintains consistency with other mathematical expressions in the document

## Manual Fixing Guidelines for Math-Prose Separation

The detector identifies "Paragraphs with inline math (promote to block)" violations that **require manual fixing** because they need semantic understanding and sentence restructuring.

### Why Manual Fixing is Required

**Automated fixers cannot:**
- Understand the semantic meaning of prose
- Decide which descriptive text belongs in `\text{...}` blocks
- Rephrase sentences to maintain educational flow
- Determine the best way to consolidate related mathematical concepts

**The detector provides:**
- Line numbers of violations
- Context to understand the issue
- But fixing requires human judgment

### Step-by-Step Manual Fixing Process

**Step 1: Run the detector to identify violations**

```bash
python3 slash-cmd-scripts/src/py_improve_md.py --detector paragraphs_with_inline_math yourfile.md
```

**Step 2: For each violation, apply one of these rephrasing patterns:**

#### Pattern 1: "where X is..." → Dedicated block

**Before:**
```markdown
where $$\sigma$$ is an activation function like ReLU or sigmoid.
```

**After:**
```markdown
with the activation function defined as:

$$
\begin{aligned}
\sigma \quad \text{(activation function like ReLU or sigmoid)}
\end{aligned}
$$
```

#### Pattern 2: "Key Insight: X controls..." → Extract notation

**Before:**
```markdown
Key Insight: The same learning rate $$\alpha$$ controls the step size for all parameters.
```

**After:**
```markdown
Key Insight: The learning rate plays the same role across all parameters:

$$
\begin{aligned}
\alpha \quad \text{(controls step size for all parameters)}
\end{aligned}
$$
```

#### Pattern 3: "Understanding X" → Separate definition

**Before:**
```markdown
Understanding $$\alpha$$ (alpha): These are the attention weights. The $$\alpha_i$$ values all add up to 1.
```

**After:**
```markdown
Understanding the attention weights: These tell us how much to focus on each element. The weights are defined as:

$$
\begin{aligned}
\alpha &\quad \text{(attention weight vector)} \newline
\alpha_i &\quad \text{(individual weights that sum to 1)}
\end{aligned}
$$
```

#### Pattern 4: Multiple inline expressions → Consolidated block

**Before:**
```markdown
Let's minimize $$f(x) = x^2$$ starting from $$x = 3$$:

Step 1: Compute the derivative: $$\frac{df}{dx} = 2x$$

Step 2: Choose learning rate: $$\alpha = 0.1$$
```

**After:**
```markdown
Let's minimize this function with the following setup:

$$
\begin{aligned}
f(x) &= x^2 \quad \text{(function to minimize)} \newline
x_0 &= 3 \quad \text{(starting point)} \newline
\frac{df}{dx} &= 2x \quad \text{(derivative)} \newline
\alpha &= 0.1 \quad \text{(learning rate)}
\end{aligned}
$$
```

#### Pattern 5: "Residual Block: X approximates..." → Rephrase

**Before:**
```markdown
Residual Block: $$\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l)$$ approximates:
```

**After:**
```markdown
Residual Block approximates the following:

$$
\begin{aligned}
\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l)
\end{aligned}
$$
```

### General Principles

1. **Rephrase to separate**: Rewrite the sentence so math isn't embedded in prose
2. **Use `\text{...}` for labels**: Put descriptive text inside aligned blocks with `\text{...}`
3. **Consolidate related math**: Group multiple related expressions into one block
4. **Maintain educational flow**: Keep the narrative clear and readable
5. **Preserve meaning**: Ensure the mathematical and pedagogical intent is unchanged

### Common Sentence Patterns to Rephrase

| Original Pattern | Rephrased Pattern |
|-----------------|-------------------|
| "If we minimize $$f(x)$$, we..." | "To minimize the function: $$f(x)$$" |
| "where $$X$$ is..." | "with X defined as: $$X$$" |
| "The $$\delta$$ terms flow..." | "The error terms flow: $$\delta$$" |
| "Understanding $$X$$: it means..." | "Understanding X: $$X$$ (definition)" |
| "For $$X$$: $$Y$$" | "For the parameters: $$X, Y$$" |

### Workflow

1. Run detector to get all violations with line numbers
2. Review each violation in context
3. Choose appropriate rephrasing pattern
4. Apply manual edit to separate math from prose
5. Re-run detector to verify fix
6. Continue until all violations resolved

### Important Notes

- **Do not automate this**: Each violation requires semantic understanding
- **Preserve pedagogy**: Maintain the teaching narrative
- **Test rendering**: Ensure math blocks render correctly
- **Check flow**: Read the section aloud to verify it still flows well

The improve-md command provides comprehensive markdown enhancement by intelligently applying the right formatting approach for each content type while maintaining strict separation between mathematical and descriptive elements. The python detector CLI design with separate `--category` and `--detector` switches enables progressive discovery and targeted analysis of formatting issues.