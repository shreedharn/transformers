---
argument-hint: [filename]
description: Standardize LaTeX mathematical typography in a markdown document
allowed-tools: Read, Edit, MultiEdit, Bash,Bash(grep:*), Bash(sed:*), Bash(awk:*)
---

Transform the markdown document $1 with mixed mathematical notation into a professionally formatted document with consistent LaTeX typography.

**IMPORTANT**: Only apply LaTeX standardization to content that contains mathematical expressions ($$, \begin{aligned}, mathematical symbols, equations). Do NOT modify standalone markdown formatting like **Features:**, **Examples:**, **Learned Weights:**: Some Non Math text, or **Example Email:** that do not contain mathematical content.

Apply the following standardizations to $1:

## 1. MathJax Format Standardization
- Convert all `$$expression$$` notation to `\begin{aligned}...\end{aligned}` blocks
- Wrap all LaTeX blocks in `$$` delimiters for proper display
- Add `{\textstyle...}` wrapper when inline-style rendering is needed
- **NEVER** have markdown bullets (`-`, `*`) containing MathJax expressions (`$$`, `\begin{aligned}`)
- **ALWAYS** separate mathematical content into dedicated `$$\begin{aligned}...\end{aligned}$$` blocks
- **REPHRASE** sentences if necessary to maintain meaning while separating elements

## 2. Block Consolidation with Boundary Respect
- Identify multiple separate `\begin{aligned}` blocks that are consecutive (without intervening non-mathematical text)
- **BOUNDARY RULE**: Stop consolidation when encountering a line with non-mathematical text between blocks
- Only merge blocks that are directly adjacent or separated only by empty lines
- Use `\newline` for line separation within consolidated blocks
- Maintain proper alignment using `&` characters for consistent positioning
- **Example boundary**: If you see `\begin{aligned}...`, then `**Some Text:**`, then another `\begin{aligned}...`, do NOT merge across the text boundary
- Do not mix markdown bullets with MathJax elements.

## 3. Typography Standards
- Use `\mathbf{expression}` for bold mathematical expressions within LaTeX blocks (not regular `**text**`)
- Wrap descriptive text in `\text{description}` for proper LaTeX text rendering
- Replace markdown bullets (`-`) with LaTeX bullets (`\bullet`) ONLY in mathematical contexts that contain equations or symbols
- Use `\quad` for consistent spacing between mathematical elements

## 4. Underscore Escaping (Critical for Markdown Compatibility)
- **ALWAYS** escape raw underscores in LaTeX expressions as `\_` to prevent Markdown italic parsing conflicts
- Detect patterns like `\mathcal{L}_{CLM}`, `x_{t+1}`, `W^{(1)}`, etc. that contain unescaped underscores
- Transform `_{subscript}` to `\_{subscript}` within all LaTeX blocks (`$$...$$`, `\begin{aligned}...\end{aligned}`)
- **CRITICAL**: This prevents LaTeX math from breaking due to Markdown interpreting `_text_` as italics
- Apply escaping to ALL underscore usage in mathematical contexts: subscripts, variable names, function names

## 5. Content Integration
- Move standalone descriptive text into LaTeX blocks as introductory lines ONLY when it directly relates to mathematical expressions
- Use alignment characters (`&`) to create visual hierarchy within mathematical contexts
- Integrate mathematical symbols properly (e.g., `\rightarrow` for arrows)
- Do NOT move general section headers, examples, or feature lists into LaTeX blocks

## 6. Structural Consistency
- Ensure all related mathematical content follows the same formatting pattern
- Maintain proper spacing with `\newline` between related items
- Use `\textbf{}` for bold text within LaTeX environments


## Output Format Examples

### Before (Mixed Notation):
```markdown
- **$$x W$$:**
- **$$+ b$$:** Bias allows shifting the activation threshold
- **\begin{aligned} \sigma(\cdot) \end{aligned}:** Non-linearity enables learning complex patterns

\begin{aligned} h^{(1)} &= \sigma^{(1)}(x W^{(1)} + b^{(1)}) \end{aligned}

\begin{aligned} h^{(2)} &= \sigma^{(2)}(h^{(1)} W^{(2)} + b^{(2)}) \end{aligned}
```

### After (Standardized):
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

### Example 5: Boundary Respect (DO NOT MERGE)

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

**Key Point**: The markdown text "**Layer Naming Convention:**" creates a natural boundary that prevents consolidation.

### Example 6: Underscore Escaping (Critical Fix)

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

## Implementation Strategy
1. **Analyze file structure** using grep to identify mathematical content boundaries
2. **Detect and escape underscores** in all LaTeX expressions using regex patterns
3. **Apply transformations** respecting natural text boundaries
4. **Validate results** ensuring no unintended merging occurred and all underscores are properly escaped
5. **Generate summary** of changes made

Apply these changes systematically throughout the file to create consistent, professional LaTeX typography while respecting document structure boundaries.