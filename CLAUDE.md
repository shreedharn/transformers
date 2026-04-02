## Repo Structure

```
transformers/
├── nn_intro.md                 # AI/ML/DL foundations, basic neural network concepts
├── mlp_intro.md                # MLP step-by-step tutorial, multi-layer network fundamentals
├── rnn_intro.md                # RNN step-by-step tutorial, sequential modeling fundamentals
├── transformers_fundamentals.md # Transformer architecture & core concepts, attention, layers
├── transformers_advanced.md    # Training, optimization & deployment, fine-tuning, quantization
├── transformers_math1.md       # Mathematical foundations Part 1 — intuition and core concepts
├── transformers_math2.md       # Mathematical foundations Part 2 — advanced concepts and scaling
├── math_quick_ref.md           # Math reference: formulas, intuitions, neural network applications
├── knowledge_store.md          # LLM weights vs vector stores — internalized vs external knowledge
├── pytorch_ref.md              # PyTorch implementation guide, code patterns, practical examples
├── glossary.md                 # Technical terms and definitions
├── further.md                  # Further reading: foundation papers, modern developments
├── docs/                       # MkDocs source directory (mirrors root *.md for MkDocs site)
├── pynb/                       # Jupyter notebooks
│   ├── basic/                  # Foundational ML notebooks
│   ├── dl/                     # Deep learning notebooks
│   ├── math_ref/               # Math reference notebooks
│   └── vector_search/          # Vector search notebooks
├── slash-cmd-scripts/src/      # Markdown automation scripts
│   ├── py_fix_md.py            # Automated markdown/LaTeX fixer
│   ├── py_improve_md.py        # Markdown issue detector
│   └── improve_list_md.py      # List formatting detector
├── .claude/
│   ├── agents/                 # Subagent definitions
│   │   ├── ml-notebook-author.md
│   │   └── create-pycode.md
│   └── commands/               # Project slash commands (md-fixer, improve-pycode, improve-mdv2)
├── mkdocs.yml                  # MkDocs site configuration
├── sync-docs.sh                # Syncs root *.md to docs/
└── README.md                   # Project overview and navigation
```

## Tutorial Generation

- Use skill `/improve-tutorial-md <filename>` — writes and formats the tutorial end-to-end

## Jupyter Notebook (.ipynb) Generation

- Use agent `ml-notebook-author`

## Markdown Document Improvement

- Use skill `/improve-tutorial-md <filename>` — unified workflow combining automated fixes, list formatting, and ML tutorial writing standards

## Syncing Docs

Root-level `*.md` files are the source of truth. Run `sync-docs.sh` to sync them to `docs/` before building the MkDocs site.
