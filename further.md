# Further Reading

A curated collection of foundational papers and modern developments in transformer architectures, optimization, and scaling techniques.

---

## Foundation Papers

### Core Transformer Architecture

**[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017)
- Original transformer paper introducing the attention mechanism
- Established multi-head self-attention and positional encoding
- Demonstrated state-of-the-art performance on machine translation

**[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)** (Devlin et al., 2018)
- Bidirectional encoder representations from transformers
- Masked language modeling (MLM) training objective
- Revolutionized transfer learning for NLP tasks

**[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)** (Radford et al., 2018)
- Original GPT paper introducing generative pre-training
- Demonstrated effectiveness of unsupervised pre-training followed by supervised fine-tuning
- Foundation for decoder-only transformer architectures

**[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** (Radford et al., 2019)
- GPT-2: Scaling up language models
- Zero-shot task transfer capabilities
- Demonstrated emergent capabilities with scale

**[T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)** (Raffel et al., 2019)
- Text-to-text transfer transformer
- Unified framework for NLP tasks
- Comprehensive study of transfer learning approaches

**[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)** (Brown et al., 2020)
- GPT-3: Demonstrated in-context learning at scale
- Few-shot learning without gradient updates
- Showed scaling trends in model capabilities

---

## Mathematical Foundations

**[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** (Kaplan et al., 2020)
- Optimal compute allocation strategies
- Power-law relationships between scale and performance
- Guidelines for model size, dataset size, and training compute

**[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)** (Su et al., 2021)
- RoPE position encoding
- Improved relative position representation
- Better extrapolation to longer sequences

---

## Efficiency & Scaling

**[FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022)
- Memory-efficient attention computation
- IO-aware algorithm design
- Significant speedups for training and inference

**[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)** (Shazeer, 2019)
- Multi-query attention (MQA)
- Reduced KV cache memory requirements
- Faster autoregressive generation

**[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** (Hu et al., 2021)
- Parameter-efficient fine-tuning
- Low-rank decomposition of weight updates
- Efficient adaptation without full fine-tuning

**[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)** (Touvron et al., 2023)
- Modern decoder-only architectures
- Training optimizations and architectural improvements
- Open foundation models

---

## Training & Optimization

**[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)** (Loshchilov & Hutter, 2019)
- AdamW optimizer
- Proper weight decay implementation
- Improved generalization

**[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)** (Xiong et al., 2020)
- Pre-LayerNorm vs Post-LayerNorm
- Training stability improvements
- Analysis of gradient flow

**[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)** (Press & Wolf, 2017)
- Weight tying between embedding and output layers
- Reduced parameter count
- Improved performance

---

## Alternative Architectures

**[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)** (Gu et al., 2023)
- Alternative to transformer attention
- Linear-time sequence modeling
- Selective state space models

---

*Last updated: 2025*
