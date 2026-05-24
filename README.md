# Statistical Language Modeling: N-Gram Architectures and Smoothing Techniques

**Developer:** Yonghao Lee  
**Date:** April 29, 2026  
**Tech Stack:** Python, Jupyter, spaCy, Hugging Face Datasets, LaTeX

---

## Project Overview
This project contains an end-to-end implementation and comparative analysis of statistical Language Models (LMs) trained on the Wikitext-2 corpus. It explores the mathematical properties of Unigram and Bigram Maximum Likelihood Estimation (MLE) models, diagnoses the zero-probability pathology common in sparse training sets, and resolves sequence probability collapse using a linearly interpolated smoothing architecture.

The implementation balances computational efficiency with structural correctness, making it a robust example of classical NLP engineering.

---

## Technical Architecture & Core Phases

### Phase 1: High-Performance Tokenization & Model Training
* **Pipeline Optimization**: Implemented a highly accelerated preprocessing pipeline using spaCy's `en_core_web_sm` model with heavy syntactic components (`parser` and `ner`) explicitly disabled to maximize CPU throughput.
* **Text Normalization**: Documents are processed line-by-line, lowercased, lemmatized, and stripped of non-alphabetic elements like punctuation and numbers using strict token filtering.
* **Boundary Tracking**: Prepends a custom `<START>` sentinel to establish consistent context bounds for trailing tokens.
* **State Management**: Tracks word frequencies (`unigram_counts`), bigram combinations (`bigram_counts`), and historical context totals (`context_totals`) in optimized memory structures to allow $\mathcal{O}(1)$ runtime lookup and probability calculation.

### Phase 2: Autoregressive Sequence Continuation
* Developed a predictive inference module to calculate conditional distributions over arbitrary text prompts.
* Evaluates context matrices using a greedy argmax search to extract the absolute highest conditional probability for sequential next-word prediction.

### Phase 3: Evaluation Metrics (Log-Probability & Joint Perplexity)
* Implemented mathematical scoring functions to calculate total sequence probabilities in log-space, preventing floating-point underflow errors.
* Evaluates cross-sentence sequence uncertainty by calculating the joint perplexity over hidden validation sequences.

### Phase 4: Linear Interpolation Smoothing
* Created a robust hybrid language model utilizing a convex combination of dense unigram and sparse bigram token distributions.
* Applies static weights ($\lambda_{bi} = 2/3$, $\lambda_{uni} = 1/3$) to establish a valid probability distribution that guarantees finite perplexity scores over unseen data pairs.

---

## Project Metrics & Empirical Results

### Next-Word Prediction Performance
* **Input Context String:** `[<START>, i, have, a, house, in]`
* **Predicted Target Continuation:** `"the"`
* **Empirical Occurrences:** $12,754$ joint bigram instances in training data
* **Calculated MLE Probability:** $\hat{P}(\text{the} \mid \text{in}) \approx 0.2859$

### Performance & Perplexity Comparison Metrics
Evaluated over two benchmark text validation strings ($M = 12$ total target tokens):
* *String 1 (Sparse Transition):* `"Brad Pitt was born in Oklahoma"`
* *String 2 (Attested Transition):* `"The actor was born in USA"`

| Evaluation Metric | Pure Bigram Model (MLE) | Linearly Interpolated Model |
| :--- | :---: | :---: |
| **String 1 Log-Probability** | $-\infty$ | $-36.2215$ |
| **String 2 Log-Probability** | $-29.7299$ | $-31.0107$ |
| **String 1 Probability $P(s)$** | $0$ | $1.859 \times 10^{-16}$ |
| **String 2 Probability $P(s)$** | $1.226 \times 10^{-13}$ | $3.406 \times 10^{-14}$ |
| **Joint Perplexity ($M=12$)** | $+\infty$ | **271.15** |

### Engineering Insights
* **The Zero-Probability Pathology**: Under the baseline Bigram model, String 1 fails completely with a probability of $0$. This happens because the exact sequence `(<START>, brad)` never occurred in the training split, forcing the entire multiplicative chain to collapse.
* **The Smoothing Solution**: The Linear Interpolation model rescues the sequence calculation by providing a strict positive lower bound derived from global unigram occurrences. This maintains the structural and contextual strength of bigram tracking while preventing mathematical out-of-bounds errors on sparse data.

---

## Setup & Execution Guide

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/yourusername/statistical-language-modeling.git](https://github.com/yourusername/statistical-language-modeling.git)
   cd statistical-language-modeling
