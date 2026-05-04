# Attention Mechanism Modification in BERT

## Overview
This project explores modifications to the attention mechanism in BERT, aiming to improve model performance on specific NLP tasks such as hate speech detection.

## Motivation
Standard attention mechanisms treat all tokens equally in computation. However, in tasks involving semantic bias (e.g., hate speech), certain tokens may carry more importance.

## Approach
- Modified the self-attention layer in BERT
- Introduced bias toward specific token types
- Evaluated impact on classification performance

## Key Findings
- Directly increasing attention weights on specific tokens led to overfitting
- Highlighted the limitation of naive attention biasing
- Motivated alternative approaches (e.g., topic-based modeling)

## Tech Stack
- Python
- PyTorch / Transformers (HuggingFace)

## Notes
This repository focuses on experimentation with model internals rather than production-ready implementation.
