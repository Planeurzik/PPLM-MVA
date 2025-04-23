# Plug and Play Language Models: A Simple Approach to Controlled Text Generation

## Overview

This project explores the concept of Plug and Play Language Models (PnP-LMs) for controlled text generation. The approach leverages pre-trained transformer models and attribute classifiers to steer the generation process towards desired attributes or topics.

## Key Concepts

### Plug and Play Methods

The core idea is to use a generative model $p(x)$ and a classifier $p(a \mid x)$ to maximize the conditional probability $p(x \mid a)$. This is achieved by:

- Using a pre-trained transformer as the generative model.
- Training a classifier to guide the generation towards specific attributes.

The formula used is:
```math
p(x \mid a) \propto p(x) p(a \mid x)
```

### Steering Generation

The generation process involves:

1. **Forward Pass**: Generate text using the original distribution.
2. **Backward Pass**: Update latent variables based on the attribute model.
3. **Recompute**: Generate text with updated latents to align with desired attributes.

## Implementation

### Transformer Model

A transformer model was built from scratch using:

- Hugging Face's FineWeb-edu framework.
- Custom tokenizer with Byte Pair Encoding (BPE).
- Pre-training for one day with Key-Value (KV) caching implemented.

### Controlled Generation

#### Bag of Words (BoW)

- Controlled generation towards a specific bag of words using the formula:
```math
  \log(p(a \mid x)) = \log \left(\sum_{w_i \in \text{BoW}} p_{t_{i+1}}(w_i)\right)
```
- Ascending $\log(p(a \mid x))$ during inference to guide generation.

#### Neural Network Discriminator

- To address the drawbacks of BoW, a neural network discriminator was used for $p(a \mid x)$, specifically with GPT-2.
- Loss functions included Cross-Entropy Loss and KL-Divergence to ensure alignment with target attributes.

## Results

- Successfully created a small Language Model (LLM) from scratch.
- Implemented controlled generation using the custom LLM and GPT-2.
- Demonstrated the ability to steer text generation towards specific topics or attributes.

## Conclusion

This project showcases a practical approach to controlled text generation using Plug and Play Language Models, combining generative models with attribute classifiers to achieve desired text characteristics.