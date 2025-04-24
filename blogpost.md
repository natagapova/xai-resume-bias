# Explaining Resume Classification with Integrated Gradients

**Authors**: Natalia Agapova, Andrew Levada  
**Date**: April 24, 2025

---

## Introduction

Large Language Models (LLMs) are widely used in HR automation, including resume classification. However, these models may exhibit bias by unintentionally relying on sensitive attributes such as name, gender, or location.

In this project, we apply the **Integrated Gradients (IG)** method to a resume classification model in order to explain its predictions and assess potential unfairness in decision making.

---

## Application Domain

**Domain**: Human Resources automation — resume classification  
**Importance**: Decisions based on biased algorithms can negatively impact individuals' employment opportunities

---

## Model Description

We use the publicly available model [`bert-resume-classification`](https://huggingface.co/ahmedheakl/bert-resume-classification), which implementation and results are described in the [target paper](https://arxiv.org/html/2406.18125v1)

- **Type**: Text-to-text model

## Explainability Method: Integrated Gradients

Integrated Gradients (IG) is an attribution method that measures feature importance by integrating gradients between a baseline input and the actual input.

### Steps:

1. Choose a baseline input (e.g., empty string or [PAD] tokens)
2. Encode both baseline and actual input
3. Compute gradients across interpolated inputs
4. Sum gradients for each input token
5. Visualize attribution scores per token

---

## Token Attribution Example

Given the input:

> “Senior Java developer with 10+ years experience in backend systems, cloud, and microservices.”

We computed token attributions using IG. The visualization (see below) shows which tokens were most influential in the model’s classification decision.

![Token attribution barplot](./notebooks/example1.png)

---

## Bias Evaluation

We tested the model on resumes with different gendered names and locations. Integrated Gradients allowed us to identify whether these sensitive tokens had high attribution values. In some cases, such tokens showed non-negligible importance, indicating a possible bias.

### Examples of Bias Evaluation

We applied Integrated Gradients to samples from the original dataset to visualize token attributions and check for potential bias related to sensitive information.

**Sample 1:**

![Token attribution barplot for Sample 1](./notebooks/example1.png)

**Sample 2:**

![Token attribution barplot for Sample 2](./notebooks/example2.png)

**Sample 3:**

![Token attribution barplot for Sample 3](./notebooks/example3.png)

The results are puzzling, but do not indicate that the model is not biased

---

This project demonstrates how Integrated Gradients can be used to interpret LLM-based resume classification. Visualizations revealed how technical terms and job skills contributed to decisions — and occasionally, sensitive attributes.

For HR applications, explainability tools like IG are critical to ensure fairness and transparency.

---

## Links

- **Code repository**: `link`[https://github.com/natagapova/xai-resume-bias]
- **Method paper**: [Integrated Gradients (Sundararajan et al., 2017)](https://arxiv.org/abs/1703.01365)
