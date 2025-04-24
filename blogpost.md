# Explaining Resume Classification with Integrated Gradients

**Authors**: Natalia Agapova, Andrew Levada  
**Date**: April 25, 2025

---

## Introduction

Large Language Models (LLMs) are widely used in HR automation, including resume classification. However, these models may exhibit bias by unintentionally relying on sensitive attributes such as name, gender, or location.

In this project, we apply the **Integrated Gradients (IG)** method to a resume classification model in order to explain its predictions and assess potential unfairness in decision making.

---

## Application Domain

**Domain**: Human Resources automation — resume classification  
**Dataset**: [ResumeAtlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) (13,000+ resumes)  
**Importance**: Decisions based on biased algorithms can negatively impact individuals' employment opportunities

---

## Model Description

We use the publicly available model [`resume-classification-gemma-2b-v1`](https://huggingface.co/ahmedheakl/resume-classification-gemma-2b-v1):

- **Architecture**: Decoder-only lightweight LLM  
- **Type**: Text-to-text model  

Although the model performs well, its predictions are not interpretable. Therefore, we apply explainability techniques to understand token-level decisions.

---

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

![Token attribution barplot](./img/example1.png)

---

## Bias Evaluation

We tested the model on resumes with different gendered names and locations. Integrated Gradients allowed us to identify whether these sensitive tokens had high attribution values. In some cases, such tokens showed non-negligible importance, indicating a possible bias.

---

## Conclusion

This project demonstrates how Integrated Gradients can be used to interpret LLM-based resume classification. Visualizations revealed how technical terms and job skills contributed to decisions — and occasionally, sensitive attributes.

For HR applications, explainability tools like IG are critical to ensure fairness and transparency.

---

## Links

- **Code repository**: `<your GitHub repo link here>`  
- **Dataset**: [ResumeAtlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas)  
- **Model**: [Gemma 2B Resume Classifier](https://huggingface.co/ahmedheakl/resume-classification-gemma-2b-v1)  
- **Method paper**: [Integrated Gradients (Sundararajan et al., 2017)](https://arxiv.org/abs/1703.01365)
