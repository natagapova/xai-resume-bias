# XAI for Resume Classification using Integrated Gradients

This project is part of the XAI course at Innopolis University.

We implemented the **Integrated Gradients (IG)** method to explain predictions made by the `resume-classification-gemma-2b-v1` model — a lightweight LLM used for automatic resume classification.

---

## Application Domain

**Domain**: HR automation / Resume classification  
**Dataset**: [ResumeAtlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas)  
**Model**: [`resume-classification-gemma-2b-v1`](https://huggingface.co/ahmedheakl/resume-classification-gemma-2b-v1)  
**Goal**: Check whether the model’s predictions depend on sensitive features like gender, name, or location.

---

## XAI Method: Integrated Gradients

We use **Integrated Gradients**, an attribution method that helps explain which input tokens most influenced the model's output. This is especially important for understanding fairness in LLM-based decision-making.

### Highlights:
- Implemented IG with Captum
- Visualized attributions for individual resume samples
- Identified potential signs of bias

---

## Project Contents

- `blogpost.md` — Full written explanation of the project and results  
- `notebooks/integrated_gradients.ipynb` — Main Colab notebook with implementation  
- `img/example1.png` — Visualization of token importance  
- (optional) `src/ig_explainer.py` — Separated reusable logic

---

## Example Result

Below is an example visualization showing which tokens were most important for the model's prediction:

![Example Token Attribution](./img/example1.png)

---

## Authors

- Natalia Agapova  
- Andrew Levada  

---

## Submission Links

- Blogpost: see `blogpost.md`  
- Pull Request: [PR to course repo](https://github.com/IU-PR/xai/pulls)
