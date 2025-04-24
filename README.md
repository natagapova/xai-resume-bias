# XAI for Resume Classification using Integrated Gradients

This project is part of the XAI course at Innopolis University.

We implemented the **Integrated Gradients (IG)** method to explain predictions made by the `bert-resume-classification` model — a lightweight LLM used for automatic resume classification.

---

## Application Domain

**Domain**: HR automation / Resume classification  
**Dataset**: [ResumeAtlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas)  
**Model**: [`bert-resume-classification`](https://huggingface.co/ahmedheakl/bert-resume-classification)
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

---

## Example Result

The visualization shows which tokens were most important for the model's prediction:

![Example Token Attribution](./notebooks/example1.png)

---

## Authors

- Natalia Agapova
- Andrew Levada

---

## Submission Links

- Blogpost: see `blogpost.md`
- Pull Request: [PR to course repo](https://github.com/IU-PR/xai/pulls)
