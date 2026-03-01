# Harmful Text Detection and Group Bias in Small Transformer Models

## Participants
* **Maxim German** (322542887) - maximgerman1@mail.tau.ac.il
* **Eran Shufaro** (209074731) - shufaru@mail.tau.ac.il
* **Ilay Abramovich** (322271032) - ilaya@mail.tau.ac.il
* **Itay Hazan** (209277367) - itayhazan@mail.tau.ac.il

## Project Description
This project investigates the effectiveness of small pre-trained transformer encoder models (BERT family) when fine-tuned for harmful and toxic text detection. We specifically focus on identifying unintended bias across identity groups—such as gender, religion, and sexual orientation—within these models.

While Large Language Models (LLMs) are currently prominent, many deployed moderation systems still rely on smaller transformer-based classifiers due to their stability and ease of deployment. Prior research has shown that these classifiers can exhibit biased behavior toward specific groups even when the text itself is non-toxic.

## Methodology
* **Models**: Fine-tuning small encoder-based transformer models (BERT variants).
* **Dataset**: Google Jigsaw Unintended Bias in Toxicity Classification dataset.
* **Evaluation**: Performance is measured using ROC-AUC, subgroup AUC, and False Negative Rates (FNR) to identify potential bias gaps.

## Requirements
* **Environment**: Hugging Face Transformers and Datasets ecosystem.
* **Resources**: University-provided computational resources.
* **Access**: All data, weights, and APIs used are publicly accessible.

## AI Usage Disclosure
A generative AI tool (ChatGPT) was used to assist with brainstorming and refining the academic phrasing and structure of the project proposal to ensure conciseness and clarity.

## References
1. Dixon et al. "Measuring and Mitigating Unintended Bias in Text Classification." (2018).
2. Xie et al. "SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal Behaviors." (2025).