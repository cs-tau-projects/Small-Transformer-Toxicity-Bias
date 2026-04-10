# Proposal for Project Improvements: Toxicity Bias in Small Transformers

Based on our discussion regarding the grading criteria (especially "Ambitiousness" and "Quality of research question") and the existing literature, here are three concrete ways to elevate the project proposition. These options transform the project from a simple reproduction of older research into a novel, comparative study.

## Option 1: The Architectural Evolution Comparison (Highly Recommended)

**The Concept:** Instead of just measuring bias in one small transformer (like BERT), compare its bias profile against a modern, instruction-tuned LLM (like Llama-3-8B or a smaller Pythia model).

**Why it works:**
*   **Novelty:** It Directly addresses the gap between the 2018 era (CNNs/RNNs) and the 2024+ era (LLMs with safety guardrails).
*   **The "Hook":** Modern LLMs often score higher on overall toxicity detection but fail dramatically on subgroup bias (often over-flagging minority groups due to aggressive safety tuning). Proving this empirically against a "dumber" BERT model makes for a fantastic research paper.
*   **Workload:** Low to Medium. You already have the dataset pipeline. You just need to add a zero-shot inference script for the LLM on a subset of the test data.

**Suggested Quote to add to the Proposal:**
> *"While early work (Dixon et al., 2018) identified bias in pre-transformer architectures, and recent studies highlight safety-refusal failures in massive instruction-tuned LLMs, this project compares the bias profiles of these two paradigms. We will investigate whether fine-tuned small encoder models (like BERT) inherently mitigate, exacerbate, or exhibit different subgroup biases compared to the zero-shot performance of modern, safety-aligned LLMs."*

---

## Option 2: The Mitigation Trial

**The Concept:** Measure the bias in a standard BERT model, and then actively attempt to fix or reduce it using a known mitigation strategy from the literature.

**Why it works:**
*   **Action-Oriented:** It shows you aren't just observing a problem, you are engineering a solution.
*   **Directly answers the assignment:** The syllabus suggests "improving a model for a specific task" as a good project idea.
*   **Workload:** Medium. You have to train two models instead of one. The second model would use a technique like *Data Balancing* (ensuring equal representation of toxic/non-toxic examples across identity groups during training) or *Custom Loss Weighting* (penalizing errors on minority groups more heavily).

**Suggested Quote to add to the Proposal:**
> *"In addition to quantifying the baseline Subgroup AUC and False Negative Rates in fine-tuned BERT models, this project will attempt to reduce these bias gaps through targeted data balancing techniques. We will compare the performance of the baseline model against a mitigation-aware training run to evaluate the efficacy of dataset interventions on small transformer architectures."*

---

## Option 3: The Cross-Model Family Comparison

**The Concept:** Compare the bias profiles *between* different families of small transformers (e.g., BERT vs. RoBERTa vs. DistilBERT vs. ALBERT).

**Why it works:**
*   **Deep Dive:** It investigates whether the pre-training objective itself (e.g., BERT's Next Sentence Prediction vs. RoBERTa's dynamic masking) affects how the model later inherits societal biases.
*   **Easiest to implement:** Since all these models use the Hugging Face `AutoModelForSequenceClassification` API, you barely have to change any code. You just run the same training script 3 or 4 times with different `model_name` arguments.
*   **Workload:** Low (but requires more cluster compute time).

**Suggested Quote to add to the Proposal:**
> *"This project systematically evaluates how the foundational pre-training objectives of different small transformer families (e.g., BERT, RoBERTa, and ALBERT) influence their susceptibility to unintended subgroup bias during downstream fine-tuning. We hypothesize that variations in pre-training data and masking strategies will manifest as distinct bias profiles in toxicity classification."*

---

## Recommendation

**Option 1** provides the strongest narrative for a high grade, as it touches on current hot topics (LLM safety vs. traditional models). If you are worried about the technical complexity of setting up an LLM baseline, **Option 3** is the safest and easiest way to guarantee you have a solid "Comparative Baseline" for your methodology section.
