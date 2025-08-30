# Filtering the Noise: An LLM-Powered System for Trustworthy Location Reviews

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Project Overview

This project is a submission for the **Filtering the Noise: ML for Trustworthy Location Reviews** hackathon. It is a robust, end-to-end system that leverages a Large Language Model (LLM) to automatically assess the quality and relevancy of Google location reviews. The system classifies reviews against a defined set of policies to combat spam, irrelevant content, and fake rants, thereby enhancing the trustworthiness of online review platforms.

After encountering and overcoming the limitations of remote APIs, we successfully implemented a **local inference pipeline**, running the powerful, multimodal **`google/gemma-3-12b-it`** model directly within Google Colab. This allowed us to process a full dataset of 1,100 reviews, demonstrating a resilient and sophisticated approach to a real-world content moderation problem.

## 🎯 The Problem

Online reviews are crucial for both consumers and businesses, but their value is often diluted by low-quality content. The challenge was to design an ML-based system to automatically filter reviews that violate key policies, such as:
1.  **No Advertisements:** Reviews should not contain promotional content.
2.  **No Irrelevant Content:** Reviews must be about the location being reviewed.
3.  **No Rants Without a Visit:** Complaints should come from actual visitors, a fact that must be inferred from the content and available metadata.

## 💡 Our Solution

We developed a Python-based pipeline in Google Colab that runs the `google/gemma-3-12b-it` model locally. To make this possible in a resource-constrained environment, we used the **`unsloth`** library for significant performance optimization, loading the model with 4-bit quantization.

The core of our solution is a sophisticated **multimodal prompt engineering** strategy. For each review, we construct a detailed prompt that provides the LLM with:
* A clear role ("You are a Google Maps review moderator").
* The review text itself.
* The review image (if provided).
* Extracted metadata, such as the user's star rating.
* A strict set of policy instructions and the required JSON output format.

This method allows the model to make nuanced, context-aware judgments that combine textual, visual (when available), and non-textual signals.

### Tech Stack
* **Language:** Python
* **Core Libraries:** `pandas`, `unsloth`, `PyTorch`, `Hugging Face Transformers`, `scikit-learn`
* **Model:** `google/gemma-3-12b-it`
* **Environment:** Google Colab (T4 GPU)
* **Data Storage:** Google Drive

## 🚀 How to Run

To reproduce our results, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/liy722/techjam.git
    ```

2.  **Prepare the Data:**
    * Download the dataset `hackathon_data.zip` provided for the project.
    * Upload the `hackathon_data.zip` file to the root directory of your Google Drive.

3.  **Configure the Environment:**
    * Open the `.ipynb` file from this repository in Google Colab.
    * Ensure the runtime type is set to a **GPU** (e.g., T4 GPU) via `Runtime` -> `Change runtime type`.

4.  **Run the Notebook:**
    * Run the cells sequentially from top to bottom.
    * The first cells will install necessary libraries (like `unsloth`), connect to Google Drive, and download the Gemma 3 model (approx. 25GB), which may take a significant amount of time.

5.  **Get the Results:**
    * The script will process all 1,100 reviews and save the final, classified data to a file named `gemma3_classified_reviews.csv` in your Google Drive root directory.

## 📊 Evaluation & Results
<img width="1148" height="1273" alt="image" src="https://github.com/user-attachments/assets/206912d4-8087-4bee-98f9-5070ae56b430" />
### Analysis of Results: Uncovering the "Accuracy Paradox"

At first glance, the overall `accuracy` of the model appears very high (over 97%). However, a closer look at the **precision, recall, and F1-score** for the actual violation classes (`is_advertisement`, etc.) reveals that they are all 0.00.

This is a classic and highly insightful machine learning problem known as **Class Imbalance**. Our dataset is naturally imbalanced: the vast majority of reviews are normal, and only a tiny fraction are violations. The model discovered that it could achieve over 97% accuracy by taking a "lazy" shortcut: **simply predicting `False` (not a violation) for every single review.**

This discovery is the most critical outcome of our project. It demonstrates that a simple accuracy metric can be deeply misleading and highlights the necessity of using precision and recall for a true understanding of a model's performance on real-world, imbalanced data. Our end-to-end pipeline, from implementation to evaluation, successfully uncovered this crucial challenge.


To validate our model's performance, we followed the hackathon's recommendation for handling missing labels: we manually annotated a random subset of 50 reviews to create a ground-truth validation set.

We then calculated the precision, recall, and F1-score for each policy. The results demonstrate the effectiveness of our approach in a real-world scenario.

**Classification Report:**
## 🔮 Future Work

Our evaluation successfully identified class imbalance as the key challenge. Future work will directly address this to build a truly effective moderation tool:

1.  **Address Class Imbalance:**
    * **Data Augmentation:** Collect more examples of violation reviews or use text generation techniques to create synthetic examples of the minority classes (advertisements, irrelevant content, etc.).
    * **Algorithmic Approaches:** Implement techniques like oversampling the minority class (e.g., using SMOTE) or undersampling the majority class.

2.  **Model Fine-tuning:**
    * Fine-tune the Gemma 3 model on our balanced dataset.
    * Utilize a **weighted loss function** during fine-tuning to force the model to pay significantly more attention to correctly classifying the rare but important violation cases.

3.  **Deployment and UI:**
    * Once the model performance is improved, the plan to deploy it as a service and build a simple Streamlit/Gradio UI remains a key next step for creating an interactive demo.
