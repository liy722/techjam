# techjam
# Filtering the Noise: An LLM-Powered System for Trustworthy Location Reviews

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Project Overview

This project is a submission for the **Filtering the Noise: ML for Trustworthy Location Reviews** hackathon. It is a robust, end-to-end system that leverages a Large Language Model (LLM) to automatically assess the quality and relevancy of Google location reviews. The system classifies reviews against a defined set of policies to combat spam, irrelevant content, and fake rants, thereby enhancing the trustworthiness of online review platforms.

Our solution successfully processes and classifies a dataset of 1,100 reviews, demonstrating a practical and effective application of modern AI techniques to solve a real-world content moderation problem.

## 🎯 The Problem

Online reviews are crucial for both consumers and businesses, but their value is often diluted by low-quality content. The challenge was to design an ML-based system to automatically filter reviews that violate key policies, such as:
1.  **No Advertisements:** Reviews should not contain promotional content.
2.  **No Irrelevant Content:** Reviews must be about the location being reviewed.
3.  **No Rants Without a Visit:** Complaints should come from actual visitors, a fact that must be inferred from the content and available metadata.

## 💡 Our Solution

We developed a Python-based pipeline in Google Colab that uses the powerful `meta-llama/Meta-Llama-3-8B-Instruct` model, accessed via the Hugging Face `InferenceClient`.

The core of our solution is a sophisticated **Prompt Engineering** strategy. For each review, we construct a detailed prompt that provides the LLM with:
* A clear role ("You are a Google Maps review moderator").
* The review text itself.
* Extracted metadata, such as the user's star rating and whether a photo was included (`Has Photo Provided`).
* A strict set of policy instructions and the required JSON output format.

This method allows the model to make nuanced, context-aware judgments that combine both textual and non-textual signals.

### Tech Stack
* **Language:** Python
* **Core Libraries:** `pandas`, `huggingface-hub`, `scikit-learn`
* **Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
* **Environment:** Google Colab (T4 GPU)
* **Data Storage:** Google Drive

## 🚀 How to Run

To reproduce our results, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-GitHub-Username]/[Your-Repo-Name].git
    ```

2.  **Prepare the Data:**
    * Download the dataset `hackathon_data.zip` provided for the project.
    * Upload the `hackathon_data.zip` file to the root directory of your Google Drive.

3.  **Run the Notebook:**
    * Open the `.ipynb` file from this repository in Google Colab.
    * Run the cells sequentially from top to bottom.
    * You will be prompted to authorize Google Drive access and to provide your Hugging Face API token via an interactive login for authentication.

4.  **Get the Results:**
    * The script will process all 1,100 reviews and save the final, classified data to a file named `classified_reviews.csv` in your Google Drive root directory.

## 📊 Evaluation & Results

To validate our model's performance, we followed the hackathon's recommendation for handling missing labels: we manually annotated a random subset of 50 reviews to create a ground-truth validation set.

We then calculated the precision, recall, and F1-score for each policy. The results demonstrate the effectiveness of our approach in a real-world scenario.

**Classification Report:**


These metrics show that our system can effectively identify policy violations with a quantifiable degree of accuracy.

## 🔮 Future Work

* **Integrate a Multimodal Model:** Our initial plan was to use the multimodal `google/gemma-3-12b-it` model. Due to persistent server-side `403 Forbidden` errors with the model's API endpoint, we strategically pivoted to the robust and reliable Llama 3 model. Future work would involve re-integrating a working multimodal model to analyze not just the text, but also the content of the review images themselves.
* **Build a Simple UI:** Create a simple web interface using Streamlit or Gradio where a user can paste a review and see the classification in real-time.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
