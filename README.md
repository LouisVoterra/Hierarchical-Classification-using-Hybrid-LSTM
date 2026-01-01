# 🧠 Hierarchical Text Classification using Hybrid CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange) ![Flask](https://img.shields.io/badge/Framework-Flask-green)

A Deep Learning project that implements a **Hybrid CNN-LSTM architecture** to perform Hierarchical Text Classification on the DBpedia dataset. The model predicts three levels of categorization (Level 1, Level 2, and Level 3) simultaneously from a single text abstract.

## 🚀 Key Features
* **Hybrid Architecture:** Combines **CNN** (for local feature/keyword extraction) and **LSTM** (for semantic sequence modeling).
* **Triple Classification:** Capable of predicting 3 hierarchical levels at once (Multi-output Model).
* **Custom Embeddings:** Trained from scratch using **Word2Vec (Gensim)** specific to the dataset.
* **Web GUI:** User-friendly interface built with **Flask** and **Bootstrap 5**.

## 📊 Model Performance
The model was trained on the DBpedia dataset and achieved the following accuracy on the validation set:
* **Level 1 (General Category):** 98.11% 🟢
* **Level 2 (Specific Category):** 90.31% 🟢 *(Surpassed Target >90%)*
* **Level 3 (Detailed Category):** 80.26% 🟡

## 🛠️ Architecture Overview
We utilize a hybrid approach to maximize classification performance:
1.  **Input Layer:** Raw text abstract.
2.  **Embedding Layer:** Pre-trained Word2Vec weights.
3.  **Conv1D (CNN):** Extracts n-gram features and key phrases.
4.  **LSTM:** Captures long-term dependencies and sentence context.
5.  **Multi-Output Heads:** Three separate Dense layers with Softmax activation for hierarchical prediction.

## 💻 Installation & Usage
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/LouisVoterra/Hierarchical-Classification-using-Hybrid-LSTM.git](https://github.com/LouisVoterra/Hierarchical-Classification-using-Hybrid-LSTM.git)
    cd Hierarchical-Classification-using-Hybrid-LSTM
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    Since the trained model files (`.h5`) are large and not included in this repo, you need to train the model first or download the weights (if provided).
    ```bash
    # Step 1: Train Model (Optional if model exists)
    python training_model.py
    
    # Step 2: Run Web App
    python app.py
    ```

4.  **Access the UI:**
    Open your browser at `http://127.0.0.1:5000`

## 📸 Screenshots
<img width="1920" height="846" alt="image" src="https://github.com/user-attachments/assets/4f912d27-330f-40df-88e7-70019755aff9" />


## 👥 Authors
* **160422060 - Muhammad Hanif**
* **160422075 - Gilbert Maynard Saragih**
* **160422077 - Louis Dewa Voterra** 
