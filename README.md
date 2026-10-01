# 🌱 Smart Crop Recommendation System

A machine learning web app that recommends the most suitable crop to grow from **soil nutrients** and **climate conditions**. You enter 7 values (N, P, K, temperature, humidity, pH and rainfall). The best model from our comparison (**Random Forest, 99.55% test accuracy**) then gives the best crop, a confidence score, the top 3 alternatives and the probability for each of **22 crops**.

> **Mini Project, Department of AIML, Walchand College of Engineering, Sangli (AY 2026-27)**
>
> *"How can we use soil and environmental parameters to recommend a suitable crop using a data-driven machine learning approach?"*

---

## ✨ Features

- **Interactive web interface** built with Streamlit
- **Recommended crop** with a confidence percentage
- **Top 3 crop recommendations** ranked by probability
- **Probability chart** for all supported crops
- **Input summary** table with units
- **Best-model deployment**: the app serves the top classifier from the notebook's 4-model comparison
- **Consistent preprocessing**: inputs go through the same IQR capping and scaling as the training data

---

## 🧠 Model

Four classifiers were trained on the same 80/20 stratified split (1,760 train / 440 test, `random_state=42`) and compared:

| Model | Test accuracy | Errors / 440 | 5-fold CV accuracy |
|---|---|---|---|
| ⭐ **Random Forest** (100 trees) | **99.55%** | **2** | **99.59% ± 0.33%** |
| Support Vector Machine (RBF) | 99.32% | 3 | 98.59% ± 0.39% |
| K-Nearest Neighbours (k = 5) | 98.18% | 8 | 97.77% ± 0.65% |
| Decision Tree (Gini) | 97.95% | 9 | 98.77% ± 0.68% |

The notebook picks the model with the highest test accuracy and saves it as `best_model.pkl`. **Random Forest** won, and that is the model the app uses. On the test set, 18 of 22 crops have perfect precision and recall. The only two errors are *Blackgram → Maize* and *Rice → Jute*.

**Prediction pipeline (same in the notebook and the app)**

```
User input (7 values) → IQR capping (cap_bounds.pkl) → Standard scaling (scaler.pkl)
→ Random Forest (best_model.pkl) → Crop name (label_encoder.pkl)
```

**Input features**

| Feature | Description | Unit |
|---|---|---|
| `N` | Nitrogen content in soil | kg/ha |
| `P` | Phosphorus content in soil | kg/ha |
| `K` | Potassium content in soil | kg/ha |
| `temperature` | Average temperature | °C |
| `humidity` | Relative humidity | % |
| `ph` | Soil pH | 0 – 14 |
| `rainfall` | Rainfall | mm |

**Supported crops (22)**

Apple · Banana · Blackgram · Chickpea · Coconut · Coffee · Cotton · Grapes · Jute · Kidney beans · Lentil · Maize · Mango · Moth beans · Mung bean · Muskmelon · Orange · Papaya · Pigeon peas · Pomegranate · Rice · Watermelon

---

## 📓 Notebook: Analysis & Model Comparison

[`Smart_Crop_Recommendation.ipynb`](Smart_Crop_Recommendation.ipynb) contains the full ML workflow in three sections:

1. **Data foundation:** dataset overview, integrity checks (missing values, duplicates, class balance), EDA, outlier treatment by IQR capping, correlation analysis, label encoding and standard scaling.
2. **Model & prototype:** 80/20 stratified split, then training and comparing **Decision Tree, Random Forest, KNN and SVM**, selecting the best model, saving it for the app, and the prototype prediction logic.
3. **Evaluation & testing:** accuracy, macro precision/recall/F1, 5-fold stratified cross-validation, confusion matrix, error analysis of confused crop pairs and prototype test scenarios.

**Dataset:** [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). It has 2,200 records and 7 features. The notebook downloads the dataset automatically with `kagglehub`, or uses a local `Crop_recommendation.csv` if one is present.

---

## 📁 Project Structure

```
Crop-recommendation-System/
├── app.py                            # Streamlit web application
├── Smart_Crop_Recommendation.ipynb   # EDA, preprocessing, model comparison & evaluation
├── best_model.pkl                    # Best model from the notebook (Random Forest)
├── scaler.pkl                        # StandardScaler fitted on the training features
├── label_encoder.pkl                 # Label encoder (class index ↔ crop name)
├── cap_bounds.pkl                    # IQR capping bounds for each feature
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or newer
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sam-15295/Crop-recommendation-System.git
cd Crop-recommendation-System

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> The model files were saved with **scikit-learn 1.6.1**, and `requirements.txt` pins that version. Using a different version may cause loading warnings or errors.

### Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Run the notebook

Open `Smart_Crop_Recommendation.ipynb` in Jupyter or Google Colab and run all cells from top to bottom. Running it again rebuilds `best_model.pkl`, `scaler.pkl`, `label_encoder.pkl` and `cap_bounds.pkl`, so the app always uses whichever model performs best. The notebook also needs `matplotlib`, `seaborn` and `kagglehub`:

```bash
pip install matplotlib seaborn kagglehub jupyter
```

---

## 🖥️ How to Use

1. Enter the soil nutrient values: **Nitrogen, Phosphorus and Potassium**.
2. Enter the environmental conditions: **Temperature, Humidity, Soil pH and Rainfall**.
3. Click **🔍 Recommend Crop**.
4. Read the result: the recommended crop, the confidence, the top 3 alternatives and the probability chart.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** for the web interface
- **scikit-learn** for the ML models and preprocessing
- **pandas / NumPy** for data handling
- **joblib** for model serialization
- **Matplotlib / Seaborn** for visualization (notebook)

---

## 👤 Author

**Sameer Naikwadi**
GitHub: [@sam-15295](https://github.com/sam-15295)
