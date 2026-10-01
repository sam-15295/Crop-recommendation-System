# 🌱 Smart Crop Recommendation System

A machine learning web app that recommends the most suitable crop to grow from **soil nutrients** and **climate conditions**. You enter 7 values (N, P, K, temperature, humidity, pH and rainfall). A soft-voting ensemble model then gives the best crop, a confidence score, the top 3 alternatives and the probability for every crop.

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
- **Ensemble model** that combines four strong classifiers for stable predictions

---

## 🧠 Model

The app uses a **soft-voting `VotingClassifier`** that averages the predicted probabilities of four models:

| Estimator | Configuration |
|---|---|
| Random Forest | 300 trees |
| Extra Trees | 300 trees |
| Gradient Boosting | default parameters |
| Support Vector Machine | RBF kernel inside a `StandardScaler` pipeline |

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

**Crops supported by the deployed model (12)**

Apple · Banana · Coconut · Coffee · Cotton · Grapes · Maize · Mango · Orange · Papaya · Rice · Watermelon

---

## 📓 Notebook: Analysis & Model Comparison

[`Smart_Crop_Recommendation.ipynb`](Smart_Crop_Recommendation.ipynb) contains the full ML workflow in three sections:

1. **Data foundation:** dataset overview, integrity checks (missing values, duplicates, class balance), EDA, outlier treatment by IQR capping, correlation analysis, label encoding and standard scaling.
2. **Model & prototype:** 80/20 stratified split, then training and comparing **Decision Tree, Random Forest, KNN and SVM**, selecting the best model and the prototype prediction logic.
3. **Evaluation & testing:** accuracy, macro precision/recall/F1, 5-fold stratified cross-validation, confusion matrix, error analysis of confused crop pairs and prototype test scenarios.

**Dataset:** [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). It has 2,200 records and 7 features. The notebook downloads the dataset automatically with `kagglehub`, or uses a local `Crop_recommendation.csv` if one is present.

---

## 📁 Project Structure

```
Crop-recommendation-System/
├── app.py                                   # Streamlit web application
├── crop_recommendation_ensemble_model.pkl   # Trained soft-voting ensemble model
├── crop_recommendation_label_encoder.pkl    # Label encoder (class index ↔ crop name)
├── Smart_Crop_Recommendation.ipynb          # EDA, preprocessing, model comparison & evaluation
├── requirements.txt                         # Python dependencies
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

Open `Smart_Crop_Recommendation.ipynb` in Jupyter or Google Colab and run all cells from top to bottom. The notebook also needs `matplotlib`, `seaborn` and `kagglehub`:

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
