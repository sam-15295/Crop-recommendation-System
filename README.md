# 🌱 Smart Crop Recommendation System

A machine learning web app that recommends the most suitable crop to grow from **soil nutrients** and **climate conditions**. You enter 7 values (N, P, K, temperature, humidity, pH and rainfall). The best model from our comparison (**Random Forest, 99.55% test accuracy**) then gives the best crop, a confidence score, the top 3 alternatives and the probability for each of **22 crops**.

> **Mini Project, Department of AIML, Walchand College of Engineering, Sangli (AY 2026-27)**
>
> *"How can we use soil and environmental parameters to recommend a suitable crop using a data-driven machine learning approach?"*

---

## ✨ Features

- **Clean, responsive interface** built with Streamlit, with light and dark mode and mobile support
- **One-click examples**: load real soil profiles from the test set (rice, cotton, maize, chickpea, apple, coffee)
- **Recommended crop** with a confidence percentage
- **Top 3 crops** ranked by probability
- **Interactive probability chart** and an input summary table
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
├── requirements.txt                  # Python dependencies for the app
├── requirements-notebook.txt         # Extra packages for running the notebook
├── render.yaml                       # Render deployment blueprint
├── .streamlit/config.toml            # App theme (light & dark)
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

Open `Smart_Crop_Recommendation.ipynb` in Jupyter or Google Colab and run all cells from top to bottom. Install the notebook packages first. This includes the same pinned scikit-learn version that the app uses:

```bash
pip install -r requirements-notebook.txt
```

### Retraining with a new or updated dataset

1. Put your CSV in the project folder as **`Crop_recommendation.csv`**. It needs the same columns: `N, P, K, temperature, humidity, ph, rainfall, label`. If this file is present, the notebook uses it instead of downloading from Kaggle.
2. Run all cells of the notebook. It compares the four models again and saves the best one (`best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `cap_bounds.pkl`). New or removed crops are picked up automatically.
3. Commit and push the dataset, the notebook and the four `.pkl` files:
   ```bash
   git add Crop_recommendation.csv Smart_Crop_Recommendation.ipynb *.pkl
   git commit -m "Retrain model on updated dataset"
   git push
   ```
4. Render redeploys the app automatically with the new model.

> The notebook only saves the model files when scikit-learn **1.6.1** is installed, the version Render uses. With any other version it prints **NOT SAVED** and tells you how to fix it.

---

## ☁️ Deployment (Render)

The repository includes a [`render.yaml`](render.yaml) blueprint, so you can deploy it on [Render](https://render.com) as a free web service:

1. Sign in to Render with GitHub and open **New → Blueprint**.
2. Select the `Crop-recommendation-System` repository and click **Apply**.
3. Render installs `requirements.txt` on Python 3.12 and starts the app with:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```

Every push to `main` triggers a new deploy automatically. On the free plan the service sleeps after about 15 minutes without traffic, so the first request after that takes up to a minute to wake it up.

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
