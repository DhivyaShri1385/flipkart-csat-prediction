# flipkart-csat-prediction
Customer Satisfaction Prediction using EDA and ML (LogReg, RF, XGBoost)
# 📊 Flipkart Customer Satisfaction Prediction using EDA and Machine Learning

## 📌 Project Objective
To analyze and predict **customer satisfaction (CSAT)** levels from Flipkart’s support ticket data using **Exploratory Data Analysis (EDA)** and **Machine Learning** techniques.

---

## 🧠 Problem Statement
Flipkart receives thousands of customer support queries. The goal is to predict whether a customer’s satisfaction level will be **Low**, **Medium**, or **High** based on support metadata (agent, shift, channel, etc.) — enabling better service delivery, agent training, and customer retention.

---

## 📁 Dataset Description

| Column | Description |
|--------|-------------|
| `channel_name` | Support channel (Chat, Email, Call) |
| `category`, `Sub-category` | Issue category |
| `Agent_name`, `Agent Shift` | Support staff details |
| `Tenure Bucket` | Experience level of agent |
| `connected_handling_time` | Duration of support interaction |
| `CSAT Score` | Satisfaction score (1–5) |
| `Customer_City`, `Product_category` | Customer and product metadata |

---

## 🔍 Key Steps Performed

### 1. 📦 Data Preprocessing
- Missing values handled using median imputation.
- Outliers removed using IQR method.
- Categorical features encoded using `LabelEncoder`.
- Target variable transformed into CSAT_Level: **Low**, **Medium**, **High**.

### 2. 📊 Exploratory Data Analysis (EDA)
- CSAT distribution across agents, channels, shifts.
- Boxplots and bar charts used to visualize satisfaction trends.
- Correlation heatmap for numeric features.

### 3. 🧪 Machine Learning Models
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **XGBoost (Best Performing Model)**

### 4. 🔧 Model Tuning & Evaluation
- Hyperparameter tuning with `GridSearchCV` and `RandomizedSearchCV`.
- Evaluation metrics: **Precision**, **Recall**, **F1-score**, **Accuracy**
- Feature importance visualization (Random Forest & XGBoost)

---

## 📈 Results & Insights

| Model        | Accuracy | F1 (Low) | F1 (High) | Final Verdict |
|--------------|----------|----------|-----------|----------------|
| Logistic Regression | 77% | 0.02 | 0.88 | Weak baseline |
| Random Forest       | 84% | 0.14 | 0.90 | Good performance |
| **XGBoost**         | **85%** | **0.31** | **0.91** | ✅ **Selected** |

### 🔎 Top Influencing Features:
- `connected_handling_time`
- `Agent Shift`
- `Tenure Bucket`
- `channel_name`
- `Sub-category`

---

## 🛠️ Tech Stack
- Python, Jupyter Notebook
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- xgboost

---

## ▶️ How to Run

```bash
# Clone repository
git clone https://github.com/DhivyaShri1385/flipkart-csat-prediction
cd flipkart-csat-prediction

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook Flipkart_CSAT.ipynb

