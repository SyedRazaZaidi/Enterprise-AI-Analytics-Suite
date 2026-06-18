# 🚀 Enterprise Intelligence Suite

An end-to-end, full-stack Business Intelligence and Machine Learning platform designed to process real-time data streams, forecast temporal trends, and provide unsupervised behavioral segmentation. 

Built by high-level AI engineering students in the BS-AI program at the University of Central Punjab (UCP), developed in collaboration with Sher Ali Saleem.

## 🧠 Core Architecture
* **Backend Framework:** Python, FastAPI, Uvicorn
* **Data Engineering:** PySpark, Apache Kafka (Live Stream Processing)
* **Machine Learning:** Scikit-Learn (Random Forest, K-Means Clustering)
* **Time-Series Forecasting:** Statsmodels (Holt-Winters Exponential Smoothing)
* **Frontend UI:** HTML5, CSS3, JavaScript, Chart.js
* **Automated Reporting:** Matplotlib, FPDF (Algorithmic PDF Generation)

## ⚡ Key Features

### 1. Real-Time Operations
Ingests live data streams for up-to-the-second revenue monitoring, displaying temporal performance in a responsive UI.

### 2. Predictive Intelligence (Sales & Demand ML)
* **Time-Series Forecasting:** Utilizes a Holt-Winters exponential smoothing model to mathematically project revenue 6 months into the future.
* **Random Forest Predictor:** A supervised machine learning baseline to forecast specific transaction scenarios based on quantity, discount, and region.

### 3. Customer Intelligence (Unsupervised Learning)
Executes an unsupervised **K-Means clustering** algorithm on historical data, performing RFM (Recency, Frequency, Monetary) analysis to dynamically segment customer behavior into multi-tiered categories (e.g., Loyalists, High-Value, Churn-Risk).

### 4. Model Diagnostics & Explainable AI (XAI)
Provides academic rigor by actively calculating and displaying model error margins ($R^2$, MAE, RMSE) and extracting algorithmic **Feature Importance** to explain exactly how the AI is making its baseline predictions.

### 5. Algorithmic Executive Reporting
A background processing engine that bypasses frontend screen-capturing. It uses Matplotlib to silently generate high-resolution graphs, algorithmically writes a human-readable narrative based on current system performance, and exports a fully formatted, multi-page PDF summary.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Enterprise-AI-Analytics-Suite.git](https://github.com/SyedRazaZaidi/Enterprise-AI-Analytics-Suite.git)
   cd Enterprise-AI-Analytics-Suite
Install the required dependencies:

Bash
pip install fastapi uvicorn pandas scikit-learn statsmodels chart.js matplotlib fpdf joblib
Boot the FastAPI Orchestrator:

Bash
python api_server.py
Access the Dashboard:
Open your browser and navigate to http://127.0.0.1:8000/dashboard
