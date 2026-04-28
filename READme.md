# 🌍 Earthquake Analytics AI Platform

An end-to-end Data Engineering + AI project that ingests earthquake data, transforms it using a medallion architecture, builds machine learning predictions, generates forecasts, and provides an interactive AI-powered dashboard.

---

## 🚀 Project Overview

This project demonstrates a full data pipeline lifecycle:

USGS API → Bronze → Silver → Gold → ML Model → Forecast → Explainability → Dashboard → AI Assistant

It combines:
- Data Engineering  
- Machine Learning  
- Forecasting  
- Explainable AI  
- Interactive dashboards  
- Chat-style AI assistant (offline)

---

## 📊 Features

### 🔹 Data Pipeline (Medallion Architecture)

- **Bronze Layer**
  - Raw earthquake data from USGS API
  - Stored as JSON

- **Silver Layer**
  - Cleaned and structured dataset
  - Converted to CSV

- **Gold Layer**
  - Aggregated analytics
  - Daily summaries
  - Magnitude distributions

---

### 🤖 Machine Learning

- Predicts high vs normal earthquake activity days  
- Uses features like:
  - Event counts  
  - Magnitude  
  - Depth  
  - Previous-day activity  

Outputs:
- earthquake_predictions.csv  
- Model metrics (accuracy, etc.)

---

### 🔮 Forecasting

- Generates 7-day forward forecast  
- Includes:
  - Risk score  
  - Risk label (Low / Medium / High)  
  - Forecasted event volume  

---

### 📈 Explainability

- Feature importance using ML model  
- Identifies top drivers of earthquake activity  

---

### 💬 AI Assistant (Offline)

- Chat-style interface inside dashboard  
- Answers:
  - Forecast risk  
  - Feature importance  
  - Activity trends  
  - Project architecture  
- Can draw graphs on request  

Example questions:
- What is the forecast risk?  
- Which feature is most important?  
- Draw a graph of forecast risk  
- Explain the medallion architecture  

---

### 📊 Dashboard (Streamlit)

Interactive dashboard with:
- KPIs  
- Maps  
- Trends over time  
- Forecast visualizations  
- Feature importance charts  
- AI chat assistant  

---

## 🏗️ Project Structure

earthquake_pipeline/

├── src/  
│   ├── bronze.py  
│   ├── silver.py  
│   ├── gold.py  
│   ├── model.py  
│   ├── forecast.py  
│   ├── feature_importance.py  
│   ├── llm.py  
│   ├── config.py  
│   └── __init__.py  

├── dashboard/  
│   └── app.py  

├── data/  
│   ├── bronze/  
│   ├── silver/  
│   └── gold/  

├── docker-compose.yml  
├── requirements.txt  
└── README.md  

---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone <your-repo-url>  
cd earthquake_pipeline  

---

### 2. Create virtual environment

python -m venv .venv  

Activate:

Windows:  
.venv\Scripts\activate  

Mac/Linux:  
source .venv/bin/activate  

---

### 3. Install dependencies

pip install -r requirements.txt  

---

## ▶️ Run the Pipeline

### Step 1: Bronze

python -m src.bronze  

### Step 2: Silver

python -m src.silver  

### Step 3: Gold

python -m src.gold  

### Step 4: Machine Learning

python -m src.model  

### Step 5: Forecast

python -m src.forecast  

### Step 6: Feature Importance

python -m src.feature_importance  

---

## 📊 Run Dashboard

streamlit run dashboard/app.py  

---

## 💡 Example Questions

- What is the forecast risk?  
- Which feature is most important?  
- How many high activity days are there?  
- Draw a graph of forecast risk  
- Explain Airflow  
- Explain medallion architecture  

---

## 🧠 Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- USGS Earthquake API  
- Airflow (optional)  
- Docker (optional)  

---

## 📌 Key Learnings

- Designing medallion architecture  
- Building ETL pipelines  
- Working with real-world APIs  
- Creating ML prediction models  
- Forecasting future events  
- Explainable AI  
- Chat-based analytics  

---

## 🚀 Future Improvements

- Cloud deployment (AWS / Streamlit Cloud)  
- Real-time data streaming  
- Weather API integration  
- Local LLM (RAG)  
- CI/CD pipeline  
- Authentication  

---

## 👨‍💻 Author

Tshepo Vilakazi  
Data Engineer  

---

