\# 📊 Customer Churn Prediction Project



A machine learning project to predict \*\*telecom customer churn\*\* using data analysis, modeling, and a Streamlit web app.



This project helps identify customers who are likely to leave (churn), enabling proactive retention strategies.



\## 📁 Project Structure



| File | Purpose |

|------|--------|

| `Churn\_EDA\_model\_development.ipynb` | Exploratory Data Analysis and model training |

| `Churn\_model\_metrics.ipynb` | Model comparison and performance evaluation |

| `stream\_app.py` | Interactive web app (Streamlit) for predictions |

| `requirements.txt` | Python dependencies |

| `WA\_Fn-UseC\_-Telco-Customer-Churn.csv` | Original dataset from IBM |

| `tel\_churn.csv`, `first\_telc.csv` | Processed/filtered datasets |



\## 🚀 How to Run



\### 1. Clone the repo

```bash

git clone https://github.com/AbhayChandrashekar/Customer-Churn-Prediction.git

cd Customer-Churn-Prediction

python -m venv venv
venv\Scripts\Activate


pip install -r requirements.txt

streamlit run stream_app.py
