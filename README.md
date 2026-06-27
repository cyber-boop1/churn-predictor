A machine learning web app that predicts whether a customer is likely to churn (leave) based on their usage profile — built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## 🌐 Live Demo
👉 [Click here to try the app](https://churn-predictor-kb9ctfjwse6kk2utyzfu2d.streamlit.app/)

---

## ✨ Features
- 🔮 Predicts churn probability as a percentage
- ⚠️ Flags high-risk customers with recommended retention actions
- 📊 Shows churn risk meter with key metrics
- 🎯 Trained on 2000 customer records using Gradient Boosting
- 🎨 Clean, interactive Streamlit UI

---

## 🛠 Tech Stack
| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Data generation & manipulation |
| Scikit-learn | ML model (Gradient Boosting Classifier) |
| Streamlit | Web app frontend |
| Pickle | Model serialization |

---

## 📁 Project Structure
```
churn-predictor/
├── app.py              # Main Streamlit app
├── train_model.py      # Model training script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/cyber-boop1/churn-predictor.git
cd churn-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🤖 How the ML Model Works
1. **Data** — 2000 synthetic customer records with realistic churn patterns
2. **Features** — Tenure, monthly charges, total charges, contract type, internet service, payment method, tech support, senior citizen status
3. **Model** — Gradient Boosting Classifier (100 estimators)
4. **Output** — Churn probability (%) + High/Low risk classification

---

## 📈 Key Churn Indicators
| Factor | Impact on Churn |
|---|---|
| Short tenure (< 12 months) | High |
| High monthly charges (> ₹6000) | High |
| Month-to-month contract | High |
| No tech support | Medium |
| Long-term contract (2 years) | Low |

---

## 🔮 Future Improvements
- [ ] Use real telecom dataset (IBM Watson / Kaggle)
- [ ] Add SHAP values for model explainability
- [ ] Add batch prediction (upload CSV of customers)
- [ ] Add email alert for high-risk customers

---

## 👩‍💻 Author
**Shalini Jaiswal**
- GitHub: [@cyber-boop1](https://github.com/cyber-boop1)
- LinkedIn: [jaiswal-shalini](https://linkedin.com/in/jaiswal-shalini)

---
⭐ If you found this useful, please give it a star!
