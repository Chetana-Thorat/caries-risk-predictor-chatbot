# 🦷 Caries Risk Predictor & Dental Chatbot

An AI-powered web application that predicts a patient's **Caries Risk Level** — High, Moderate, or Low — based on clinical questionnaire inputs, and provides personalized dental advice through a chatbot interface.

[🚀 Live Demo](https://caries-risk-predictor-chatbot.streamlit.app)

---

## 💡 What This Project Does

- Takes basic patient information (Age, Gender, Year) and 7 clinical input questions from a caries risk assessment form.
- Predicts the **Caries Risk Status** using a trained **XGBoost classifier**.
- Displays results in a clear summary with user-friendly formatting.
- Offers a **chat interface** powered by Groq’s LLaMA3 model for personalized dental advice and follow-up questions.

---

## 🛠️ Tech Stack

| Purpose                     | Technology                      |
|----------------------------|----------------------------------|
| UI / Frontend              | Streamlit                        |
| ML Model                   | XGBoost, scikit-learn            |
| Data Processing            | pandas, joblib, LabelEncoder     |
| LLM Chat Assistant         | Groq API (`llama3-8b-8192`)      |
| Deployment                 | Streamlit Cloud                  |
| Environment Variables      | Python `os` module               |

---

## 🤖 How It Works

1. **Input Form**: Users provide demographic info and responses to clinical risk questions.
2. **Data Encoding**: Inputs are transformed to match training schema using `get_dummies`.
3. **Prediction**: The trained model returns a risk class (High / Moderate / Low).
4. **Chatbot**: Based on the prediction and symptoms, the chatbot generates helpful advice using Groq LLM.

---

## 🧪 Sample Questions You Can Ask the Bot

- *“What does it mean if my risk is HIGH?”*
- *“How can I reduce my caries risk at home?”*
- *“Is enamel lesion serious?”*

---

## 👩‍💻 Author

**Chetana Thorat**  
Graduate Student, MS Data Science  
Indiana University Bloomington  
[GitHub Profile](https://github.com/Chetana-Thorat)

---
