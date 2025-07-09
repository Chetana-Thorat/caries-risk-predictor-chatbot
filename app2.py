import streamlit as st
import pandas as pd
import joblib
import os
from groq import Groq  # new import
import numpy as np

import json
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("model.pkl")

# Load feature columns
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Load label encoder classes and rebuild encoder
with open("label_encoder.json", "r") as f:
    label_classes = json.load(f)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_classes)

# Set your Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")  # or hardcode 'gsk-xxxxx...'
client = Groq(api_key="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")

# Question options
question_options = {
    "Number of teeth with cavitated or non cavitated (incipient) active lesions": ['1 or 2', '3 or more', 'missing'],
    "Most severe radiographically evident lesions": ['Dentin', 'Enamel only', 'missing'],
    "Presence of an exposed pulp, fistula or abscess": ['Yes', 'No', 'missing'],
    "Number of cavities/restorations/extractions due to caries in the last 3 years": ['3 or more', '1 or 2', 'missing'],
    "Conditions that may increase Risk: Exposed roots; Deep pits/fissures; fixed or removable appliances present; Defective restorations margins": ['1 or 2', '3 or more', 'missing'],
    "Unstimulated Saliva Flow": ['More than 0.2 ml/min', 'Less than .1 ml/min', '0.1-0.2 ml/min', 'missing'],
    'Visible dental plaque "evidence of sticky plaque stagnation in at risk areas"': ['Yes', 'No', 'missing']
}

st.set_page_config(page_title="Caries Risk Predictor + Advisor", page_icon="🦷")
st.title("🦷 Caries Risk Predictor & Chatbot Advisor")
st.markdown("Fill the form to predict your risk. Then ask for personalized advice!")

with st.form("risk_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ['Female', 'Male'])
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    with col2:
        year = st.selectbox("Year", list(range(2015, 2026)))

    inputs = {
        "Gender": gender,
        "Age": age,
        "Year": year
    }

    for question, options in question_options.items():
        inputs[question] = st.selectbox(question, options)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    st.subheader("✅ Prediction Summary")
    input_df = pd.DataFrame([inputs])
    st.dataframe(input_df.T.rename(columns={0: "Your Answer"}))

    input_df = input_df.astype(str).apply(lambda x: x.str.strip().str.lower())
    input_encoded = pd.get_dummies(input_df).reindex(columns=feature_columns, fill_value=0)

    pred_num = model.predict(input_encoded)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    st.success(f"🩺 **Predicted Caries Risk Level: {pred_label.upper()}**")
    st.session_state.prediction = pred_label
    st.session_state.user_input = inputs

if "prediction" in st.session_state:
    st.divider()
    st.subheader("💬 Ask the Chatbot for Help")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask how to reduce your risk, or anything else...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        try:
            # Format prompt for Groq
            symptoms = [f"{k}: {v}" for k, v in st.session_state.user_input.items() if k not in ["Gender", "Age", "Year"]]
            prompt = (
                f"Patient has caries risk level: {st.session_state.prediction.upper()}\n"
                f"Age: {st.session_state.user_input['Age']}, Gender: {st.session_state.user_input['Gender']}, Year: {st.session_state.user_input['Year']}\n"
                f"Symptoms: {', '.join(symptoms)}\n"
                f"The patient asks: {user_query}\n"
                f"Respond as a helpful dental assistant with step-by-step advice."
            )

            groq_response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a kind and smart dental assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            llm_reply = groq_response.choices[0].message.content.strip()

        except Exception as e:
            llm_reply = f"⚠️ Error generating advice: {e}"

        with st.chat_message("assistant"):
            st.markdown(llm_reply)
        st.session_state.messages.append({"role": "assistant", "content": llm_reply})