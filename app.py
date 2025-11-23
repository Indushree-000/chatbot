from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import numpy as np

app = Flask(__name__)

# ---------- Load Data and Models ----------
# PCOS
pcos_data = pd.read_csv("symptom_disease.csv")
pcos_vectorizer = joblib.load("vectorizer.pkl")
pcos_model = joblib.load("pcos_model.pkl")

# Breast Cancer
bc_model = joblib.load("breast_cancer_rf_model.pkl")
bc_scaler = joblib.load("scaler.pkl")
bc_columns = joblib.load("training_columns.pkl")

# Multilingual follow-up questions
with open("followup_questions.json", "r", encoding="utf-8") as f:
    followup_data = json.load(f)

# User session storage
user_sessions = {}

# ---------- Predict Disease from symptom ----------
def predict_disease(user_input):
    user_input = user_input.lower()  # lowercase for uniformity

    # Keyword mapping for common symptoms (all lowercase!)
    keyword_map = {
        "acne": "PCOS",
        "muttu": "PCOS",        # English transliteration
        "ಮುಟ್ಟು": "PCOS",        # Kannada script
        "lump": "Breast Cancer",
        "breast": "Breast Cancer",
        "swelling": "Breast Cancer",
        "pain": "Breast Cancer",
        "nipple": "Breast Cancer",
        "discharge": "Breast Cancer",
        "fever": "Fever",
        "cold": "Cold",
        "cough": "Cold",
        "thirst": "Diabetes",
        "urinate": "Diabetes"
    }

    # Check keyword map
    for word, disease in keyword_map.items():
        if word in user_input:   # partial match works now
            return disease

    # Check PCOS CSV as fallback
    for idx, row in pcos_data.iterrows():
        symptoms = [s.strip().lower() for s in row["Symptoms"].split(",")]
        if any(word in user_input for word in symptoms):
            return row["Disease"]

    return None

# ---------- Flask Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    data_json = request.get_json()
    user_id = data_json.get("user_id", "default")
    user_input = data_json.get("msg", "").strip()
    lang = data_json.get("lang", "en")

    # ---------- Start new session ----------
    if user_id not in user_sessions:
        disease = predict_disease(user_input)
        if disease:
            user_sessions[user_id] = {
                "disease": disease,
                "questions": followup_data.get(disease, {}).get(lang, []).copy(),
                "answers": [],
                "total_questions": len(followup_data.get(disease, {}).get(lang, []))
            }
            if user_sessions[user_id]["questions"]:
                first_q = user_sessions[user_id]["questions"].pop(0)
                return jsonify({"response": first_q, "progress": 0})
            else:
                return jsonify({"response": f"✅ Symptoms indicate **{disease}**, but no follow-up questions are configured.", "progress": 0})
        else:
            unknown_text = {
                "en": "🤔 I'm not sure. Please describe your symptoms more clearly.",
                "hi": "🤔 मुझे यकीन नहीं है। कृपया अपने लक्षणों का अधिक स्पष्ट विवरण दें।",
                "kn": "🤔 ಖಚಿತವಾಗಿ ತಿಳಿಯುವುದಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ವಿವರಿಸಿ."
            }
            return jsonify({"response": unknown_text.get(lang, unknown_text["en"]), "progress": 0})

    # ---------- Existing session ----------
    session = user_sessions[user_id]
    session["answers"].append(user_input)

    total = session["total_questions"]
    answered = len(session["answers"])
    progress = int((answered / total) * 100) if total > 0 else 100

    if session["questions"]:
        next_q = session["questions"].pop(0)
        return jsonify({"response": next_q, "progress": progress})
    else:
        disease = session["disease"]
        del user_sessions[user_id]

        # ---------- Predict probability ----------
        probability = 0
        if disease == "PCOS":
            text_input = " ".join(session["answers"])
            X_input = pcos_vectorizer.transform([text_input])
            probability = pcos_model.predict_proba(X_input)[0][1] * 100

        elif disease == "Breast Cancer":
            input_dict = {col: 0 for col in bc_columns}
            for i, col in enumerate(bc_columns):
                if i < len(session["answers"]):
                    answer = session["answers"][i]
                    input_dict[col] = 1 if answer.lower() == "yes" else 0
            df_input = pd.DataFrame([input_dict])
            df_scaled = bc_scaler.transform(df_input)
            probability = bc_model.predict_proba(df_scaled)[0][1] * 100

        final_texts = {
            "en": f"✅ Based on your responses, your likelihood of **{disease}** is approximately **{probability:.1f}%**.\nPlease consult a healthcare professional for proper evaluation.",
            "hi": f"✅ आपके उत्तरों के आधार पर, आपके **{disease}** होने की संभावना लगभग **{probability:.1f}%** है।\nकृपया उचित मूल्यांकन के लिए स्वास्थ्य विशेषज्ञ से परामर्श लें।",
            "kn": f"✅ ನಿಮ್ಮ ಪ್ರತಿಕ್ರಿಯೆಗಳ ಆಧಾರದ ಮೇಲೆ, ನಿಮ್ಮ **{disease}** ಹೊಂದುವ ಸಾಧ್ಯತೆ ಸುಮಾರು **{probability:.1f}%**.\nದಯವಿಟ್ಟು ಸರಿಯಾದ ಮೌಲ್ಯಮಾಪನಕ್ಕಾಗಿ ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
        }

        return jsonify({"response": final_texts.get(lang, final_texts["en"]), "progress": 100})

if __name__ == "__main__":
    app.run(debug=True)


