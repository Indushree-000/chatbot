import os
import logging
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- Helper: safe path resolver ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def path_in_base(filename):
    return os.path.join(BASE_DIR, filename)

# ---------- Load Data and Models ----------
pcos_data = None
pcos_vectorizer = None
pcos_model = None
bc_model = None
bc_scaler = None
bc_columns = None
followup_data = {}

try:
    pcos_data = pd.read_csv(path_in_base("symptom_disease.csv"))
    logger.info("Loaded symptom_disease.csv")
except Exception as e:
    logger.warning(f"Could not load symptom_disease.csv: {e}")

try:
    pcos_vectorizer = joblib.load(path_in_base("vectorizer.pkl"))
    pcos_model = joblib.load(path_in_base("pcos_model.pkl"))
    logger.info("Loaded PCOS vectorizer and model")
except Exception as e:
    logger.warning(f"Could not load PCOS model/vectorizer: {e}")

try:
    bc_model = joblib.load(path_in_base("breast_cancer_rf_model.pkl"))
    bc_scaler = joblib.load(path_in_base("scaler.pkl"))
    bc_columns = joblib.load(path_in_base("training_columns.pkl"))
    logger.info("Loaded Breast Cancer model, scaler and columns")
except Exception as e:
    logger.warning(f"Could not load Breast Cancer model/scaler/columns: {e}")

try:
    with open(path_in_base("followup_questions.json"), "r", encoding="utf-8") as f:
        followup_data = json.load(f)
    logger.info("Loaded followup_questions.json")
except Exception as e:
    logger.warning(f"Could not load followup_questions.json: {e}")

# Store user sessions
user_sessions = {}

# ---------- Symptom → Disease Predictor ----------
def predict_disease(user_input: str):
    if not user_input:
        return None

    user_input = user_input.lower()

    keyword_map = {
        "acne": "PCOS",
        "muttu": "PCOS",
        "ಮುಟ್ಟು": "PCOS",
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

    for word, disease in keyword_map.items():
        if word in user_input:
            return disease

    if pcos_data is not None and "Symptoms" in pcos_data.columns:
        for idx, row in pcos_data.iterrows():
            try:
                symptoms = [s.strip().lower() for s in str(row.get("Symptoms", "")).split(",")]
                if any(w in user_input for w in symptoms if w):
                    return row.get("Disease")
            except Exception:
                continue

    return None

# ---------- HOME ----------
@app.route("/")
def home():
    if os.path.exists(path_in_base(os.path.join("templates", "index.html"))):
        return render_template("index.html")
    return "PCOS Chatbot API is running. Use POST /predict or POST /get."

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ---------- BASE44 AI DIAGNOSIS ENDPOINT ----------
@app.post("/predict")
def base44_predict():
    """
    Required by Base44 AI Diagnosis Mode.
    Input: {"symptoms": "acne"}
    Output: {"diagnosis": "PCOS", "confidence": 0.85}
    """
    try:
        data = request.get_json(force=True)
        symptoms_text = data.get("symptoms", "").strip()

        if not symptoms_text:
            return jsonify({"error": "No symptoms provided"}), 400

        disease = predict_disease(symptoms_text)

        if disease:
            return jsonify({
                "diagnosis": disease,
                "confidence": 0.85
            })

        return jsonify({
            "diagnosis": "Unknown",
            "confidence": 0.0
        })

    except Exception as e:
        logger.exception(f"/predict error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# ---------- ORIGINAL CHATBOT ENDPOINT ----------
@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        data_json = request.get_json(force=True)
    except Exception:
        return jsonify({"response": "Invalid JSON"}), 400

    user_id = data_json.get("user_id", "default")
    user_input = data_json.get("msg", "").strip()
    lang = data_json.get("lang", "en")

    if user_id not in user_sessions:
        disease = predict_disease(user_input)
        if disease:
            questions_for_lang = followup_data.get(disease, {}).get(lang, []) if followup_data else []
            user_sessions[user_id] = {
                "disease": disease,
                "questions": questions_for_lang.copy(),
                "answers": [],
                "total_questions": len(questions_for_lang)
            }
            if user_sessions[user_id]["questions"]:
                first_q = user_sessions[user_id]["questions"].pop(0)
                return jsonify({"response": first_q, "progress": 0})
            else:
                return jsonify({
                    "response": f"✅ Symptoms indicate **{disease}**, but no follow-up questions configured.",
                    "progress": 0
                })
        else:
            unknown_text = {
                "en": "🤔 I'm not sure. Please describe your symptoms more clearly.",
                "hi": "🤔 मुझे यकीन नहीं है। कृपया अपने लक्षणों का अधिक स्पष्ट विवरण दें।",
                "kn": "🤔 ಖಚಿತವಾಗಿ ತಿಳಿಯುವುದಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ವಿವರಿಸಿ."
            }
            return jsonify({"response": unknown_text.get(lang, unknown_text["en"]), "progress": 0})

    session = user_sessions[user_id]

    if len(user_input) > 0:
        session["answers"].append(user_input[:1000])

    total = session.get("total_questions", 0)
    answered = len(session.get("answers", []))
    progress = int((answered / total) * 100) if total else 100

    if session.get("questions"):
        next_q = session["questions"].pop(0)
        return jsonify({"response": next_q, "progress": progress})

    disease = session.get("disease")
    del user_sessions[user_id]
    probability = 0.0

    if disease == "PCOS" and pcos_model and pcos_vectorizer:
        try:
            text_input = " ".join(session.get("answers", []))
            X_input = pcos_vectorizer.transform([text_input])
            if hasattr(pcos_model, "predict_proba"):
                probability = float(pcos_model.predict_proba(X_input)[0][1]) * 100
            else:
                pred = pcos_model.predict(X_input)[0]
                probability = 100.0 if pred == 1 else 0.0
        except Exception as e:
            logger.exception(f"PCOS model error: {e}")

    elif disease == "Breast Cancer" and bc_model and bc_scaler and bc_columns:
        try:
            input_dict = {col: 0 for col in bc_columns}
            answers = session.get("answers", [])
            for i, col in enumerate(bc_columns):
                if i < len(answers):
                    ans = answers[i].strip().lower()
                    input_dict[col] = 1 if ans in ("yes", "y", "true", "1") else 0
            df_input = pd.DataFrame([input_dict])
            df_scaled = bc_scaler.transform(df_input)
            if hasattr(bc_model, "predict_proba"):
                probability = float(bc_model.predict_proba(df_scaled)[0][1]) * 100
            else:
                pred = bc_model.predict(df_scaled)[0]
                probability = 100.0 if pred == 1 else 0.0
        except Exception as e:
            logger.exception(f"Breast cancer model error: {e}")

    final_texts = {
        "en": f"✅ Based on your responses, your likelihood of **{disease}** is **{probability:.1f}%**.\nPlease consult a doctor.",
        "hi": f"✅ आपके उत्तरों के आधार पर **{disease}** होने की संभावना **{probability:.1f}%** है।\nकृपया डॉक्टर से संपर्क करें।",
        "kn": f"✅ ನಿಮ್ಮ ಪ್ರತಿಕ್ರಿಯೆಗಳ ಆಧಾರದ ಮೇಲೆ **{disease}** ಸಂಭವನೀಯತೆ **{probability:.1f}%**.\nದಯವಿಟ್ಟು ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    }

    return jsonify({"response": final_texts.get(lang, final_texts["en"]), "progress": 100})

# Run locally only — Railway uses Gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

