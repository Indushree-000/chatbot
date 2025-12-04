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

# ---------- Load Data and Models (safe) ----------
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

# User session storage (in-memory)
user_sessions = {}

# ---------- Predict Disease from symptom ----------
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

# ---------- ROUTES ----------
@app.route("/")
def home():
    if os.path.exists(path_in_base(os.path.join("templates", "index.html"))):
        return render_template("index.html")
    return "PCOS Chatbot API is running."

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ---------- BASE44 /predict ENDPOINT ----------
@app.route("/predict", methods=["POST"])
def diagnosis_api():
    """
    Base44 requires THIS endpoint.
    Input:  {"symptoms": "acne"}
    Output: {"diagnosis": "...", "confidence": 0.xx}
    """
    try:
        data = request.get_json(force=True)
        symptoms_text = data.get("symptoms", "").strip()
    except:
        return jsonify({"error": "Invalid JSON"}), 400

    if not symptoms_text:
        return jsonify({"diagnosis": "Unknown", "confidence": 0}), 200

    disease = predict_disease(symptoms_text)

    if not disease:
        return jsonify({"diagnosis": "Unknown", "confidence": 0})

    # PCOS
    if disease == "PCOS" and pcos_model is not None and pcos_vectorizer is not None:
        try:
            X = pcos_vectorizer.transform([symptoms_text])
            prob = float(pcos_model.predict_proba(X)[0][1])
            return jsonify({"diagnosis": "PCOS", "confidence": prob})
        except:
            return jsonify({"diagnosis": "PCOS", "confidence": 0.50})

    # Breast Cancer
    if disease == "Breast Cancer":
        return jsonify({"diagnosis": "Breast Cancer", "confidence": 0.75})

    # Others
    return jsonify({"diagnosis": disease, "confidence": 0.50})

# ---------- CHATBOT /get ENDPOINT ----------
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
            questions_for_lang = followup_data.get(disease, {}).get(lang, [])
            user_sessions[user_id] = {
                "disease": disease,
                "questions": questions_for_lang.copy() if isinstance(questions_for_lang, list) else [],
                "answers": [],
                "total_questions": len(questions_for_lang)
            }
            if user_sessions[user_id]["questions"]:
                first_q = user_sessions[user_id]["questions"].pop(0)
                return jsonify({"response": first_q, "progress": 0})
            else:
                return jsonify({"response": f"Symptoms indicate **{disease}**, but no follow-up questions found.", "progress": 0})

        else:
            unknown_text = {
                "en": "🤔 I'm not sure. Please describe your symptoms more clearly.",
                "hi": "🤔 मुझे यकीन नहीं है। कृपया अपने लक्षणों को और स्पष्ट रूप से बताएं।",
                "kn": "🤔 ಖಚಿತವಾಗಿ ತಿಳಿಯುವುದಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ವಿವರಿಸಿ."
            }
            return jsonify({"response": unknown_text.get(lang, unknown_text["en"]), "progress": 0})

    session = user_sessions[user_id]
    if user_input:
        session["answers"].append(user_input[:1000])

    total = session.get("total_questions", 0)
    answered = len(session["answers"])
    progress = int((answered / total) * 100) if total > 0 else 100

    if session.get("questions"):
        next_q = session["questions"].pop(0)
        return jsonify({"response": next_q, "progress": progress})

    disease = session.get("disease")
    del user_sessions[user_id]
    probability = 0

    if disease == "PCOS" and pcos_model and pcos_vectorizer:
        try:
            X = pcos_vectorizer.transform([" ".join(session["answers"])])
            probability = float(pcos_model.predict_proba(X)[0][1]) * 100
        except:
            probability = 0

    elif disease == "Breast Cancer" and bc_model and bc_scaler and bc_columns:
        try:
            input_dict = {col: 0 for col in bc_columns}
            for i, col in enumerate(bc_columns):
                if i < len(session["answers"]):
                    ans = session["answers"][i].lower()
                    input_dict[col] = 1 if ans in ("yes", "y", "true", "1") else 0

            df_input = pd.DataFrame([input_dict])
            df_scaled = bc_scaler.transform(df_input)
            probability = float(bc_model.predict_proba(df_scaled)[0][1]) * 100
        except:
            probability = 0

    final_texts = {
        "en": f"Based on your responses, chance of **{disease}** is **{probability:.1f}%**.",
        "hi": f"आपके उत्तरों के आधार पर **{disease}** होने की संभावना **{probability:.1f}%** है।",
        "kn": f"ನಿಮ್ಮ ಉತ್ತರಗಳ ಆಧಾರದ ಮೇಲೆ **{disease}** ಸಂಭವಿಸುವ ಸಾಧ್ಯತೆ **{probability:.1f}%**."
    }

    return jsonify({"response": final_texts.get(lang, final_texts["en"]), "progress": 100})

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))



