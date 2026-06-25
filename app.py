import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ── Train model on startup (no pkl file needed) ──────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'temperature':   np.random.uniform(15, 40, n),
        'humidity':      np.random.uniform(30, 100, n),
        'rainfall':      np.random.uniform(0, 200, n),
        'soil_moisture': np.random.uniform(10, 80, n),
        'soil_ph':       np.random.uniform(4.5, 8.5, n),
    })

    def label(row):
        if row['humidity'] > 70 and 18 < row['temperature'] < 30 and row['rainfall'] > 10:
            return 2  # High Risk
        elif row['humidity'] > 55 or row['rainfall'] > 5:
            return 1  # Medium Risk
        else:
            return 0  # Low Risk

    data['risk'] = data.apply(label, axis=1)
    X = data.drop('risk', axis=1)
    y = data['risk']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ── Disease & Recommendation data ────────────────────────────────────────────
def get_disease_info(risk_level, crop):
    diseases = {
        "Tomato": {
            2: ("Late Blight (Phytophthora infestans)", [
                "🧪 Apply copper-based fungicide within 24–48 hours",
                "💧 Avoid overhead irrigation — switch to drip",
                "🌿 Remove and destroy infected leaves immediately",
                "👀 Monitor crop daily for next 7 days"
            ]),
            1: ("Early Blight Risk Detected", [
                "🧪 Apply preventive fungicide spray",
                "💧 Ensure proper field drainage",
                "🌱 Maintain optimal plant spacing",
                "📋 Monitor weather forecast closely"
            ]),
            0: ("No Significant Disease Risk", [
                "✅ Conditions are currently safe",
                "📅 Continue regular crop monitoring",
                "📊 Maintain soil nutrition schedule"
            ])
        },
        "Rice": {
            2: ("Rice Blast (Magnaporthe oryzae)", [
                "🧪 Apply Tricyclazole or Isoprothiolane fungicide",
                "💧 Drain fields temporarily to reduce humidity",
                "🌾 Avoid excessive nitrogen fertiliser",
                "👀 Scout for lesions on leaves every 2 days"
            ]),
            1: ("Brown Spot Risk Detected", [
                "🌱 Improve soil potassium levels",
                "💧 Ensure uniform water distribution",
                "🧪 Prepare fungicide as precaution"
            ]),
            0: ("No Significant Disease Risk", [
                "✅ Conditions are currently safe",
                "📅 Continue regular crop monitoring"
            ])
        },
        "Wheat": {
            2: ("Yellow Rust (Puccinia striiformis)", [
                "🧪 Apply Propiconazole fungicide immediately",
                "🌬️ Improve air circulation between rows",
                "🌾 Avoid dense sowing in next season",
                "👀 Check for yellow stripes on leaves"
            ]),
            1: ("Powdery Mildew Risk", [
                "🧪 Apply sulfur-based fungicide",
                "💨 Ensure good field ventilation",
                "📋 Monitor humidity levels daily"
            ]),
            0: ("No Significant Disease Risk", [
                "✅ Conditions are currently safe",
                "📅 Continue regular crop monitoring"
            ])
        },
        "Cotton": {
            2: ("Bacterial Blight (Xanthomonas citri)", [
                "🧪 Apply copper oxychloride spray",
                "✂️ Remove and burn infected plant parts",
                "🚫 Avoid working in wet fields",
                "🌱 Use disease-resistant varieties next season"
            ]),
            1: ("Boll Rot Risk Detected", [
                "💧 Reduce irrigation frequency",
                "🌬️ Improve canopy air circulation",
                "👀 Inspect bolls regularly"
            ]),
            0: ("No Significant Disease Risk", [
                "✅ Conditions are currently safe",
                "📅 Continue regular crop monitoring"
            ])
        },
        "Potato": {
            2: ("Late Blight (Phytophthora infestans)", [
                "🧪 Apply Mancozeb + Cymoxanil fungicide",
                "💧 Avoid furrow irrigation during outbreak",
                "🥔 Hill up soil around plants",
                "👀 Check undersides of leaves for white spores"
            ]),
            1: ("Early Blight Risk Detected", [
                "🧪 Apply chlorothalonil spray",
                "🌱 Ensure adequate plant nutrition",
                "📋 Monitor for brown spots on older leaves"
            ]),
            0: ("No Significant Disease Risk", [
                "✅ Conditions are currently safe",
                "📅 Continue regular crop monitoring"
            ])
        }
    }
    return diseases[crop][risk_level]

def get_risk_score(probs):
    return int(probs[0] * 10 + probs[1] * 55 + probs[2] * 100)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FarmShield AI",
    page_icon="🌿",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#1B4332;padding:24px;border-radius:12px;
            text-align:center;margin-bottom:24px'>
    <h1 style='color:white;margin:0;font-size:2.2em'>🌿 FarmShield AI</h1>
    <p style='color:#52B788;margin:6px 0 0 0;font-size:1.05em'>
        Early Crop Disease Forecasting &amp; Yield Optimization
    </p>
    <p style='color:#95D5B2;margin:4px 0 0 0;font-size:0.88em'>
        Predict. Prevent. Protect. Prosper.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown("### 🌾 Enter Your Field Conditions")

col1, col2 = st.columns(2)

with col1:
    crop         = st.selectbox("Crop Type",
                      ["Tomato", "Rice", "Wheat", "Cotton", "Potato"])
    temperature  = st.slider("Temperature (°C)", 10.0, 45.0, 28.5)
    rainfall     = st.slider("Rainfall (mm)", 0.0, 200.0, 12.4)

with col2:
    growth_stage = st.selectbox("Growth Stage",
                      ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"])
    humidity     = st.slider("Humidity (%)", 20.0, 100.0, 65.0)
    soil_moisture= st.slider("Soil Moisture (%)", 10.0, 80.0, 35.0)

soil_ph = st.slider("Soil pH", 4.5, 8.5, 6.5)

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Analyse Disease Risk", use_container_width=True):

    features    = np.array([[temperature, humidity, rainfall, soil_moisture, soil_ph]])
    prediction  = model.predict(features)[0]
    probs       = model.predict_proba(features)[0]
    risk_score  = get_risk_score(probs)
    disease_name, recommendations = get_disease_info(prediction, crop)

    # Colour scheme by risk
    if prediction == 2:
        color, risk_label, border = "#E63946", "🔴 HIGH RISK", "#E63946"
    elif prediction == 1:
        color, risk_label, border = "#F4A261", "🟡 MEDIUM RISK", "#F4A261"
    else:
        color, risk_label, border = "#2D6A4F", "🟢 LOW RISK", "#52B788"

    # Risk card
    st.markdown(f"""
    <div style='background:#f8f9fa;border:2px solid {border};
                border-radius:12px;padding:20px;margin:10px 0'>
        <h2 style='color:{color};margin:0'>{risk_label}</h2>
        <h1 style='color:{color};margin:4px 0;font-size:3em'>
            {risk_score}<span style='font-size:0.4em'>/100</span>
        </h1>
        <p style='color:#333;font-size:1.1em;margin:8px 0 0 0'>
            <b>Likely Disease:</b> {disease_name}
        </p>
        <p style='color:#666;font-size:0.9em;margin:4px 0 0 0'>
            Forecast window: next 5–7 days &nbsp;|&nbsp;
            Crop: {crop} &nbsp;|&nbsp; Stage: {growth_stage}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("### 📋 Recommended Actions")
    for rec in recommendations:
        st.markdown(f"""
        <div style='background:white;border-left:4px solid #52B788;
                    padding:10px 14px;margin:6px 0;border-radius:4px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.08)'>
            {rec}
        </div>
        """, unsafe_allow_html=True)

    # Yield tip
    st.markdown("### 📈 Yield Optimization Tip")
    yield_tips = {
        2: "⚠️ Act within 48 hours. Prompt treatment can save up to 80% of projected yield loss.",
        1: "📊 Take preventive measures now. Early action typically improves yield by 15–20%.",
        0: "✅ Maintain current practices. Optimal conditions — focus on fertiliser timing for best yield."
    }
    st.info(yield_tips[prediction])

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#aaa;font-size:0.82em'>
    FarmShield AI · TechExpo Hackathon 2026 · Predict. Prevent. Protect. Prosper.
</p>
""", unsafe_allow_html=True)
