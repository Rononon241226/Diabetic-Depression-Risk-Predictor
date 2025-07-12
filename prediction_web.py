import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib



# ========== 1. 加载模型与预处理器 ==========
model = joblib.load("model/best_model_xgb.pkl")  # 你保存好的模型
preprocessor = model.named_steps["pre"]
clf = model.named_steps["clf"]

# 获取变量名
numerical_cols = ["Age", "ADL_Limitations"]
categorical_cols = ["Gender", "SelfRatedHealth", "ReportedPsychIssue", "ExerciseFreq", "EmploymentStatus", "Vision"]

cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
final_feature_names = numerical_cols + list(cat_feature_names)

# ========== 2. 页面设置 ==========
st.set_page_config(page_title="Diabetic Depression Risk Predictor", layout="centered")
st.title("🧠 Depression Risk Prediction in People with Diabetes")
st.markdown(
    "This tool uses machine learning to estimate the probability of depression in adults aged 45–85 with **diabetes**, "
    "based on demographic and health-related information."
)

# ========== 3. 用户输入 ==========
gender = st.selectbox("Gender", ["Male", "Female"])
self_health = st.selectbox("Self-Rated Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
psych = st.selectbox("Reported Psychological Problem", ["Yes", "No"])
exercise = st.selectbox("Exercise Frequency", ["Every day", "1 per week", "1-3 per mon", "Never"])
employment = st.selectbox("Employment Status", [
    "Working full-time", "Working part-time", "Unemployed", "Partly retired", "Retired", "Disabled", "Not in LbrF"
])
vision = st.selectbox("Vision", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
age = st.slider("Age", 45, 85, 60)
adl = st.slider("ADL Limitations", 0, 5, 0)
if st.button("🔍 Predict Depression Risk"):
    # ========== 4. 构建 DataFrame ==========
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "SelfRatedHealth": self_health,
        "ADL_Limitations": adl,
        "ReportedPsychIssue": psych,
        "ExerciseFreq": exercise,
        "EmploymentStatus": employment,
        "Vision": vision
    }])

    # ========== 5. 预处理 ==========
    processed = preprocessor.transform(input_df)
    prob = clf.predict_proba(processed)[0, 1]
    pred = clf.predict(processed)[0]

    st.subheader("🧾 Prediction Result")
    st.write(f"**Predicted Probability of Depression:** `{prob:.2f}`")
    st.write(f"**Prediction:** {'Yes' if pred else 'No'}")

    st.markdown("###Interpretation Suggestion")
    if prob >= 0.5:
        st.markdown(f"""
            Based on your input, the model estimates a **{prob:.0%} chance** of experiencing depressive symptoms.

            **Recommendation:** You may be at elevated risk of depression. It is recommended to consult a mental health professional 
            or talk to your primary care provider for further evaluation.
            """)
    else:
        st.markdown(f"""
            Based on your input, the model estimates a **{prob:.0%} chance** of depression, which is relatively low.

            **Recommendation:** Currently no major concern. Maintaining regular physical activity and strong social support 
            remains beneficial for mental well-being.
            """)
    # ========== 6. 显示 SHAP 解读 ==========
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(processed)

    if hasattr(shap_values, "toarray"):
        shap_values = shap_values.toarray()

    st.subheader("🔍 Model Interpretation")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0].flatten(),
        feature_names=final_feature_names,
        max_display=10
    )
    st.pyplot(fig)