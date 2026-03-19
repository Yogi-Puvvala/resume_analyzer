import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title = "AI Resume Screener",
    page_icon  = "R",
    layout     = "centered"
)

st.title("AI Resume Screener")
st.markdown("Upload your resume and select a target role to get instant feedback!")

# ── Inputs ──────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Resume (PDF only)",
    type = ["pdf"]
)

target_role = st.selectbox(
    "Target Job Role",
    options = sorted([
        "Information-Technology", "Engineering", "Accountant",
        "Finance", "Healthcare", "HR", "Designer", "Sales",
        "Digital-Media", "Banking", "Consultant", "Teacher",
        "Business-Development", "Public-Relations", "Advocate",
        "Aviation", "Agriculture", "Automobile", "BPO",
        "Chef", "Fitness", "Apparel", "Arts", "Construction"
    ])
)

# ── Predict ─────────────────────────────────────────
if st.button("Analyze Resume"):

    if uploaded_file is None:
        st.warning("Please upload a PDF resume first!")

    else:
        with st.spinner("Analyzing your resume..."):
            try:
                # ── Read bytes explicitly ───────────
                pdf_bytes = uploaded_file.read()

                response = requests.post(
                    f"{API_URL}/predict",
                    files = {"file": (uploaded_file.name,
                                      pdf_bytes,
                                      "application/pdf")},
                    data  = {"target_role": target_role}
                )

                if response.status_code == 200:
                    result = response.json()

                    # ── Match Score ─────────────────
                    st.subheader("Match Score")
                    score = result["match_score"]

                    if score >= 70:
                        st.success(f"Strong Match: {score}%")
                    elif score >= 40:
                        st.warning(f"Moderate Match: {score}%")
                    else:
                        st.error(f"Low Match: {score}%")

                    st.progress(min(max(int(score), 0), 100))

                    # ── Predicted Role ──────────────
                    st.subheader("Best Matched Role")
                    st.info(f"**{result['predicted_role']}**")

                    # ── Confidence Scores ───────────
                    st.subheader("Confidence per Role")
                    for role, conf in result["confidence_scores"].items():
                        st.progress(
                            min(int(conf), 100),
                            text = f"{role}: {conf}%"
                        )

                    # ── Skills ──────────────────────
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Present Skills")
                        if result["present_skills"]:
                            for skill in result["present_skills"]:
                                st.write(f"- {skill}")
                        else:
                            st.write("No matching skills found")

                    with col2:
                        st.subheader("Missing Skills")
                        if result["missing_skills"]:
                            for skill in result["missing_skills"]:
                                st.write(f"- {skill}")
                        else:
                            st.write("No missing skills!")

                    # ── Suggested Roles ─────────────
                    st.subheader("Suggested Roles")
                    for role in result["suggested_roles"]:
                        st.write(f"-> {role}")

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Connection Error: {e}")