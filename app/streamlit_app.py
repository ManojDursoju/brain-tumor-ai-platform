import streamlit as st
from PIL import Image
import pandas as pd
import requests

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Brain Tumor AI Platform",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor AI Medical Intelligence Platform")

st.markdown("""
Upload an MRI scan to:

- Detect the brain tumor type
- View Grad-CAM visualization
- Generate an AI medical report
- Download a PDF report
- Review previous predictions
""")

st.divider()

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded MRI Image")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )

    file_size = uploaded_file.size / 1024

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Filename:** {uploaded_file.name}")

    with col2:
        st.write(f"**Size:** {file_size:.2f} KB")

    st.divider()

    if st.button(
        "Click Here to Predict Brain Tumor",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Running AI prediction..."):

            try:

                # ----------------------------------
                # Prediction API
                # ----------------------------------
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files
                )

                if response.status_code != 200:
                    st.error("Prediction failed.")
                    st.code(response.text)
                    st.stop()

                result = response.json()

                st.success("Prediction completed successfully!")

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Tumor",
                        result["prediction"].title()
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.2f}%"
                    )

                # ----------------------------------
                # Grad-CAM API
                # ----------------------------------
                st.divider()

                st.subheader("Grad-CAM Visualization")

                uploaded_file.seek(0)

                gradcam_files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )
                }

                gradcam_response = requests.post(
                    f"{API_URL}/gradcam",
                    files=gradcam_files
                )

                if gradcam_response.status_code == 200:

                    gradcam_result = gradcam_response.json()

                    gradcam_image = Image.open(
                        gradcam_result["gradcam_path"]
                    )

                    st.image(
                        gradcam_image,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )

                else:
                    st.error("Failed to generate Grad-CAM.")
                    st.code(gradcam_response.text)

                # ----------------------------------
                # AI Medical Report API
                # ----------------------------------
                st.divider()

                st.subheader(" AI Medical Report")

                uploaded_file.seek(0)

                report_files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )
                }

                report_response = requests.post(
                    f"{API_URL}/report",
                    files=report_files
                )

                if report_response.status_code == 200:

                    report = report_response.json()

                    st.markdown("###  Diagnosis")
                    st.success(report["prediction"].title())

                    st.markdown("###  Confidence")
                    st.info(f"{report['confidence']:.2f}%")

                    st.markdown("###  Description")
                    st.write(report["description"])

                    st.markdown("###  Recommendations")

                    for recommendation in report["recommendations"]:
                        st.write(f"• {recommendation}")

                    st.markdown("###  Disclaimer")
                    st.warning(report["disclaimer"])

                else:
                    st.error("Failed to generate AI medical report.")
                    st.code(report_response.text)

                # ----------------------------------
                # PDF Report Download
                # ----------------------------------
                st.divider()

                st.subheader(" Download Medical Report")

                uploaded_file.seek(0)

                pdf_files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )
                }

                pdf_response = requests.post(
                    f"{API_URL}/report/pdf",
                    files=pdf_files
                )

                if pdf_response.status_code == 200:

                    st.download_button(
                        label="⬇ Download PDF Report",
                        data=pdf_response.content,
                        file_name="brain_tumor_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                else:
                    st.error("Failed to generate PDF report.")
                    st.code(pdf_response.text)

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to the FastAPI server.\n\n"
                    "Please make sure FastAPI is running:\n\n"
                    "uvicorn src.api.main:app --reload"
                )

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

else:
    st.info(
        "Please upload a brain MRI image to begin analysis."
    )
    # ----------------------------------
# Prediction History
# ----------------------------------
st.divider()

st.subheader(" Prediction History")

history_response = requests.get(
    f"{API_URL}/predictions"
)

if history_response.status_code == 200:

    history = history_response.json()

    if len(history) == 0:

        st.info("No prediction history available.")

    else:

        df = pd.DataFrame(history)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.markdown("###  Delete Prediction")

        prediction_ids = df["id"].tolist()

        selected_id = st.selectbox(
            "Select Prediction ID",
            prediction_ids
        )

        if st.button(
            "Delete Selected Prediction",
            use_container_width=True
        ):

            delete_response = requests.delete(
                f"{API_URL}/predictions/{selected_id}"
            )

            if delete_response.status_code == 200:
                st.success("Prediction deleted successfully.")
                st.rerun()
            else:
                st.error("Unable to delete prediction.")

else:

    st.error("Unable to load prediction history.")