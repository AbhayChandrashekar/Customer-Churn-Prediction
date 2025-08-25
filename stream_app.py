import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Load model
model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Load images
logo_image = 'icone.png'
banner_image = 'image.png'

def main():
    # Page configuration
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon=logo_image,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar with compact details
    with st.sidebar:
        st.image(logo_image, use_container_width=True)
        st.markdown(
            """
            ### Welcome to the Churn Prediction System
            Leverage **AI-powered insights** to predict customer churn with high accuracy.
            
            #### Key Features:
            - Online predictions for single customers
            - Batch predictions for CSV uploads
            
            """
        )
        st.markdown("---")
        mode = st.selectbox("Select Prediction Mode:", ["Online Prediction", "Batch Prediction"])

    # Header section with optimized banner
    st.image(banner_image, use_container_width="always", caption="AI-Powered Customer Insights", output_format="PNG")
    st.title("Customer Churn Prediction Dashboard")
    st.markdown(
        """
        **Empower your business with predictive analytics.** 
        This system allows you to identify potential customer churn and take proactive measures to retain them.
        """
    )
    st.markdown("---")

    if mode == "Online Prediction":
        # Online prediction layout
        st.header("Enter Customer Details")
        st.write("Fill in the form below to predict churn probability for a single customer.")

        # Input form using columns for better organization
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["male", "female"])
            seniorcitizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Has Partner", ["yes", "no"])
            dependents = st.selectbox("Has Dependents", ["yes", "no"])
            phoneservice = st.selectbox("Has Phone Service", ["yes", "no"])
            multiplelines = st.selectbox("Has Multiple Lines", ["yes", "no", "no_phone_service"])
            internetservice = st.selectbox("Internet Service", ["dsl", "no", "fiber_optic"])

        with col2:
            onlinesecurity = st.selectbox("Online Security", ["yes", "no", "no_internet_service"])
            onlinebackup = st.selectbox("Online Backup", ["yes", "no", "no_internet_service"])
            deviceprotection = st.selectbox("Device Protection", ["yes", "no", "no_internet_service"])
            techsupport = st.selectbox("Tech Support", ["yes", "no", "no_internet_service"])
            streamingtv = st.selectbox("Streaming TV", ["yes", "no", "no_internet_service"])
            streamingmovies = st.selectbox("Streaming Movies", ["yes", "no", "no_internet_service"])
            contract = st.selectbox("Contract Type", ["month-to-month", "one_year", "two_year"])
            paperlessbilling = st.selectbox("Paperless Billing", ["yes", "no"])
            paymentmethod = st.selectbox(
                "Payment Method",
                ["bank_transfer_(automatic)", "credit_card_(automatic)", "electronic_check", "mailed_check"],
            )
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=240, value=0, step=1)
        monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        totalcharges = tenure * monthlycharges

        # Predict button and results
        if st.button("Predict Churn"):
            input_dict = {
                "gender": gender,
                "seniorcitizen": seniorcitizen,
                "partner": partner,
                "dependents": dependents,
                "phoneservice": phoneservice,
                "multiplelines": multiplelines,
                "internetservice": internetservice,
                "onlinesecurity": onlinesecurity,
                "onlinebackup": onlinebackup,
                "deviceprotection": deviceprotection,
                "techsupport": techsupport,
                "streamingtv": streamingtv,
                "streamingmovies": streamingmovies,
                "contract": contract,
                "paperlessbilling": paperlessbilling,
                "paymentmethod": paymentmethod,
                "tenure": tenure,
                "monthlycharges": monthlycharges,
                "totalcharges": totalcharges,
            }

            # Prediction
            X = dv.transform([input_dict])
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5

            # Display results
            st.markdown(
                f"""
                ### Prediction Result:
                - **Churn Risk:** {"Yes" if churn else "No"}
                - **Risk Score:** {y_pred:.2f}
                """
            )

    elif mode == "Batch Prediction":
        # Batch prediction section
        st.header(" Batch Predictions")
        st.write("Upload a CSV file with customer data for bulk predictions.")

        # File uploader
        file_upload = st.file_uploader("Upload CSV File", type=["csv"])

        if file_upload:
            try:
                data = pd.read_csv(file_upload)
                X = dv.transform(data.to_dict(orient="records"))
                y_pred = model.predict_proba(X)[:, 1]
                data["Churn Risk"] = y_pred >= 0.5
                data["Risk Score"] = y_pred

                # Display results
                st.success("Predictions completed successfully!")
                st.write(data)

                # Download button for results
                st.download_button(
                    label="Download Results",
                    data=data.to_csv(index=False),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
