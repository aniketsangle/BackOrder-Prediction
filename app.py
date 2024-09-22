import streamlit as st
import pandas as pd
import sklearn
import pickle

# Load the model and preprocessing components using pickle
with open(r"C:\Users\adity\PycharmProjects\pythonProject2\randomforestmodel.pkl", 'rb') as file:
    model = pickle.load(file)

with open(r"C:\Users\adity\PycharmProjects\pythonProject2\labelencoders.pkl", 'rb') as file:
    label_encoders = pickle.load(file)

with open(r"C:\Users\adity\PycharmProjects\pythonProject2\targetencoder.pkl", 'rb') as file:
    target_encoder = pickle.load(file)

with open(r"C:\Users\adity\PycharmProjects\pythonProject2\scaler (1).pkl", 'rb') as file:
    scaler = pickle.load(file)

# Specify features in the required order
numerical_features = ['forecast_3_month', 'national_inv', 'sales_3_month',
                      'perf_6_month_avg', 'in_transit_qty', 'min_bank',
                      'lead_time', 'local_bo_qty', 'pieces_past_due']

categorical_features = ['potential_issue', 'oe_constraint', 'ppap_risk',
                        'deck_risk', 'rev_stop', 'stop_auto_buy']

# Features that should take only integer values
integer_features = ['forecast_3_month', 'national_inv', 'sales_3_month',
                    'in_transit_qty', 'min_bank', 'lead_time',
                    'local_bo_qty', 'pieces_past_due']

# Features that should take values between 0 and 1 in decimal
decimal_features = ['perf_6_month_avg']

# Function to preprocess input data
def preprocess_input(data):
    # Encode categorical features
    for feature in categorical_features:
        le = label_encoders.get(feature)
        if le:
            data[feature] = le.transform(data[feature])

    # Standardize numerical features
    data[numerical_features] = scaler.transform(data[numerical_features])

    return data

# Streamlit App
def main():
    st.title("Back Order Prediction")

    # Input Form
    with st.form("prediction_form"):
        st.write("Enter the details for prediction:")

        # Numerical Inputs for integer values
        numerical_inputs = {}
        for feature in integer_features:
            numerical_inputs[feature] = st.number_input(
                feature.replace('_', ' ').capitalize(),
                value=0,
                min_value=0,
                step=1,
                format="%d"
            )

        # Numerical Inputs for decimal values between 0 and 1
        for feature in decimal_features:
            numerical_inputs[feature] = st.number_input(
                feature.replace('_', ' ').capitalize(),
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f"
            )

        # Categorical Inputs
        categorical_inputs = {}
        for feature in categorical_features:
            le = label_encoders.get(feature)
            if le:
                categorical_inputs[feature] = st.selectbox(
                    feature.replace('_', ' ').capitalize(),
                    le.classes_
                )

        # Submit Button
        submit = st.form_submit_button("Predict")

    if submit:
        # Combine numerical and categorical inputs
        input_data = {**numerical_inputs, **categorical_inputs}

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Ensure the correct order of columns
        input_df = input_df[numerical_features + categorical_features]

        # Preprocess the input data
        preprocessed_data = preprocess_input(input_df.copy())

        # Make prediction
        prediction = model.predict(preprocessed_data)
        prediction_label = target_encoder.inverse_transform(prediction)[0]

        result = "The Order Went On BackOrder" if prediction_label == "Yes" else "The Order Did Not Go On BackOrder"
        color = "red" if prediction_label == "Yes" else "green"

        # Display Result
        st.markdown(f"<h2 style='color: {color};'>{result}</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
