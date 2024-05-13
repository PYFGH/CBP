import streamlit as st
import pandas as pd
import joblib

# Function to load the trained model
@st.cache  # This decorator ensures that the function is only run once
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Function to make predictions
def predict_bankruptcy(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Main function to run the Streamlit app
def main():
    # Set page title and description
    st.title("Company Bankruptcy Prediction")
    st.write("This is a basic app to predict whether a company will go bankrupt based on its financial features.")

    # Load the trained model
    model_path = r "C:\Users\yapyf\Documents\Capstone\17 May\bankruptcy_prediction_model.pkl"  # Path to your trained model file
    model = load_model(model_path)

    # Add user input for feature values
    st.sidebar.header("Input Features")
    feature1 = st.sidebar.number_input("Feature 1:", value=0.0)
    feature2 = st.sidebar.number_input("Feature 2:", value=0.0)
    feature3 = st.sidebar.number_input("Feature 3:", value=0.0)
    feature4 = st.sidebar.number_input("Feature 4:", value=0.0)
    feature5 = st.sidebar.number_input("Feature 5:", value=0.0)
    # Selected top 5 features to predict based on the feature importances

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Feature 1': ['Total income/ total expense'],
        'Feature 2': ['Debt to equity ratio'],
        'Feature 3': ['Interest Coverage ratio'],
        'Feature 4': ['Borrowring dependency'],
        'Feature 5': ['Current ratio'],
            })

    # Make predictions
    if st.sidebar.button("Predict"):
        prediction = predict_bankruptcy(model, input_data)
        if prediction[0] == 1:
            st.sidebar.error("WARNING: Company is predicted to go bankrupt!")
        else:
            st.sidebar.success("Company is predicted to be safe from bankruptcy.")

if __name__ == "__main__":
    main()
