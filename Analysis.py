import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string

st.set_page_config(
    page_title="NLP Sentiment Analysis",
    layout="wide"
)

# LOAD FILES
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# TEXT CLEANING FUNCTION
def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# PREDICTION FUNCTION
def predict_sentiment(text):

    cleaned = clean_text(text)

    # TF-IDF Transform
    vectorized = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(vectorized)[0]

    # Decode Label
    label = label_encoder.inverse_transform([prediction])[0]

    # Confidence Score
    confidence = 0.85

    try:

        # Get decision score
        score = model.decision_function(vectorized)

        # Convert to single value safely
        if isinstance(score, np.ndarray):

            score = np.max(score)

        # Convert to probability-like score
        confidence = 1 / (1 + np.exp(-abs(score)))

        confidence = round(float(confidence), 3)

    except:

        confidence = 0.85

    return label, confidence

# SIDEBAR
with st.sidebar:

    st.markdown("## 📊 About this model")

    st.write("**Model:** SVM")

    st.write("**Task:** Binary sentiment classification")

    st.write("**Features:** TF-IDF on cleaned review text")

    st.markdown("---")

    st.markdown("## 📖 How to read predictions")

    st.markdown("""
- **Label** → predicted sentiment

- **Confidence** → model confidence score

- Higher confidence means stronger prediction
""")

# MAIN TITLE
st.title("📝 NLP Sentiment Analysis")

st.write(
    "Binary sentiment classifier using TF-IDF and SVM model."
)

# TABS
tab1, tab2, tab3 = st.tabs(
    ["✏️ Single Review", "📂 Batch Mode", "ℹ️ About the Model"]
)

# TAB 1 : SINGLE REVIEW
with tab1:

    review = st.text_area(
        "Paste product review here:",
        placeholder="Example: Battery life is amazing and camera quality is excellent."
    )

    sample_reviews = {
        "1-star": "Worst product. Battery drains quickly and camera is poor.",
        "5-star": "Amazing mobile phone. Performance and battery life are excellent."
    }

    sample_choice = st.selectbox(
        "Try a sample",
        list(sample_reviews.keys())
    )

    if st.button("Use this sample"):

        st.session_state["sample_review"] = sample_reviews[sample_choice]

    if "sample_review" in st.session_state:

        review = st.session_state["sample_review"]

    if st.button("🔍 Analyze sentiment"):

        if review.strip() == "":

            st.warning("Please enter a review.")

        else:

            label, confidence = predict_sentiment(review)

            if str(label).lower() in ["positive", "1"]:

                st.success(f"Prediction: {label}")

            else:

                st.error(f"Prediction: {label}")

            st.metric("Confidence Score", confidence)

# TAB 2 : BATCH MODE
with tab2:

    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader(
        "Upload CSV containing review column",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data")

        st.dataframe(df.head())

        review_column = st.selectbox(
            "Select review column",
            df.columns
        )

        if st.button("Run Batch Prediction"):

            predictions = []

            confidences = []

            for text in df[review_column]:

                label, confidence = predict_sentiment(text)

                predictions.append(label)

                confidences.append(confidence)

            df["predicted_sentiment"] = predictions

            df["confidence"] = confidences

            # Metrics
            total = len(df)

            positive_count = (
                df["predicted_sentiment"]
                .astype(str)
                .str.lower()
                .isin(["positive", "1"])
                .sum()
            )

            negative_count = total - positive_count

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Reviews", total)

            col2.metric("Positive", positive_count)

            col3.metric("Negative", negative_count)

            st.write("## Prediction Results")

            st.dataframe(df)

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download full results as CSV",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

# TAB 3 : ABOUT MODEL
with tab3:

    st.subheader("Pipeline")

    st.markdown("""
1. Data cleaning

2. TF-IDF vectorization

3. SVM model training

4. Sentiment prediction
""")

    st.subheader("Why SVM?")

    st.markdown("""
- Works well for NLP text classification

- Handles high-dimensional TF-IDF features effectively

- Good performance for sentiment analysis

- Faster and efficient for smaller datasets
""")

    st.subheader("Known Limitations")

    st.markdown("""
- May struggle with sarcasm

- Depends on training data quality

- Confidence score is approximate
""")