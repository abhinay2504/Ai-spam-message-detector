"""Streamlit user interface for the AI Spam Message Detector."""
import streamlit as st
from model import train_model, predict_message

st.set_page_config(page_title="AI Spam Detector", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    return train_model()

model = load_model()

st.title("🛡️ AI Spam Message Detector")
st.caption("Check whether a message looks like spam using a Machine Learning model.")
st.info("Enter a message below. The model uses TF-IDF text features and Logistic Regression to make a prediction.")

examples = {
    "Choose an example...": "",
    "Possible spam message": "Congratulations! You have won a free gift card. Claim now!",
    "Normal message": "Hey, are we still meeting for lunch tomorrow?",
}
choice = st.selectbox("Try a sample message", list(examples.keys()))
message = st.text_area("Enter your message", value=examples[choice], height=130,
                       placeholder="Type or paste a message here...")

if st.button("Analyze message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message before analyzing it.")
    else:
        result = predict_message(model, message)
        if result["label"] == "spam":
            st.error("🚨 This message is predicted to be SPAM.")
        else:
            st.success("✅ This message is predicted to be NOT SPAM.")
        st.metric("Model confidence", f"{result['confidence'] * 100:.1f}%")
        st.caption("Confidence is the model's estimated probability, not a guarantee. Always use your judgment.")

with st.expander("How does this project work?"):
    st.markdown("""
    1. **Training data:** Example spam and normal messages are stored in `data/messages.csv`.
    2. **TF-IDF:** Text is converted into numerical features based on word importance.
    3. **Logistic Regression:** A supervised Machine Learning model learns patterns from labeled examples.
    4. **Prediction:** The model classifies a new message as `spam` or `ham` (not spam).

    This is an educational prototype trained on a small sample dataset. It is not a production-grade spam filter.
    """)

st.divider()
st.caption("Built with Python, Streamlit and scikit-learn • Educational project")