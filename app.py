import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib

# 🔧 Set page config as the FIRST command
st.set_page_config(page_title="News Tag Generator", layout="centered")

# Load the model, tokenizer, and label binarizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert_model")
    tokenizer = BertTokenizer.from_pretrained("bert_model")
    mlb = joblib.load("bert_label_binarizer.pkl")
    return model, tokenizer, mlb

model, tokenizer, mlb = load_model()

# Streamlit UI setup
st.title("📰 Financial News Tag Generator")
st.markdown("Enter a financial news headline and get predicted tags using a fine-tuned BERT model.")

# Text input
headline = st.text_area("Enter Headline", height=100)

# Predict button
if st.button("Generate Tags"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    else:
        with st.spinner("Generating tags..."):
            # Preprocess and predict
            inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits).numpy()
                preds = (probs > 0.5).astype(int)
                tags = mlb.inverse_transform(preds)

            # Display result
            st.success("Predicted Tags:")
            if tags and tags[0]:
                for tag in tags[0]:
                    st.markdown(f"- **{tag}**")
            else:
                st.info("No tags were confidently predicted.")
