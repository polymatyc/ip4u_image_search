import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('clip-ViT-B-32', device=device)
    return model

model = load_model()

# -----------------------------
# LOAD DATA (Embeddings + Metadata)
# -----------------------------
@st.cache_data
def load_data():
    embeddings = np.load("database/embeddings.npy")
    df = pd.read_csv("database/metadata.csv")
    return embeddings, df

embeddings, df = load_data()

# -----------------------------
# APP UI
# -----------------------------
st.title("Industrial AI Image & Text Search")

search_type = st.radio("Search Type:", ["Text Search", "Image Search"])

# -----------------------------
# TEXT SEARCH
# -----------------------------
if search_type == "Text Search":

    query = st.text_input("Enter product description")

    if st.button("Search") and query:

        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-5:][::-1]

        st.subheader("Top Matches")

        for idx in top_indices:

            product = df.iloc[idx]

            st.image(product["image_path"], width=250)

            st.markdown(f"### {product['title']}")
            st.write("**Vendor:**", product["vendor"])
            st.write("**Price:** ₹", product["price"])

            if "specs" in df.columns:
                st.write("**Specifications:**", product["specs"])

            st.write("**Similarity Score:**", round(similarities[idx], 4))
            st.markdown("---")

# -----------------------------
# IMAGE SEARCH
# -----------------------------
else:

    uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png", "jpeg"])

    if uploaded_file and st.button("Search"):

        image = Image.open(uploaded_file).convert("RGB")
        query_embedding = model.encode(image)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = similarities.argsort()[-5:][::-1]

        st.subheader("Top Matches")

        for idx in top_indices:

            product = df.iloc[idx]

            st.image(product["image_path"], width=250)

            st.markdown(f"### {product['title']}")
            st.write("**Vendor:**", product["vendor"])
            st.write("**Price:** ₹", product["price"])

            if "specs" in df.columns:
                st.write("**Specifications:**", product["specs"])

            st.write("**Similarity Score:**", round(similarities[idx], 4))

            st.markdown("---")
