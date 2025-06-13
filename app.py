import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from nltk.tokenize import TreebankWordTokenizer

# Load models
model_sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_wmd = api.load("glove-wiki-gigaword-100")

# App title
st.title("Subjective Answer Evaluation")

# Text inputs
actual_answer = st.text_area("Enter Actual Answer:")
student_answer = st.text_area("Enter Student Answer:")

# When the user clicks the "Submit" button
if st.button("Submit"):
    if actual_answer.strip() and student_answer.strip():
        sentences = [actual_answer, student_answer]
        sentence_vecs = model_sentence_transformer.encode(sentences)
        
        similarity = cosine_similarity([sentence_vecs[0]], [sentence_vecs[1]])[0][0]
        tokenizer = TreebankWordTokenizer()
        actual_tokens = tokenizer.tokenize(actual_answer.lower())
        student_tokens = tokenizer.tokenize(student_answer.lower())
        wmd_distance = model_wmd.wmdistance(actual_tokens, student_tokens)
        st.success(f"Cosine Similarity Score: {similarity}")
        st.success(f"Word Mover Distance Score: {wmd_distance}%")
    else:
        st.warning("Please fill in both answers before submitting.")