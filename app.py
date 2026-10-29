
import streamlit as st
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

st.set_page_config(page_title="Natural Language to SQL", page_icon="🗃️")
st.title("Natural Language to SQL")
st.write("Enter a database schema and a question in plain English.")

@st.cache_resource
def load_assets():
    base_name = "google/flan-t5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    model = PeftModel.from_pretrained(base, "./sql_lora_adapter").to(device)
    ds = load_dataset("b-mc2/sql-create-context")["train"]
    sample = ds.shuffle(seed=42).select(range(min(1000, len(ds))))
    split = sample.train_test_split(test_size=0.2, seed=42)["train"]
    texts = [f"Schema: {x['context']}\nQuestion: {x['question']}" for x in split]
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    vectors = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = NearestNeighbors(n_neighbors=min(3, len(texts)), metric="cosine").fit(vectors)
    return device, tokenizer, model, split, texts, embedder, index

schema = st.text_area("Database schema", "CREATE TABLE students (id INTEGER, marks INTEGER)")
question = st.text_input("Question", "Find students who scored more than 90.")

if st.button("Generate SQL"):
    with st.spinner("Generating SQL..."):
        try:
            device, tokenizer, model, split, texts, embedder, index = load_assets()
            qvec = embedder.encode([f"Schema: {schema}\nQuestion: {question}"], convert_to_numpy=True, normalize_embeddings=True)
            distances, indices = index.kneighbors(qvec, n_neighbors=min(2, len(texts)))
            prompt = (
                "Convert the question into SQL using the given database schema. Return only the SQL query.\n"
                f"Schema: {schema}\nQuestion: {question}\n"
            )
            for rank, idx in enumerate(indices[0], 1):
                ex = split[int(idx)]
                prompt += f"Similar example {rank}: Question: {ex['question']} SQL: {ex['answer']}\n"
            prompt += "SQL:"
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                output = model.generate(**tokens, max_new_tokens=128, num_beams=2)
            sql = tokenizer.decode(output[0], skip_special_tokens=True)
            st.subheader("Generated SQL")
            st.code(sql, language="sql")
            st.caption("Always review generated SQL before running it against a real database.")
        except Exception as e:
            st.error(f"Could not load/generate. Ensure ./sql_lora_adapter is present and dependencies are installed. Details: {e}")
