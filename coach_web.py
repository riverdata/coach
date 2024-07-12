import gc
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import json

# Função para limpar a memória
def clear_memory():
    gc.collect()

# Limpar memória no início do script
clear_memory()

# Carregar o modelo para interpretação de perguntas
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Carregar o dicionário de perguntas e respostas
def load_qa_dict(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)['questions_answers']

qa_dict = load_qa_dict('coaching_qa.json')

def get_response(question, qa_dict):
    questions = [item['question'] for item in qa_dict]
    embeddings = model.encode(questions, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    best_match_idx = torch.argmax(scores).item()
    return qa_dict[best_match_idx]['answer']

# Interface Streamlit
st.title("Coach by Quanti.ca")
st.subheader("MVP v0.0.1")

user_input = st.text_input("Você:")
if user_input:
    response = get_response(user_input, qa_dict)
    st.text_area("Coach:", value=response, height=200)

