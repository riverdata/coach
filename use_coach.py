import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util

# Caminho do diretório onde o modelo e o tokenizador foram salvos
model_dir = "C:/Users/claud/Downloads/coach_mvp/results/checkpoint-615"

# Carregar o tokenizador salvo
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Carregar o modelo salvo
model = AutoModelForSequenceClassification.from_pretrained(model_dir, from_tf=False, use_safetensors=True)
model.eval()

# Criar um pipeline de classificação
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Carregar o dicionário de perguntas e respostas de um arquivo JSON
with open('C:/Users/claud/Downloads/coach_mvp/results/checkpoint-615/questions_answers.json', 'r', encoding='utf-8') as f:
    questions_answers = json.load(f)

# Carregar o modelo de embeddings para similaridade de texto
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Pré-calcular as embeddings das perguntas no dicionário
questions = list(questions_answers.keys())
question_embeddings = embedder.encode(questions, convert_to_tensor=True)

# Carregar o modelo de geração de texto (GPT-2)
gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_response(prompt):
    # Gerar uma resposta baseada no prompt fornecido
    input_ids = gen_tokenizer.encode(prompt, return_tensors='pt')
    gen_output = gen_model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=gen_tokenizer.eos_token_id
    )
    generated_text = gen_tokenizer.decode(gen_output[0], skip_special_tokens=True)
    return generated_text

def get_response(user_input):
    # Classificar a intenção da pergunta do usuário usando o modelo treinado
    classification_result = classifier(user_input)
    label = classification_result[0]['label']

    # Obter a embedding da pergunta do usuário
    user_embedding = embedder.encode(user_input, convert_to_tensor=True)
    
    # Calcular a similaridade de cosseno entre a pergunta do usuário e as perguntas no dicionário
    cos_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    
    # Encontrar a pergunta mais similar no dicionário
    best_match_idx = torch.argmax(cos_scores).item()
    best_match_question = questions[best_match_idx]
    
    # Obter a resposta base do dicionário
    base_response = questions_answers.get(best_match_question, "Desculpe, não consegui entender sua resposta. Pode reformular?")
    
    # Criar um prompt para o modelo generativo
    prompt = f"Pergunta: {user_input}\nResposta base: {base_response}\nResposta natural:"

    # Gerar uma resposta mais natural baseada no prompt
    generated_response = generate_response(prompt)
    return generated_response

# Loop para interação contínua com o usuário
while True:
    # Capturar a pergunta do usuário
    user_input = input("Você: ")
    
    # Permitir que o usuário saia do loop
    if user_input.lower() in ["sair", "exit", "quit"]:
        print("Encerrando a conversa. Até logo!")
        break

    # Obter e imprimir a resposta
    response = get_response(user_input)
    print(f"MentorIA: {response}")
