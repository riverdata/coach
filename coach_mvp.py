#pip install transformers datasets language-tool-python peft nltk safetensors python-docx

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Função para limpar a memória
def clear_memory():
    gc.collect()

# Limpar memória no início do script
clear_memory()

import docx
import time
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import language_tool_python
import nltk
from peft import get_peft_model, LoraConfig, TaskType
import os
from safetensors.torch import load_file as safetensors_load

# Baixar o recurso 'punkt' da NLTK
nltk.download('punkt')

def docx_to_txt(docx_file, txt_file):
    doc = docx.Document(docx_file)
    with open(txt_file, 'w', encoding='utf-8') as f:
        for para in doc.paragraphs:
            f.write(para.text + '\n')
    assert os.path.exists(txt_file), f"Arquivo {txt_file} não foi criado."
    print(f"Arquivo {txt_file} criado com sucesso.")

def preprocess_text(text):
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    return text

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            self.start_time = time.time()

        progress = state.global_step / state.max_steps
        print(f"Treinamento em progresso: Step {state.global_step}/{state.max_steps} ({progress:.2%})")

        if state.global_step > 0:
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time / state.global_step * state.max_steps
            remaining_time = estimated_total_time - elapsed_time
            remaining_time_formatted = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
            print(f"Tempo estimado restante: {remaining_time_formatted}")

def load_trained_model():
    try:
        # Carregar o tokenizador primeiro
        tokenizer = AutoTokenizer.from_pretrained("coach_model")

        # Carregar o modelo e redimensionar os embeddings
        model = AutoModelForCausalLM.from_pretrained("botbot-ai/CabraLlama3-8b")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

        # Carregar o estado salvo do modelo utilizando safetensors
        state_dict = safetensors_load("coach_model/adapter_model.safetensors")
        model.load_state_dict(state_dict, strict=False)

        return model, tokenizer
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        raise e

def clean_text(text):
    text = " ".join(text.split())
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return text

def correct_text(text):
    tool = language_tool_python.LanguageTool('pt-BR')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def split_into_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text, language='portuguese')

def combine_sentences(sentences):
    return " ".join(sentences)

def is_valid_response(response):
    return response.endswith('.') and len(response.split()) > 5

def train_cabra_llama_model(training_file, progress_callback):
    try:
        print("Iniciando o treinamento do modelo...")
        with open(training_file, 'r', encoding='utf-8') as file:
            text = file.read()
        preprocessed_text = preprocess_text(text)
        print("Texto pré-processado para treinamento.")

        dataset = Dataset.from_dict({'text': [preprocessed_text]})

        tokenizer = AutoTokenizer.from_pretrained("botbot-ai/CabraLlama3-8b")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Token especial de preenchimento adicionado.")

        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        print("Dataset tokenizado.")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        print("Data collator criado.")

        model = AutoModelForCausalLM.from_pretrained("botbot-ai/CabraLlama3-8b")
        model.resize_token_embeddings(len(tokenizer))  # Redimensionar modelo para novo token
        print("Modelo carregado e redimensionado.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir="./coach_model",
            overwrite_output_dir=True,
            num_train_epochs=100,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=50,
            evaluation_strategy="no",
            learning_rate=5e-5,
            warmup_steps=500
        )
        print("Argumentos de treinamento configurados.")

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets,
            callbacks=[progress_callback]
        )

        print("Iniciando o treinamento do modelo...")
        trainer.train()
        print("Treinamento concluído.")

        model.save_pretrained('./coach_model')
        tokenizer.save_pretrained('./coach_model')

        print("Treinamento concluído! O modelo e o tokenizador foram salvos como 'coach_model'.")

    except Exception as e:
        print(f"Erro durante o treinamento do modelo: {e}")
        raise e

def main():
    print("Coach Terminal Interface")

    if os.path.exists('./coach_model'):
        print("Verificando se já foi treinado")
        print("Treinamento já realizado anteriormente, iniciando carregamento do modelo...")
        model, tokenizer = load_trained_model()
    else:
        from google.colab import files
        uploaded = files.upload()

        for file_name in uploaded.keys():
            if file_name.endswith('.docx'):
                docx_file = file_name
                print("Arquivo carregado com sucesso!")

                txt_file = "training_coach.txt"
                docx_to_txt(docx_file, txt_file)

                train_cabra_llama_model(txt_file, ProgressCallback())

                model, tokenizer = load_trained_model()

    while True:
        user_input = input("Você: ")

        if user_input.lower() in ["sair", "exit"]:
            break

        if user_input:
            user_input = correct_text(user_input)
            sentences = split_into_sentences(user_input)
            responses = []

            for sentence in sentences:
                input_ids = tokenizer.encode(sentence, return_tensors='pt')
                output = model.generate(
                    input_ids,
                    max_length=200,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    top_k=20,
                    top_p=0.85,
                    temperature=0.7,
                    early_stopping=True
                )
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                response = clean_text(response)
                response = correct_text(response)
                responses.append(response)

            final_response = combine_sentences(responses)
            print(f"Coach: {final_response}")
        else:
            print("Por favor, digite uma mensagem.")

if __name__ == "__main__":
    main()
