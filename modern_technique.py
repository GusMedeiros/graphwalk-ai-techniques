import os
import json
import random
import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import time

# --- 1. CONFIGURAÇÃO ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME = "openai/graphwalks"
TARGET_SAMPLES_PER_TYPE = 100
MAX_NEW_TOKENS = 512

STATE_FILE = "processing_state.json"
RESULTS_FILE = "collected_results.json"
INDICES_PARENTS_FILE = "indices_parents.json"
INDICES_BFS_FILE = "indices_bfs.json"


# --- 2. FUNÇÕES DE ESTADO E INICIALIZAÇÃO ---
def initialize_state_and_indices():
    print("Verificando estado e arquivos de índice...")
    if os.path.exists(INDICES_PARENTS_FILE) and os.path.exists(STATE_FILE):
        print("Arquivos de estado encontrados. O script será retomado.")
        return

    print("Nenhum estado anterior encontrado. Iniciando do zero.")
    print("Carregando dataset (pode levar um momento)...")
    dataset = load_dataset(DATASET_NAME, split="train")

    print("Separando e embaralhando índices por 'problem_type'...")
    parents_indices = [i for i, item in enumerate(dataset) if item['problem_type'] == 'parents']
    bfs_indices = [i for i, item in enumerate(dataset) if item['problem_type'] == 'bfs']

    random.shuffle(parents_indices)
    random.shuffle(bfs_indices)

    with open(INDICES_PARENTS_FILE, 'w') as f:
        json.dump(parents_indices, f)
    with open(INDICES_BFS_FILE, 'w') as f:
        json.dump(bfs_indices, f)
    print(f"Salvos {len(parents_indices)} índices de 'parents' em {INDICES_PARENTS_FILE}")
    print(f"Salvos {len(bfs_indices)} índices de 'bfs' em {INDICES_BFS_FILE}")

    initial_state = {'pointers': {'parents': 0, 'bfs': 0}}
    with open(STATE_FILE, 'w') as f:
        json.dump(initial_state, f, indent=4)

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            json.dump([], f, indent=4)

    print("Inicialização completa.")


def load_state():
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    with open(INDICES_PARENTS_FILE, 'r') as f:
        indices_parents = json.load(f)
    with open(INDICES_BFS_FILE, 'r') as f:
        indices_bfs = json.load(f)
    return state, results, indices_parents, indices_bfs


def save_state(state, results):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)


# --- 3. FUNÇÃO DE PROCESSAMENTO PRINCIPAL ---
def process_item(pipe, tokenizer, context_window, dataset, index, problem_type):
    prompt_text = dataset[int(index)]['prompt']
    messages = [{"role": "user", "content": prompt_text}]

    try:
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_token_count = len(input_ids)

        if prompt_token_count > context_window:
            print(
                f"    -> PRÉ-VERIFICAÇÃO FALHOU: Prompt tem {prompt_token_count} tokens, excede a janela de {context_window}. Pulando.")
            return None
    except Exception as e:
        print(f"    -> PRÉ-VERIFICAÇÃO FALHOU: Erro inesperado durante a tokenização: {e}")
        return None

    print(f"    -> PRÉ-VERIFICAÇÃO OK: Prompt com {prompt_token_count} tokens. Apostando na execução...")
    try:
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipe(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        generated_text = outputs[0]['generated_text'][-1]['content']

        return {
            "problem_type": problem_type,
            "prompt": prompt_text,
            "output": generated_text
        }
    except Exception as e:
        print(f"    -> EXECUÇÃO FALHOU: A 'aposta' não deu certo. Erro durante a inferência: {e}")
        return None


# --- 4. ORQUESTRADOR / LOOP PRINCIPAL ---
def main():
    initialize_state_and_indices()
    state, results, indices_parents, indices_bfs = load_state()

    print("\n--- Configurando modelo para 8GB VRAM (Quantização 4-bit + SDPA) ---")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    print("--- Carregando modelo e pipeline (isso pode demorar um pouco)... ---")
    pipe = pipeline("text-generation", model=MODEL_ID,
                    model_kwargs={"quantization_config": quantization_config, "attn_implementation": "sdpa"},
                    device_map="auto")

    tokenizer = pipe.tokenizer
    context_window = pipe.model.config.max_position_embeddings
    print(f"Modelo carregado. Janela de contexto: {context_window} tokens. Iniciando a coleta...\n")

    full_dataset = load_dataset(DATASET_NAME, split="train")

    while True:
        counts = {'parents': 0, 'bfs': 0}
        for res in results:
            counts[res['problem_type']] += 1

        print(
            f"Progresso: Parents [{counts['parents']}/{TARGET_SAMPLES_PER_TYPE}] | BFS [{counts['bfs']}/{TARGET_SAMPLES_PER_TYPE}]")

        if counts['parents'] >= TARGET_SAMPLES_PER_TYPE and counts['bfs'] >= TARGET_SAMPLES_PER_TYPE:
            print("\nMeta de coleta atingida para ambos os tipos!")
            break

        ptr_parents = state['pointers']['parents']
        ptr_bfs = state['pointers']['bfs']

        process_parents = counts['parents'] < TARGET_SAMPLES_PER_TYPE
        process_bfs = counts['bfs'] < TARGET_SAMPLES_PER_TYPE

        if not process_parents and not process_bfs: break

        if process_parents and (not process_bfs or ptr_parents <= ptr_bfs):
            current_type = 'parents'
            indices_list = indices_parents
        elif process_bfs:
            current_type = 'bfs'
            indices_list = indices_bfs
        else:
            break

        current_pointer = state['pointers'][current_type]

        if current_pointer >= len(indices_list):
            print(f"AVISO: Esgotados todos os exemplos para o tipo '{current_type}' antes de atingir a meta.")
            if current_type == 'parents': state['pointers']['parents'] = float('inf')
            if current_type == 'bfs': state['pointers']['bfs'] = float('inf')
            continue

        dataset_index_to_process = indices_list[current_pointer]
        print(
            f"\nProcessando tipo '{current_type}', ponteiro {current_pointer}, índice do dataset {dataset_index_to_process}...")

        result = process_item(pipe, tokenizer, context_window, full_dataset, dataset_index_to_process, current_type)

        state['pointers'][current_type] += 1

        if result:
            print("    -> SUCESSO: Amostra coletada.")
            results.append(result)

        save_state(state, results)

    print("\n--- Coleta Finalizada ---")
    print(f"Total de amostras salvas em {RESULTS_FILE}: {len(results)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcesso interrompido pelo usuário. O estado foi salvo e pode ser retomado.")