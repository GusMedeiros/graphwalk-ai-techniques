import json
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
# Arquivos de entrada e saída
RESULTS_FILE = "collected_results.json"
AUGMENTED_RESULTS_FILE = "collected_results_augmented.json"
DATASET_NAME = "openai/graphwalks"


def main():
    """
    Script para enriquecer os resultados coletados com os dados do gabarito ("answer_nodes").
    """
    print(f"Lendo os resultados coletados de: {RESULTS_FILE}")
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            collected_results = json.load(f)
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{RESULTS_FILE}' não foi encontrado. Execute o script de coleta primeiro.")
        return
    except json.JSONDecodeError:
        print(f"ERRO: O arquivo '{RESULTS_FILE}' está corrompido ou vazio.")
        return

    if not collected_results:
        print("O arquivo de resultados está vazio. Nada a fazer.")
        return

    # --- ETAPA 1: Criar o mapa de busca (Prompt -> Answer) ---
    print(f"\nCarregando o dataset original '{DATASET_NAME}' para criar o mapa de busca...")
    # Usar streaming=True pode ser mais rápido se a memória for um problema, mas para este tamanho é ok.
    full_dataset = load_dataset(DATASET_NAME, split="train")

    print("Construindo o mapa de busca 'prompt' -> 'answer_nodes'. Isso pode levar um minuto...")
    prompt_to_answer_map = {}
    # Usamos tqdm para mostrar uma barra de progresso durante esta operação longa
    for item in tqdm(full_dataset, desc="Mapeando gabaritos"):
        prompt_to_answer_map[item['prompt']] = item['answer_nodes']

    print(f"Mapa de busca criado com {len(prompt_to_answer_map)} entradas.")

    # --- ETAPA 2: Aumentar os resultados ---
    print(f"\nEnriquecendo {len(collected_results)} resultados com os gabaritos...")
    augmented_results = []
    found_count = 0
    not_found_count = 0

    for result in tqdm(collected_results, desc="Enriquecendo resultados"):
        prompt = result.get('prompt')

        # Encontra a resposta no mapa
        answer_nodes = prompt_to_answer_map.get(prompt)

        if answer_nodes is not None:
            # Adiciona o campo do gabarito ao resultado
            result['answer_nodes'] = answer_nodes
            found_count += 1
        else:
            # Isso não deveria acontecer se os dados estiverem consistentes, mas é uma boa verificação
            result['answer_nodes'] = None  # Marca como não encontrado
            not_found_count += 1
            print(f"AVISO: O prompt a seguir não foi encontrado no mapa de busca:\n{prompt[:100]}...")

        augmented_results.append(result)

    # --- ETAPA 3: Salvar o novo arquivo ---
    print("\nSalvando os resultados enriquecidos em:", AUGMENTED_RESULTS_FILE)
    with open(AUGMENTED_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(augmented_results, f, indent=4)

    print("\n--- Processo Concluído ---")
    print(f"Total de resultados processados: {len(augmented_results)}")
    print(f"Gabaritos encontrados e adicionados: {found_count}")
    if not_found_count > 0:
        print(f"AVISO: {not_found_count} prompts não foram encontrados no dataset original.")


if __name__ == "__main__":
    main()