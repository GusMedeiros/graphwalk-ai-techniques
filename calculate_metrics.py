import json
import re
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
# Use o nome do arquivo gerado pelo script anterior
INPUT_JSON_FILE = 'collected_results_augmented.json'
OUTPUT_JSON_FILE = f'{INPUT_JSON_FILE.replace(".json", "")}_with_metrics.json'

def extract_prediction_from_prompt(prompt_text):
    """
    Extrai a lista de nós da chave 'prediction' dentro do texto do prompt.
    Usa regex para uma extração robusta.
    """
    # Regex para encontrar "prediction": seguido por uma lista [...]
    match = re.search(r'"prediction":\s*(\[.*?\])', prompt_text)
    if not match:
        return [] # Retorna lista vazia se não encontrar a predição

    prediction_str = match.group(1)
    try:
        # Converte a string da lista em uma lista Python
        prediction_list = json.loads(prediction_str)
        return prediction_list
    except json.JSONDecodeError:
        # Caso a string encontrada não seja um JSON válido
        return []

def calculate_scores(true_nodes, predicted_nodes):
    """
    Calcula precisão, recall e F1-score a partir de duas listas de nós.
    """
    true_set = set(true_nodes)
    predicted_set = set(predicted_nodes)

    if not true_set and not predicted_set:
        # Caso especial: ambos vazios. Perfeito, mas sem TP/FP/FN.
        return 1.0, 1.0, 1.0

    true_positives = len(true_set.intersection(predicted_set))
    false_positives = len(predicted_set - true_set)
    false_negatives = len(true_set - predicted_set)

    # Cálculo da Precisão (evita divisão por zero)
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0 # Se não previu nada, a precisão é 0

    # Cálculo do Recall (evita divisão por zero)
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        # Isso só aconteceria se o conjunto verdadeiro fosse vazio.
        # Se previu algo, recall é 0. Se não previu nada, é 1.0 (já tratado no início).
        recall = 0.0

    # Cálculo do F1-Score (evita divisão por zero)
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score

# --- SCRIPT PRINCIPAL ---
print(f"Lendo o arquivo de entrada: {INPUT_JSON_FILE}")
with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

enriched_data = []
print(f"Calculando métricas para {len(data)} exemplos...")

for item in tqdm(data, desc="Processando itens"):
    # Extrai os nós verdadeiros e previstos
    true_nodes = item.get('answer_nodes', [])
    predicted_nodes = extract_prediction_from_prompt(item.get('prompt', ''))

    # Calcula as métricas
    precision, recall, f1_score = calculate_scores(true_nodes, predicted_nodes)

    # Adiciona as novas métricas ao item
    item['precision'] = precision
    item['recall'] = recall
    item['f1_score'] = f1_score
    enriched_data.append(item)

print(f"\nSalvando dados enriquecidos com métricas em: {OUTPUT_JSON_FILE}")
with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
    json.dump(enriched_data, f, ensure_ascii=False, indent=4)

print("\nScript de cálculo de métricas concluído com sucesso!")