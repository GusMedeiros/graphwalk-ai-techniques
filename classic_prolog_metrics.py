import json


def calculate_metrics(predicted_nodes: list | None, actual_nodes: list) -> dict:
    """
    Calcula precision, recall e f1-score para um conjunto de nós previstos e reais.

    Args:
        predicted_nodes: Lista de nós retornados pelo script. Pode ser None.
        actual_nodes: Lista de nós esperados (gabarito).

    Returns:
        Um dicionário contendo as métricas calculadas.
    """
    # Trata o caso de o resultado ser None, considerando como uma lista vazia
    if predicted_nodes is None:
        predicted_nodes = []

    # Usar sets para facilitar o cálculo de interseção e diferenças
    predicted_set = set(predicted_nodes)
    actual_set = set(actual_nodes)

    # Caso especial: se ambos os conjuntos estiverem vazios, é uma previsão perfeita.
    if not predicted_set and not actual_set:
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}

    # Calcula os componentes da matriz de confusão
    true_positives = len(predicted_set.intersection(actual_set))
    false_positives = len(predicted_set.difference(actual_set))
    false_negatives = len(actual_set.difference(predicted_set))

    # Calcula as métricas, tratando a divisão por zero
    # Precision = TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


if __name__ == "__main__":
    input_filename = "classic_prolog_results.json"
    output_filename = "classic_prolog_results_with_metrics.json"

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada '{input_filename}' não foi encontrado.")
        print("Por favor, execute o script principal primeiro para gerar este arquivo.")
        exit(1)

    results_with_metrics = []

    for entry in all_results:
        # Pega as listas de nós para comparação
        predicted = entry.get('prolog_result')
        actual = entry.get('expected_answer')

        # Calcula as métricas para a entrada atual
        metrics = calculate_metrics(predicted, actual)

        # Cria uma nova entrada de dicionário que inclui os dados originais e as novas métricas
        # O operador ** é uma forma elegante de mesclar dicionários
        updated_entry = {**entry, **metrics}

        results_with_metrics.append(updated_entry)

    # Salva a lista completa com as métricas no novo arquivo JSON
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_with_metrics, f, indent=4, ensure_ascii=False)

    print(f"Análise concluída com sucesso!")
    print(f"Resultados com métricas de desempenho foram salvos em '{output_filename}'.")