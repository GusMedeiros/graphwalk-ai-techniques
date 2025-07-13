import json
from tqdm import tqdm
import os


def calculate_metrics(predicted_nodes, true_nodes):
    """
    Calcula precisão, recall e F1-score para um único exemplo.

    Args:
        predicted_nodes (list): A lista de nós retornada pelo seu script.
        true_nodes (list): A lista de nós do gabarito.

    Returns:
        dict: Um dicionário contendo 'precision', 'recall' e 'f1_score'.
    """
    # Lida com casos onde o output pode ser uma string de erro em vez de uma lista.
    if not isinstance(predicted_nodes, list):
        predicted_nodes = []

    # Usar conjuntos (sets) é a maneira mais eficiente de comparar as listas.
    predicted_set = set(predicted_nodes)
    true_set = set(true_nodes)

    # Casos de borda onde um ou ambos os conjuntos estão vazios.
    if not predicted_set and not true_set:
        # Previsão correta de um conjunto vazio. Perfeito.
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}

    true_positives = len(predicted_set.intersection(true_set))

    # Cálculo da Precisão
    if len(predicted_set) == 0:
        precision = 0.0
    else:
        precision = true_positives / len(predicted_set)

    # Cálculo do Recall
    if len(true_set) == 0:
        # Se não havia nada a ser encontrado, e não encontramos nada, o recall é perfeito.
        recall = 1.0 if len(predicted_set) == 0 else 0.0
    else:
        recall = true_positives / len(true_set)

    # Cálculo do F1-Score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def main():
    """
    Script principal para ler um arquivo de resultados, calcular métricas e
    escrever um novo arquivo com as métricas adicionadas.
    """
    # Nomes de arquivo fixos
    input_filename = "graphwalks_results.jsonl"
    output_filename = "graphwalks_results_with_metrics.jsonl"
    general_metrics_filename = "graphwalks_general_metrics.json" # Novo arquivo para métricas gerais

    # Verifica se o arquivo de entrada existe
    if not os.path.exists(input_filename):
        print(f"Erro: O arquivo de entrada '{input_filename}' não foi encontrado.")
        print("Certifique-se de executar o script de processamento primeiro.")
        return

    print(f"Lendo resultados de: '{input_filename}'")
    print(f"Salvando resultados com métricas individuais em: '{output_filename}'")
    print(f"Salvando métricas gerais em: '{general_metrics_filename}'")

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    item_count = 0

    try:
        # Conta o total de linhas para a barra de progresso
        num_lines = sum(1 for line in open(input_filename, 'r', encoding='utf-8'))

        with open(input_filename, 'r', encoding='utf-8') as infile, \
                open(output_filename, 'w', encoding='utf-8') as outfile:

            for line in tqdm(infile, total=num_lines, desc="Calculando métricas"):
                data = json.loads(line)

                predicted = data.get("output", [])
                true_answers = data.get("answer_nodes", [])

                # Calcula as métricas para a linha atual
                metrics = calculate_metrics(predicted, true_answers)

                # Adiciona as métricas ao dicionário original
                data.update(metrics)

                # Escreve o dicionário atualizado no arquivo de saída
                outfile.write(json.dumps(data) + '\n')

                # Acumula os totais para a média final
                total_precision += metrics['precision']
                total_recall += metrics['recall']
                total_f1 += metrics['f1_score']
                item_count += 1

    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON no arquivo de entrada: {e}")
        return

    print("\nProcessamento concluído.")

    # Exibe e salva as médias gerais
    if item_count > 0:
        avg_precision = total_precision / item_count
        avg_recall = total_recall / item_count
        avg_f1 = total_f1 / item_count

        general_metrics = {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1_score": avg_f1,
            "total_items_processed": item_count
        }

        print("\n--- Métricas Médias Gerais ---")
        print(f"Precisão Média: {avg_precision:.4f}")
        print(f"Recall Médio:   {avg_recall:.4f}")
        print(f"F1-Score Médio: {avg_f1:.4f}")
        print("----------------------------")

        # Salva as métricas gerais em um arquivo JSON
        try:
            with open(general_metrics_filename, 'w', encoding='utf-8') as gm_file:
                json.dump(general_metrics, gm_file, indent=4)
            print(f"Métricas gerais salvas em '{general_metrics_filename}'")
        except IOError as e:
            print(f"Erro ao salvar métricas gerais em '{general_metrics_filename}': {e}")
    else:
        print("Nenhum item foi processado para calcular as métricas médias.")


main()