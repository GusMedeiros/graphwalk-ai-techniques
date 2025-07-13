import json
from tqdm import tqdm
import os


def calculate_metrics(predicted_nodes, true_nodes):
    """
    Calcula precisão, recall e F1-score para um único exemplo.
    (Esta função permanece inalterada)
    """
    if not isinstance(predicted_nodes, list):
        predicted_nodes = []

    predicted_set = set(predicted_nodes)
    true_set = set(true_nodes)

    if not predicted_set and not true_set:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}

    true_positives = len(predicted_set.intersection(true_set))

    if len(predicted_set) == 0:
        precision = 0.0
    else:
        precision = true_positives / len(predicted_set)

    if len(true_set) == 0:
        recall = 1.0 if len(predicted_set) == 0 else 0.0
    else:
        recall = true_positives / len(true_set)

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
    Script principal para ler um arquivo de resultados, calcular métricas por
    estrato e escrever um novo arquivo com as métricas individuais e um
    resumo geral.
    """
    input_filename = "graphwalks_results.jsonl"
    output_filename = "graphwalks_results_with_metrics.jsonl"
    general_metrics_filename = "graphwalks_general_metrics.json"

    # --- ALTERAÇÕES PRINCIPAIS AQUI ---
    # Chave correta para identificar o estrato.
    STRATUM_KEY = "problem_type"
    # Dicionário para acumular totais, com valores em minúsculas.
    totals = {
        'bfs': {'precision': 0, 'recall': 0, 'f1': 0, 'count': 0},
        'parents': {'precision': 0, 'recall': 0, 'f1': 0, 'count': 0}
    }
    # --- FIM DAS ALTERAÇÕES PRINCIPAIS ---

    if not os.path.exists(input_filename):
        print(f"Erro: O arquivo de entrada '{input_filename}' não foi encontrado.")
        print("Certifique-se de executar o script de processamento primeiro.")
        return

    print(f"Lendo resultados de: '{input_filename}'")
    print(f"Salvando resultados com métricas individuais em: '{output_filename}'")
    print(f"Salvando métricas gerais em: '{general_metrics_filename}'")

    total_items = 0

    try:
        num_lines = sum(1 for line in open(input_filename, 'r', encoding='utf-8'))

        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile:

            for line in tqdm(infile, total=num_lines, desc="Calculando métricas"):
                data = json.loads(line)
                predicted = data.get("output", [])
                true_answers = data.get("answer_nodes", [])

                metrics = calculate_metrics(predicted, true_answers)
                data.update(metrics)
                outfile.write(json.dumps(data) + '\n')

                # Acumula as métricas para o estrato correspondente
                stratum = data.get(STRATUM_KEY)
                if stratum in totals:
                    totals[stratum]['precision'] += metrics['precision']
                    totals[stratum]['recall'] += metrics['recall']
                    totals[stratum]['f1'] += metrics['f1_score']
                    totals[stratum]['count'] += 1

                total_items += 1

    except (json.JSONDecodeError, IOError) as e:
        print(f"Erro ao processar o arquivo de entrada: {e}")
        return

    print("\nProcessamento concluído.")

    if total_items > 0:
        accuracy_metrics = {}
        overall_totals = {'precision': 0, 'recall': 0, 'f1': 0, 'count': 0}

        # Calcula as médias para cada estrato
        for stratum, values in totals.items():
            count = values['count']
            if count > 0:
                accuracy_metrics[stratum.upper()] = { # Salva com chave em maiúscula para consistência
                    "average_precision": values['precision'] / count,
                    "average_recall": values['recall'] / count,
                    "average_f1_score": values['f1'] / count,
                    "total_items_processed": count
                }
                overall_totals['precision'] += values['precision']
                overall_totals['recall'] += values['recall']
                overall_totals['f1'] += values['f1']
                overall_totals['count'] += count
            else:
                accuracy_metrics[stratum.upper()] = {
                    "average_precision": 0.0,
                    "average_recall": 0.0,
                    "average_f1_score": 0.0,
                    "total_items_processed": 0
                }

        # Calcula a média geral
        if overall_totals['count'] > 0:
             accuracy_metrics['GERAL'] = {
                "average_precision": overall_totals['precision'] / overall_totals['count'],
                "average_recall": overall_totals['recall'] / overall_totals['count'],
                "average_f1_score": overall_totals['f1'] / overall_totals['count'],
                "total_items_processed": overall_totals['count']
             }

        general_metrics = {
            "accuracy_metrics": accuracy_metrics,
            "execution_time_metrics": {
                "GERAL": {"number_of_examples": 1150, "total_execution_time_s": 10.5203, "average_execution_time_s": 0.009148},
                "BFS": {"number_of_examples": 550, "total_execution_time_s": 5.3770, "average_execution_time_s": 0.009776},
                "PARENTS": {"number_of_examples": 600, "total_execution_time_s": 5.1433, "average_execution_time_s": 0.008572}
            }
        }

        print("\n--- Métricas Médias de Acurácia ---")
        display_order = ['BFS', 'PARENTS', 'GERAL']
        for category in display_order:
            if category in general_metrics['accuracy_metrics']:
                metrics = general_metrics['accuracy_metrics'][category]
                print(f"\nCategoria: {category} ({metrics['total_items_processed']} itens)")
                print(f"  - Precisão Média: {metrics['average_precision']:.4f}")
                print(f"  - Recall Médio:   {metrics['average_recall']:.4f}")
                print(f"  - F1-Score Médio: {metrics['average_f1_score']:.4f}")
        print("-----------------------------------")

        print("\n--- Métricas de Tempo de Execução ---")
        for category, metrics in general_metrics['execution_time_metrics'].items():
            print(f"\nCategoria: {category}")
            print(f"  - Número de exemplos:     {metrics['number_of_examples']}")
            print(f"  - Tempo de Execução Total: {metrics['total_execution_time_s']:.4f}s")
            print(f"  - Tempo de Execução Médio: {metrics['average_execution_time_s']:.6f}s")
        print("-----------------------------------")

        try:
            with open(general_metrics_filename, 'w', encoding='utf-8') as gm_file:
                json.dump(general_metrics, gm_file, indent=4)
            print(f"\nMétricas gerais salvas em '{general_metrics_filename}'")
        except IOError as e:
            print(f"Erro ao salvar métricas gerais em '{general_metrics_filename}': {e}")
    else:
        print("Nenhum item foi processado para calcular as métricas médias.")


if __name__ == '__main__':
    main()