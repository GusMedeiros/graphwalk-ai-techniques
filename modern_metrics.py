import json
import re
from typing import List, Tuple, Dict, Any


def parse_output(output_str: str) -> Tuple[List[str], bool]:
    """
    Extrai a lista de nós da string de saída do modelo.

    Retorna uma tupla contendo:
    1. A lista de nós encontrados (List[str]).
    2. Um booleano indicando se o formato 'Final Answer: [...]' foi encontrado (bool).
    """
    # Procura pelo padrão 'Final Answer:' ignorando maiúsculas/minúsculas e espaços
    match = re.search(r"Final Answer:\s*\[(.*?)\]", output_str, re.IGNORECASE | re.DOTALL)

    if not match:
        # O padrão não foi encontrado, retorna lista vazia e False
        return [], False

    content = match.group(1).strip()

    if not content:
        # O padrão foi encontrado, mas a lista estava vazia ("[]")
        return [], True

    # Remove aspas e espaços extras de cada nó
    nodes = [node.strip().strip("'\"") for node in content.split(',')]
    # O padrão foi encontrado e continha nós
    return nodes, True


def calculate_metrics(predicted_nodes: List[str], true_nodes: List[str]) -> Tuple[float, float, float]:
    """
    Calcula precision, recall e f1-score comparando as duas listas de nós.
    """
    set_predicted = set(predicted_nodes)
    set_true = set(true_nodes)

    if not set_predicted and not set_true:
        return 1.0, 1.0, 1.0

    true_positives = len(set_predicted.intersection(set_true))

    precision = true_positives / len(set_predicted) if len(set_predicted) > 0 else 0.0

    if len(set_true) == 0:
        recall = 1.0 if not set_predicted else 0.0
    else:
        recall = true_positives / len(set_true)

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def _calculate_average_metrics(metrics_list: List[Tuple[float, float, float]], total_items: int) -> Dict[str, Any]:
    """Função auxiliar para calcular e formatar as métricas médias."""
    if not metrics_list:
        return {
            "evaluated_items": 0,
            "average_precision": 0.0,
            "average_recall": 0.0,
            "average_f1_score": 0.0
        }

    num_items = len(metrics_list)
    avg_precision = sum(m[0] for m in metrics_list) / num_items
    avg_recall = sum(m[1] for m in metrics_list) / num_items
    avg_f1 = sum(m[2] for m in metrics_list) / num_items

    return {
        "evaluated_items": num_items,
        "average_precision": round(avg_precision, 4),
        "average_recall": round(avg_recall, 4),
        "average_f1_score": round(avg_f1, 4)
    }


def main(input_file_path: str):
    """
    Lê um arquivo JSON, calcula métricas para cada item, descarta falhas de parsing,
    e salva os resultados em dois novos arquivos, com métricas gerais segregadas.
    """
    output_file_path = input_file_path.replace(".json", "_with_metrics.json")
    general_metrics_file_path = "modern_general_metrics.json"

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada '{input_file_path}' não foi encontrado.")
        return
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{input_file_path}' não é um JSON válido.")
        return

    # Lista para armazenar apenas os itens processados com sucesso
    processed_data = []
    # Dicionário para armazenar as métricas por tipo de problema
    metrics_by_type = {
        "bfs": [],
        "parents": []
    }

    total_items_initial = len(data)
    parsing_failures = 0

    print(f"Processando {total_items_initial} itens do arquivo '{input_file_path}'...")

    for item in data:
        output_str = item.get("output", "")
        true_nodes = item.get("answer_nodes", [])
        problem_type = item.get("problem_type")  # "bfs" ou "parents"

        # 1. Faz o parsing
        predicted_nodes, parsing_successful = parse_output(output_str)

        # 2. Se o parsing falhar, descarta o item e continua para o próximo
        if not parsing_successful:
            parsing_failures += 1
            continue  # Pula o restante do loop para este item

        # 3. Calcula as métricas de performance
        precision, recall, f1 = calculate_metrics(predicted_nodes, true_nodes)

        # Armazena as métricas na categoria correta
        if problem_type in metrics_by_type:
            metrics_by_type[problem_type].append((precision, recall, f1))
        else:
            print(
                f"Aviso: Tipo de problema desconhecido '{problem_type}' encontrado. Item ignorado nas métricas gerais.")

        # 4. Adiciona os novos campos ao item
        item['predicted_nodes'] = predicted_nodes
        item['metrics'] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        }

        # Adiciona o item processado com sucesso à nova lista
        processed_data.append(item)

    # 5. Salva o JSON enriquecido (apenas com os itens bem-sucedidos)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Resultados detalhados ({len(processed_data)} itens) foram salvos em: '{output_file_path}'")

    # 6. Calcula e salva as métricas gerais segregadas
    if total_items_initial > 0:
        parsing_success_rate = (total_items_initial - parsing_failures) / total_items_initial

        # Calcula métricas para cada tipo de problema
        bfs_metrics = _calculate_average_metrics(metrics_by_type["bfs"], total_items_initial)
        parents_metrics = _calculate_average_metrics(metrics_by_type["parents"], total_items_initial)

        # Calcula métricas gerais (overall) combinando as listas
        all_successful_metrics = metrics_by_type["bfs"] + metrics_by_type["parents"]
        overall_metrics = _calculate_average_metrics(all_successful_metrics, total_items_initial)

        general_metrics = {
            "total_items_in_file": total_items_initial,
            "parsing_success_rate": round(parsing_success_rate, 4),
            "items_with_failed_parsing": parsing_failures,
            "overall": overall_metrics,
            "by_problem_type": {
                "bfs": bfs_metrics,
                "parents": parents_metrics
            }
        }

        with open(general_metrics_file_path, 'w', encoding='utf-8') as f:
            json.dump(general_metrics, f, indent=4, ensure_ascii=False)
        print(f"Métricas gerais salvas em: '{general_metrics_file_path}'")
    else:
        print("Nenhum item encontrado no arquivo para calcular as métricas.")


if __name__ == "__main__":
    # Substitua pelo nome do seu arquivo de entrada
    input_json_filename = "collected_results_augmented.json"
    main(input_json_filename)