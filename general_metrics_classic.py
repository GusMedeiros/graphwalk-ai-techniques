import json
import os
from collections import defaultdict
from tqdm import tqdm


def analyze_metrics():
    """
    Lê o arquivo de métricas, calcula as médias gerais e por tipo de problema,
    e salva o resumo em um novo arquivo JSON.
    """
    input_filename = "graphwalks_results_with_metrics.jsonl"
    output_filename = "classic_general_metrics.json"

    # Verifica se o arquivo de entrada existe
    if not os.path.exists(input_filename):
        print(f"Erro: O arquivo de entrada '{input_filename}' não foi encontrado.")
        print("Certifique-se de executar os scripts anteriores primeiro.")
        return

    # Estrutura para armazenar a SOMA das métricas e a CONTAGEM de cada tipo
    # Usamos defaultdict para simplificar a inicialização
    metrics_aggregator = defaultdict(lambda: defaultdict(float))

    print(f"Analisando o arquivo: '{input_filename}'")

    try:
        # Conta as linhas para a barra de progresso
        num_lines = sum(1 for line in open(input_filename, 'r', encoding='utf-8'))

        with open(input_filename, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile, total=num_lines, desc="Processando métricas"):
                data = json.loads(line)
                problem_type = data.get("problem_type")

                # Ignora linhas que não tenham as métricas calculadas
                if "f1_score" not in data:
                    continue

                # Agrega para a categoria específica ('bfs' ou 'parents')
                if problem_type in ['bfs', 'parents']:
                    metrics_aggregator[problem_type]['precision_sum'] += data['precision']
                    metrics_aggregator[problem_type]['recall_sum'] += data['recall']
                    metrics_aggregator[problem_type]['f1_score_sum'] += data['f1_score']
                    metrics_aggregator[problem_type]['count'] += 1
                else:  # Conta outliers
                    metrics_aggregator['outliers']['count'] += 1

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")
        return

    # ----- Cálculo das Médias e Preparação para Saída -----

    # Estrutura final para o JSON e para a impressão
    final_metrics_summary = {}

    # Categorias que queremos no nosso relatório
    categories = ['bfs', 'parents']

    for category in categories:
        count = metrics_aggregator[category]['count']
        if count > 0:
            final_metrics_summary[category] = {
                "precision": metrics_aggregator[category]['precision_sum'] / count,
                "recall": metrics_aggregator[category]['recall_sum'] / count,
                "f1_score": metrics_aggregator[category]['f1_score_sum'] / count,
                "count": int(count)
            }
        else:  # Caso uma categoria não tenha exemplos
            final_metrics_summary[category] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "count": 0}

    # Calcula as métricas "Overall"
    total_count = sum(data['count'] for data in final_metrics_summary.values())
    if total_count > 0:
        overall_precision = sum(metrics_aggregator[cat]['precision_sum'] for cat in categories) / total_count
        overall_recall = sum(metrics_aggregator[cat]['recall_sum'] for cat in categories) / total_count
        overall_f1 = sum(metrics_aggregator[cat]['f1_score_sum'] for cat in categories) / total_count
        final_metrics_summary['overall'] = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
            "count": int(total_count)
        }
    else:
        final_metrics_summary['overall'] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "count": 0}

    final_metrics_summary['outliers'] = {
        "count": int(metrics_aggregator['outliers']['count'])
    }

    # ----- Impressão do Relatório Formatado -----

    print("\n\n" + "=" * 60)
    print("       Análise de Métricas de Desempenho do GraphWalks")
    print("=" * 60)
    print(f"Análise baseada em {final_metrics_summary['overall']['count']} exemplos totais.")
    print(
        f"({final_metrics_summary['bfs']['count']} do tipo 'bfs', {final_metrics_summary['parents']['count']} do tipo 'parents', e {final_metrics_summary['outliers']['count']} Outlier(s))\n")

    # Obtém os dados para a tabela
    o = final_metrics_summary['overall']
    b = final_metrics_summary['bfs']
    p = final_metrics_summary['parents']

    # Imprime a tabela
    print(f"{'':<11} {'Overall':>8} {'BFS':>8} {'Parents':>8}")
    print(f"{'precision':<11} {o['precision']:>8.4f} {b['precision']:>8.4f} {p['precision']:>8.4f}")
    print(f"{'recall':<11} {o['recall']:>8.4f} {b['recall']:>8.4f} {p['recall']:>8.4f}")
    print(f"{'f1_score':<11} {o['f1_score']:>8.4f} {b['f1_score']:>8.4f} {p['f1_score']:>8.4f}")

    print("\n" + "=" * 60)
    print("Análise concluída.")

    # ----- Salvando o Resumo em JSON -----

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # indent=4 torna o arquivo JSON legível
            json.dump(final_metrics_summary, f, indent=4)
        print(f"\nResumo das métricas salvo em: '{output_filename}'")
    except IOError as e:
        print(f"Não foi possível salvar o arquivo de resumo: {e}")


if __name__ == "__main__":
    analyze_metrics()