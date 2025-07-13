import json
from collections import Counter


def calculate_summary_stats(results: list) -> dict:
    """
    Calcula estatísticas agregadas a partir de uma lista de resultados.
    """
    if not results:
        return {}

    total_entries = len(results)

    # Acumuladores para as métricas gerais
    total_f1 = sum(entry.get('f1_score', 0) for entry in results)
    total_precision = sum(entry.get('precision', 0) for entry in results)
    total_recall = sum(entry.get('recall', 0) for entry in results)
    total_time_ms = sum(entry.get('execution_time_ms', 0) for entry in results)

    # Contagem de status
    status_counts = Counter(entry.get('status') for entry in results)

    # Contagem de detalhes de mismatch
    mismatch_details = Counter(
        entry.get('mismatch_details') for entry in results if entry.get('status') == 'MISMATCH'
    )

    # Calcula as médias
    summary = {
        "count": total_entries,
        "matches": status_counts.get('MATCH', 0),
        "mismatches": status_counts.get('MISMATCH', 0),
        "mismatch_details": dict(mismatch_details),
        "average_f1_score": total_f1 / total_entries if total_entries > 0 else 0,
        "average_precision": total_precision / total_entries if total_entries > 0 else 0,
        "average_recall": total_recall / total_entries if total_entries > 0 else 0,
        "average_time_ms": total_time_ms / total_entries if total_entries > 0 else 0,
        "total_time_ms": total_time_ms
    }
    return summary


def print_summary(title: str, stats: dict):
    """Formata e imprime um bloco de estatísticas."""
    if not stats:
        print(f"--- {title}: Sem dados para exibir ---")
        return

    print("-" * (len(title) + 6))
    print(f"|  {title}  |")
    print("-" * (len(title) + 6))

    print(f"  Total de Entradas: {stats['count']}")
    if 'matches' in stats:
        accuracy = (stats['matches'] / stats['count']) * 100 if stats['count'] > 0 else 0
        print(f"  Acertos (Match):   {stats['matches']} ({accuracy:.2f}%)")
        print(f"  Erros (Mismatch):  {stats['mismatches']}")
        if stats['mismatches'] > 0:
            print("    Detalhes dos Erros:")
            for detail, count in stats['mismatch_details'].items():
                print(f"      - {str(detail).replace('_', ' ').title()}: {count}")

    print("\nMétricas de Desempenho Médias:")
    print(f"  - F1-Score:  {stats['average_f1_score']:.4f}")
    print(f"  - Precision: {stats['average_precision']:.4f}")
    print(f"  - Recall:    {stats['average_recall']:.4f}")

    print("\nMétricas de Tempo:")
    print(f"  - Tempo Médio por Entrada: {stats['average_time_ms']:.2f} ms")
    if 'total_time_ms' in stats:
        print(f"  - Tempo Total de Execução: {stats['total_time_ms'] / 1000:.2f} s")
    print("\n")


if __name__ == "__main__":
    input_filename = "classic_prolog_results_with_metrics.json"

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada '{input_filename}' não foi encontrado.")
        print("Por favor, execute primeiro o script 'analyze_results.py' para gerar este arquivo.")
        exit(1)

    # 1. Calcular estatísticas gerais
    general_stats = calculate_summary_stats(all_results)

    # 2. Separar resultados por tipo de problema
    results_by_type = {}
    for entry in all_results:
        ptype = entry.get("problem_type")
        if ptype not in results_by_type:
            results_by_type[ptype] = []
        results_by_type[ptype].append(entry)

    # 3. Calcular estatísticas para cada tipo de problema
    stats_by_type = {}
    for ptype, results_list in results_by_type.items():
        stats_by_type[ptype] = calculate_summary_stats(results_list)

    # 4. Imprimir o relatório final
    print("\n" + "=" * 50)
    print(" " * 10 + "RELATÓRIO DE DESEMPENHO GERAL")
    print("=" * 50 + "\n")

    print_summary("Estatísticas Gerais (Todos os Problemas)", general_stats)

    print("=" * 50)
    print(" " * 8 + "ANÁLISE POR TIPO DE PROBLEMA")
    print("=" * 50 + "\n")

    for ptype, stats in sorted(stats_by_type.items()):
        print_summary(f"Estatísticas para '{ptype.upper()}'", stats)