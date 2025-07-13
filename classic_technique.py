import json
import collections
import re
from datasets import load_dataset
from itertools import islice
import os
import time


# ==============================================================================
# 1. Funções de Lógica de Grafo (sem alterações na lógica interna)
# ==============================================================================

def find_parents(graph, target_node):
    """Encontra todos os nós pais do nó alvo."""
    parents = set()
    for source_node, destinations in graph.items():
        if target_node in destinations:
            parents.add(source_node)
    return sorted(list(parents))


def find_bfs_at_depth(graph, start_node, depth):
    """Executa BFS e retorna nós na profundidade exata."""
    if depth == 0:
        return []

    visited = {start_node}
    current_level_nodes = {start_node}

    for _ in range(depth):
        next_level_nodes = set()
        for node in current_level_nodes:
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level_nodes.add(neighbor)

        current_level_nodes = next_level_nodes
        if not current_level_nodes:
            break

    return sorted(list(current_level_nodes))


def executar_operacao_do_prompt(prompt_text):
    """
    Faz o parsing de um prompt, executa a operação de grafo e retorna a lista de nós.
    Retorna uma string de erro se o parsing falhar.
    """
    try:
        problem_section = prompt_text.split("<end example>")[1]
    except IndexError:
        return "Erro de Parsing: Marcação '<end example>' não encontrada."

    try:
        graph_section = problem_section.split("The graph has the following edges:\n")[1].split("\n\n\nOperation:")[0]
        graph = collections.defaultdict(list)
        for line in graph_section.strip().split('\n'):
            if ' -> ' in line:
                source, dest = map(str.strip, line.split(' -> '))
                graph[source].append(dest)
    except IndexError:
        return "Erro de Parsing: Seção do grafo não encontrada na seção do problema."

    try:
        operation_line = problem_section.split("Operation:\n")[1].split("\n\nYou should")[0].strip()
        if 'BFS' in operation_line:
            match = re.search(r"Perform a BFS from node (\w+) with depth (\d+)", operation_line)
            if match:
                start_node, depth = match.groups()
                return find_bfs_at_depth(graph, start_node, int(depth))
            else:
                return "Erro de Parsing: Formato da operação BFS não reconhecido."
        elif 'parents' in operation_line:
            match = re.search(r"Find the parents of node (\w+)", operation_line)
            if match:
                target_node = match.group(1)
                return find_parents(graph, target_node)
            else:
                return "Erro de Parsing: Formato da operação 'parents' não reconhecido."
        else:
            return "Erro de Parsing: Tipo de operação desconhecido."
    except IndexError:
        return "Erro de Parsing: Seção da operação não encontrada na seção do problema."


# ==============================================================================
# 2. Função Principal para Processar o Dataset (com medição de tempo por entrada)
# ==============================================================================

def main():
    """
    Carrega o dataset, processa cada prompt com capacidade de retomada, mede o tempo de
    execução de cada um e salva os resultados.
    """
    output_filename = "graphwalks_results.jsonl"

    processed_count = 0
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for line in f if line.strip())

    if processed_count > 0:
        print(f"Arquivo de resultados encontrado. Retomando o trabalho após {processed_count} exemplos já processados.")
    else:
        print("Nenhum progresso anterior encontrado. Começando do zero.")

    print("Carregando o dataset 'openai/graphwalks' em modo streaming...")
    dataset = load_dataset("openai/graphwalks", split="train", streaming=True)

    dataset_to_process = islice(dataset, processed_count, None)

    print("Iniciando o processamento dos prompts...")
    total_start_time = time.time() # Mede o tempo total da execução

    with open(output_filename, 'a', encoding='utf-8') as f:
        for index, example in enumerate(dataset_to_process, start=processed_count):
            prompt = example['prompt']

            # <--- ALTERAÇÃO: Medição de tempo para uma única entrada ---
            entry_start_time = time.time()
            returned_nodes = executar_operacao_do_prompt(prompt)
            entry_end_time = time.time()
            execution_time_s = entry_end_time - entry_start_time
            # <--- FIM DA ALTERAÇÃO ---

            # <--- ALTERAÇÃO: Adiciona o campo de tempo de execução ao resultado ---
            result_entry = {
                "id": index,
                "problem_type": example['problem_type'],
                "prompt": prompt,
                "output": returned_nodes,
                "answer_nodes": example['answer_nodes'],
                "execution_time_s": execution_time_s # Novo campo com o tempo em segundos
            }
            # <--- FIM DA ALTERAÇÃO ---

            # SALVAMENTO INCREMENTAL
            f.write(json.dumps(result_entry) + '\n')
            f.flush()

            # PRINT DE PROGRESSO
            if (index + 1) % 100 == 0:
                elapsed_time = time.time() - total_start_time
                print(
                    f"--- Progresso: {index + 1} exemplos processados. Último salvo: id={index}. Tempo decorrido: {elapsed_time:.2f}s ---")

    print(f"\nProcessamento concluído. Todos os resultados foram salvos em '{output_filename}'.")
    final_count = 0
    with open(output_filename, 'r', encoding='utf-8') as f:
        final_count = sum(1 for line in f if line.strip())
    print(f"Total de exemplos no arquivo: {final_count}.")


# ==============================================================================
# 3. Ponto de Entrada do Script
# ==============================================================================

if __name__ == "__main__":
    main()