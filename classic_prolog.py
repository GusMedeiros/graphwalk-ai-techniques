import json
import time  # 1. Importar o módulo time
from datasets import load_dataset
from pyswip import Prolog
from collections import deque


def parse_graph_to_prolog(graph_string: str) -> list[str]:
    prolog_facts = []
    lines = graph_string.strip().split('\n')

    for line in lines:
        if '->' in line:
            parts = line.split('->')
            node1 = parts[0].strip()
            node2 = parts[1].strip()
            fact = f"points_to('{node1}', '{node2}')"
            if fact not in prolog_facts:
                prolog_facts.append(fact)
    return prolog_facts


def bfs_iterative(prolog_facts: list[str], start_node: str, target_depth: int) -> list[str]:
    """
    Implementação de referência de BFS em Python puro, usando uma fila.
    Serve como "gabarito" para validar a implementação em Prolog.
    """
    adj_list = {}
    for fact in prolog_facts:
        try:
            parts = fact.strip("points_to()").replace("'", "").split(',')
            u, v = parts[0].strip(), parts[1].strip()
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append(v)
        except (ValueError, IndexError):
            continue

    if start_node not in adj_list and target_depth > 0:
        return []

    queue = deque([(start_node, 0)])
    visited = {start_node: 0}
    nodes_at_target_depth = []

    while queue:
        current_node, current_depth = queue.popleft()

        if current_depth >= target_depth:
            if current_depth == target_depth:
                nodes_at_target_depth.append(current_node)
            continue

        for neighbor in adj_list.get(current_node, []):
            if neighbor not in visited:
                visited[neighbor] = current_depth + 1
                queue.append((neighbor, current_depth + 1))

    final_results = [node for node in nodes_at_target_depth if node != start_node]
    return sorted(list(set(final_results)))


def extract_graph_block(full_text: str):
    main_content = full_text.split("Here is the graph to operate on:")[1]
    return main_content.rsplit("Operation:", 1)[0]


def limpar_base_prolog(prolog: Prolog):
    list(prolog.query("retractall(points_to(_, _))"))


def extract_target_node(full_text: str) -> tuple[str | None, int | None]:
    for line in full_text.split('\n'):
        if "Find the parents of node" in line:
            parts = line.split("of node")
            if len(parts) > 1: return parts[1].strip().strip('.'), None
        elif "Perform a BFS from node" in line:
            parts = line.split("from node")
            if len(parts) > 1:
                node_str, depth_str = parts[1].strip().split(" with depth ")
                return node_str.strip(), int(depth_str.strip('.'))
    return None, None


def processar_query_prolog(prolog_facts: list[str], problem_type: str, query_node: str, prolog_engine: Prolog,
                           depth: int | None = None):
    for fact in prolog_facts:
        prolog_engine.assertz(fact.strip('.'))

    result_nodes = []
    variable_name = "Result"

    if problem_type == "parents":
        query_string = f"parent({variable_name}, '{query_node}')"
        query_results = list(prolog_engine.query(query_string))
        for res in query_results:
            if variable_name in res:
                node = res[variable_name]
                result_nodes.append(node.decode('utf-8') if isinstance(node, bytes) else str(node))
    elif problem_type == "bfs":
        query_string = f"bfs('{query_node}', {depth}, {variable_name})"
        query_results = list(prolog_engine.query(query_string))
        if query_results:
            prolog_list = query_results[0].get(variable_name, [])
            result_nodes = [str(atom) for atom in prolog_list]
            if query_node in result_nodes:
                result_nodes.remove(query_node)

    return sorted(list(set(result_nodes)))


if __name__ == "__main__":
    dataset = load_dataset("openai/graphwalks", split="train")

    prolog = Prolog()
    prolog.assertz(":- use_module(library(lists))")
    prolog.assertz(":- dynamic points_to/2")
    prolog.assertz("parent(Node1, Node2) :- points_to(Node1, Node2)")

    prolog.assertz("bfs(Start, TargetDepth, Result) :- bfs_engine([Start], [Start], TargetDepth, Result)")
    prolog.assertz("bfs_engine(CurrentLevel, _, 0, CurrentLevel)")
    prolog.assertz("""
        bfs_engine(CurrentLevel, Visited, TargetDepth, Result) :-
            TargetDepth > 0,
            findall(Next, (member(Node, CurrentLevel), points_to(Node, Next), \\+ member(Next, Visited)), NextNodes),
            sort(NextNodes, NextLevel),
            NextLevel \\= [], !,
            union(Visited, NextLevel, NewVisited),
            NewDepth is TargetDepth - 1,
            bfs_engine(NextLevel, NewVisited, NewDepth, Result)
    """)
    prolog.assertz("bfs_engine(_, _, TargetDepth, []) :- TargetDepth > 0")

    all_results = []

    for i, entry in enumerate(dataset):
        start_time = time.perf_counter()  # 3. Registrar o tempo de início

        prompt: str = entry["prompt"]
        problem_type: str = entry["problem_type"]
        answer_nodes: list = entry["answer_nodes"]

        # 2. Modificar a mensagem de log
        print(f"--- Processing Entry {i + 1}/{len(dataset)} (Prompt size: {len(prompt)}) ---")

        operation_text = prompt.rsplit("Operation:", 1)[1]
        query_node, depth = extract_target_node(operation_text)
        if query_node is None:
            print(f"Skipping entry {i + 1}, could not extract query node.")
            continue

        graph_block = extract_graph_block(prompt)
        prolog_facts = parse_graph_to_prolog(graph_block)

        prolog_result, python_result, final_result = [], [], []

        result_record = {
            "entry_index": i,
            "problem_type": problem_type,
            "query_node": query_node,
            "depth": depth,
            "expected_answer": sorted(answer_nodes),
            "prolog_result": None,
            "python_reference_result": None,
            "status": None,
            "mismatch_details": None,
            "execution_time_ms": None  # Preparar o campo para o tempo
        }

        if problem_type == "parents":
            final_result = processar_query_prolog(prolog_facts, "parents", query_node, prolog)
            result_record["prolog_result"] = final_result
        elif problem_type == "bfs":
            prolog_result = processar_query_prolog(prolog_facts, "bfs", query_node, prolog, depth)
            python_result = bfs_iterative(prolog_facts, query_node, depth)
            final_result = prolog_result
            result_record["prolog_result"] = prolog_result
            result_record["python_reference_result"] = python_result

        limpar_base_prolog(prolog)

        if sorted(answer_nodes) != final_result:
            result_record["status"] = "MISMATCH"
            print("--- MISMATCH FOUND ---")
            print(f"Problem: Find {problem_type} of '{query_node}' with depth {depth}")
            print(f"Expected (Dataset): {sorted(answer_nodes)}")
            print(f"Result (Prolog):    {final_result}")

            if problem_type == "bfs":
                print(f"Result (Python):    {python_result}")
                if prolog_result == python_result:
                    details = "Prolog e Python concordam, mas divergem do gabarito do dataset."
                    print(f"\nConclusion: {details}")
                    result_record["mismatch_details"] = "PROLOG_PYTHON_AGREE_DATASET_DIFFERENT"
                else:
                    details = "ERRO CRÍTICO: Implementações de Prolog e Python discordam. Bug na lógica Prolog."
                    print(f"\nConclusion: {details}")
                    result_record["mismatch_details"] = "PROLOG_PYTHON_DISAGREE"

            print("----------------------")
        else:
            result_record["status"] = "MATCH"

        end_time = time.perf_counter()  # Registrar o tempo de fim
        # 4. Calcular e 5. Adicionar a duração em ms ao registro
        execution_time_ms = (end_time - start_time) * 1000
        result_record["execution_time_ms"] = execution_time_ms

        all_results.append(result_record)

    output_filename = "classic_prolog_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n--- All entries processed! Results saved to {output_filename} ---")