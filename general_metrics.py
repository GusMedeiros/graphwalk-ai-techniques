import pandas as pd
import json

# --- CONFIGURAÇÕES ---
# Use o nome do arquivo gerado pelo SCRIPT 1
METRICS_FILE = 'collected_results_augmented_with_metrics.json'

# --- SCRIPT PRINCIPAL ---
print(f"Lendo e analisando o arquivo: {METRICS_FILE}\n")

try:
    # Carrega o arquivo JSON diretamente em um DataFrame do pandas
    df = pd.read_json(METRICS_FILE)
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado. Verifique se o arquivo '{METRICS_FILE}' existe.")
    exit()

# Métricas a serem analisadas
metrics_to_analyze = ['precision', 'recall', 'f1_score']

# 1. Estatísticas Gerais (para todo o dataset)
overall_stats = df[metrics_to_analyze].mean()

# 2. Estatísticas por tipo de problema
# Filtra o DataFrame para cada tipo
df_bfs = df[df['problem_type'] == 'bfs']
df_parents = df[df['problem_type'] == 'parents']

# Calcula as médias para cada tipo
bfs_stats = df_bfs[metrics_to_analyze].mean()
parents_stats = df_parents[metrics_to_analyze].mean()

# 3. Criar uma tabela unificada para exibição
summary_df = pd.DataFrame({
    'Overall': overall_stats,
    'BFS': bfs_stats,
    'Parents': parents_stats
})

# Formatação para melhor visualização
pd.options.display.float_format = '{:.4f}'.format

# Exibe os resultados
print("="*60)
print("       Análise de Métricas de Desempenho da Amostra")
print("="*60)
print(f"Análise baseada em {len(df)} exemplos totais.")
print(f"({len(df_bfs)} do tipo 'bfs', {len(df_parents)} do tipo 'parents', e {len(df) - len(df_bfs) - len(df_parents)} Outlier(s))\n")

print(summary_df)

print("\n" + "="*60)
print("Análise concluída.")