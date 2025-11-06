import json

# Ler notebook
with open('run_backtest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Encontrar a célula de bootstrap
cell_idx = None
for i, cell in enumerate(nb['cells']):
    if 'source' in cell:
        src = ''.join(cell['source'])
        if 'simulate_robust_bootstrap' in src and 'all_equity_curves.append' in src:
            cell_idx = i
            break

if cell_idx is not None:
    # Código corrigido
    new_code = """# === PASSO 7: Análise de Robustez com Bootstrap de Retornos ===
import plotly.graph_objects as go
import numpy as np
import pandas as pd

print("\\n" + "="*80)
print("ANÁLISE DE ROBUSTEZ - SIMULAÇÕES BOOTSTRAP".center(80))
print("="*80)

def simulate_robust_bootstrap(prices_df, target_weights_df, n_simulations=500):
    \"\"\"
    Simula a estratégia usando bootstrap de retornos históricos.
    Mais realista que adicionar ruído aos pesos.
    \"\"\"
    all_equity_curves = []
    
    # Calcular retornos históricos
    returns = prices_df.pct_change().fillna(0)
    
    print(f"\\nExecutando {n_simulations} simulações bootstrap...")
    
    for i in range(n_simulations):
        # Amostrar retornos com reposição
        idx = np.random.choice(len(returns), size=len(returns), replace=True)
        bootstrap_returns = returns.iloc[idx].values
        bootstrap_returns = pd.DataFrame(
            bootstrap_returns, 
            columns=returns.columns, 
            index=returns.index
        )
        
        # Calcular retornos do portfólio
        weights_aligned = target_weights_df.reindex(bootstrap_returns.index).fillna(method='ffill').fillna(0)
        portfolio_ret = (weights_aligned * bootstrap_returns).sum(axis=1)
        
        # Aplicar custos de transação
        turnover = weights_aligned.diff().abs().sum(axis=1)
        transaction_costs = turnover * 0.0005  # 5 bps
        net_returns = portfolio_ret - transaction_costs
        
        # Calcular curva de equity começando de 1.0
        equity = pd.Series(1.0, index=net_returns.index)
        equity.iloc[1:] = (1 + net_returns.iloc[1:]).cumprod()
        all_equity_curves.append(equity)
        
        if (i + 1) % 100 == 0:
            print(f"  ✓ {i + 1}/{n_simulations} simulações concluídas")
    
    return all_equity_curves

# Executar simulações bootstrap
bootstrap_curves = simulate_robust_bootstrap(prices_df, target_weights_df, n_simulations=500)

# Calcular percentis
equity_matrix = pd.DataFrame(bootstrap_curves)
percentile_5 = equity_matrix.quantile(0.05, axis=0)
percentile_25 = equity_matrix.quantile(0.25, axis=0)
percentile_50 = equity_matrix.quantile(0.50, axis=0)
percentile_75 = equity_matrix.quantile(0.75, axis=0)
percentile_95 = equity_matrix.quantile(0.95, axis=0)

# Estratégia original
original_equity = (1 + bt.results).cumprod()

# Criar gráfico com Plotly
fig = go.Figure()

# Intervalo de confiança 90% (preenchido)
fig.add_trace(go.Scatter(
    x=list(original_equity.index) + list(original_equity.index[::-1]),
    y=list(percentile_5.values) + list(percentile_95.values[::-1]),
    fill='toself',
    fillcolor='rgba(0, 100, 200, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='IC 90%'
))

# Intervalo IQR 50% (preenchido)
fig.add_trace(go.Scatter(
    x=list(original_equity.index) + list(original_equity.index[::-1]),
    y=list(percentile_25.values) + list(percentile_75.values[::-1]),
    fill='toself',
    fillcolor='rgba(0, 150, 255, 0.3)',
    line=dict(color='rgba(255,255,255,0)'),
    name='IQR (25%-75%)'
))

# Mediana
fig.add_trace(go.Scatter(
    x=original_equity.index,
    y=percentile_50,
    mode='lines',
    line=dict(color='blue', width=2),
    name='Mediana das Simulações'
))

# Estratégia original (com destaque)
fig.add_trace(go.Scatter(
    x=original_equity.index,
    y=original_equity,
    mode='lines',
    line=dict(color='red', width=3),
    name='Estratégia Real (Backtest)'
))

# Benchmark
if hasattr(config, 'BENCHMARK_TICKER') and config.BENCHMARK_TICKER in prices_df.columns:
    benchmark_returns = prices_df[config.BENCHMARK_TICKER].pct_change().fillna(0)
    benchmark_equity = (1 + benchmark_returns).cumprod()
    fig.add_trace(go.Scatter(
        x=original_equity.index,
        y=benchmark_equity,
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name=f'Benchmark ({config.BENCHMARK_TICKER})'
    ))

fig.update_layout(
    title='Análise de Robustez: Simulações Bootstrap vs. Estratégia Real',
    xaxis_title='Data',
    yaxis_title='Crescimento do Capital (log scale)',
    yaxis_type='log',
    height=700,
    hovermode='x unified',
    template='plotly_white',
    font=dict(size=12)
)

fig.write_html(os.path.join(output_dir, 'monte_carlo_simulations.html'))
try:
    fig.write_image(os.path.join(output_dir, 'monte_carlo_simulations.pdf'))
except:
    print("⚠️ PDF não gerado (kaleido pode não estar instalado)")

fig.show()

# Estatísticas das simulações
print("\\n" + "="*80)
print("ESTATÍSTICAS DAS SIMULAÇÕES".center(80))
print("="*80)

final_values = equity_matrix.iloc[-1]
print(f"\\nRetorno Final (capital x1):")
print(f"  Percentil 5%:   {percentile_5.iloc[-1]:.2f}x")
print(f"  Percentil 25%:  {percentile_25.iloc[-1]:.2f}x")
print(f"  Mediana (50%):  {percentile_50.iloc[-1]:.2f}x")
print(f"  Percentil 75%:  {percentile_75.iloc[-1]:.2f}x")
print(f"  Percentil 95%:  {percentile_95.iloc[-1]:.2f}x")
print(f"  Estratégia Real: {original_equity.iloc[-1]:.2f}x")

print(f"\\nRetornos Totais (%):")
print(f"  Pior caso (5%):     {(percentile_5.iloc[-1] - 1) * 100:.1f}%")
print(f"  Caso médio (50%):   {(percentile_50.iloc[-1] - 1) * 100:.1f}%")
print(f"  Melhor caso (95%):  {(percentile_95.iloc[-1] - 1) * 100:.1f}%")
print(f"  Estratégia Real:    {(original_equity.iloc[-1] - 1) * 100:.1f}%")

print(f"\\n✓ Análise de robustez concluída!")
"""

    # Aplicar mudança
    nb['cells'][cell_idx]['source'] = new_code.split('\n')

    # Salvar
    with open('run_backtest.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("✓ Célula de simulação corrigida!")
    print(f"  - Célula {cell_idx} atualizada")
    print(f"  - Linha de cumprod() corrigida para começar de 1.0")
else:
    print("Célula não encontrada!")
