#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script RÁPIDO para gerar o gráfico correto"""

import matplotlib.pyplot as plt
import config
from backtester import Backtester
import portfolio_construction
import modeling
import feature_engineering
import data_ingestion
import pandas as pd
import warnings
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

warnings.filterwarnings('ignore')


print("\n" + "="*80)
print("GERANDO GRÁFICO COM DADOS CORRETOS".center(80))
print("="*80 + "\n")

# Rodar pipeline
unified_data = data_ingestion.get_unified_dataset()
full_feature_set = feature_engineering.create_feature_set(unified_data)
model_runner = modeling.WalkForwardModel()
predictions = model_runner.generate_predictions(full_feature_set)
ativos_validos = [
    c for c in unified_data.columns if c in config.ASSET_UNIVERSE and unified_data[c].notna().mean() > 0.5]
prices_df = unified_data[ativos_validos]
target_weights_df = portfolio_construction.generate_target_portfolio(
    predictions, prices_df)
bt = Backtester(prices_df=prices_df, target_weights_df=target_weights_df)
results = bt.run_backtest()

# Criar gráfico
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(16, 9))

equity_curve = (1 + bt.results).cumprod()
ax.plot(equity_curve.index, equity_curve,
        label='Cassandra Strategy', color='#1f77b4', lw=2.5)

if hasattr(config, 'BENCHMARK_TICKER') and config.BENCHMARK_TICKER in prices_df.columns:
    benchmark_returns = prices_df[config.BENCHMARK_TICKER].pct_change().fillna(
        0)
    benchmark_equity = (1 + benchmark_returns).cumprod()
    ax.plot(benchmark_equity.index, benchmark_equity, label=f'Benchmark ({config.BENCHMARK_TICKER})',
            color='#ff7f0e', ls='--', lw=2.5, alpha=0.7)

ax.set_title('Cassandra Strategy - Performance Completa (2018-2023)',
             fontsize=18, fontweight='bold', pad=20)
ax.set_ylabel('Crescimento do Capital (log scale)',
              fontsize=14, fontweight='bold')
ax.set_xlabel('Data', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('equity_curve.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico salvo como 'equity_curve.png'")

plt.show()

# Mostrar resultados
print("\n" + "="*80)
print("RESULTADOS".center(80))
print("="*80 + "\n")
print(
    f"Período: {equity_curve.index[0].strftime('%Y-%m-%d')} a {equity_curve.index[-1].strftime('%Y-%m-%d')}")
print(f"Retorno Total: {equity_curve.iloc[-1] - 1:.2%}")
print(f"Capital Final: ${equity_curve.iloc[-1]:.2f}")
print(f"Total de Dias: {len(equity_curve)}")
print(f"Dias com Retorno Positivo: {(bt.results > 0).sum()}/{len(bt.results)}")
print(f"Retornos Únicos (variações): {len(bt.results.unique())}")
print("\n" + "="*80 + "\n")
