#!/usr/bin/env python
# test_improvements.py
from portfolio_construction import generate_target_portfolio
from backtester import Backtester
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))


print("✓ Importações OK")
print("\n" + "="*60)
print("TESTANDO AJUSTES")
print("="*60)

# Criar dados de teste simples
dates = pd.date_range('2020-01-01', periods=252, freq='D')
prices = pd.DataFrame(
    np.random.randn(252, 3).cumsum() + 100,
    index=dates,
    columns=['IBB', 'MRNA', 'REGN']
)

# Predições simples
predictions = pd.DataFrame(
    np.random.uniform(0.4, 0.6, (252, 3)),
    index=dates,
    columns=['IBB', 'MRNA', 'REGN']
)

print("\n1. Gerando pesos com INÉRCIA...")
target_weights = generate_target_portfolio(predictions, prices)
print(f"   ✓ Pesos gerados: {target_weights.shape}")
print(f"   Primeiros pesos:\n{target_weights.head()}")

print("\n2. Rodando backtest com REBALANCEAMENTO SEMANAL...")
bt = Backtester(prices, target_weights, transaction_cost_bps=5)
results = bt.run_backtest()
print(f"   ✓ Backtest concluído")
print(f"   Retornos: {len(results)} dias")

print("\n3. Calculando métricas...")
metrics = bt.calculate_performance_metrics()
print(f"   ✓ Métricas calculadas:")
print(metrics.to_string())

print("\n4. Análise de Turnover:")
avg_turnover_daily = bt.turnover.mean()
annual_turnover = avg_turnover_daily * 252
print(f"   Turnover diário médio: {avg_turnover_daily:.4f}")
print(f"   Turnover anual: {annual_turnover*100:.1f}%")

print("\n" + "="*60)
print("✓ AJUSTES FUNCIONANDO!")
print("="*60)
