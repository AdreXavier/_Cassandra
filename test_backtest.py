#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script de teste completo do backtest"""

from backtester import Backtester
import portfolio_construction
import modeling
import feature_engineering
import data_ingestion
import config
import pandas as pd
import numpy as np
import warnings
import sys
import os

# Adicionar src ao path ANTES de importar
sys.path.insert(0, os.path.abspath('src'))

warnings.filterwarnings('ignore')


def main():
    try:
        print("="*80)
        print("TESTE COMPLETO DO BACKTEST".center(80))
        print("="*80)

        # 1. Carregar dados
        print("\n1. Carregando dados...")
        unified_data = data_ingestion.get_unified_dataset()
        print(f"   ✓ Dados carregados: {len(unified_data)} linhas")
        print(
            f"   ✓ Período: {unified_data.index[0]} a {unified_data.index[-1]}")
        print(f"   ✓ Colunas: {len(unified_data.columns)}")

        # 2. Criar features
        print("\n2. Criando features...")
        full_feature_set = feature_engineering.create_feature_set(unified_data)
        print(f"   ✓ Features criadas: {full_feature_set.shape}")

        # 3. Gerar previsões
        print("\n3. Gerando previsões com modelo...")
        model_runner = modeling.WalkForwardModel()
        predictions = model_runner.generate_predictions(full_feature_set)
        print(f"   ✓ Previsões geradas: {len(predictions)} dias")

        # 4. Construir portfólio
        print("\n4. Construindo portfólio...")
        ativos_validos = [c for c in unified_data.columns
                          if c in config.ASSET_UNIVERSE and
                          unified_data[c].notna().mean() > 0.5]
        print(f"   ✓ Ativos válidos: {ativos_validos}")

        prices_df = unified_data[ativos_validos]
        target_weights_df = portfolio_construction.generate_target_portfolio(
            predictions, prices_df
        )
        print(f"   ✓ Pesos do portfólio: {target_weights_df.shape}")

        # 5. Executar backtest
        print("\n5. Executando backtest...")
        bt = Backtester(prices_df=prices_df,
                        target_weights_df=target_weights_df)
        results = bt.run_backtest()
        print(f"   ✓ Backtest concluído!")
        print(f"   ✓ Retornos calculados: {len(results)} dias")

        # 6. Calcular métricas
        print("\n6. Calculando métricas de desempenho...")
        metrics = bt.calculate_performance_metrics()
        print(f"   ✓ Métricas calculadas: {len(metrics)} métricas")

        # 7. Exibir principais métricas
        print("\n7. Principais Métricas:")
        print("-" * 80)

        def format_value(x, is_pct=False):
            if pd.isna(x):
                return "N/A"
            try:
                if isinstance(x, str):
                    x = float(x.strip('%').replace(',', ''))
                if isinstance(x, (pd.Series, np.ndarray)):
                    x = x.iloc[0] if len(x) > 0 else np.nan
                return f"{x*100:.2f}%" if is_pct else f"{x:.3f}"
            except:
                return "N/A"

        metrics_to_show = [
            ("CAGR", True),
            ("Cumulative Return", True),
            ("Annualized Volatility", True),
            ("Maximum Drawdown (MDD)", True),
            ("Sharpe Ratio", False),
            ("Sortino Ratio", False),
            ("Calmar Ratio", False),
            ("Hit Rate", True),
        ]

        for metric_name, is_pct in metrics_to_show:
            if metric_name in metrics.index:
                value = metrics.loc[metric_name, "Value"]
                formatted = format_value(value, is_pct)
                print(f"   {metric_name:.<50} {formatted}")

        print("\n" + "="*80)
        print("✓ BACKTEST EXECUTADO COM SUCESSO!".center(80))
        print("="*80)

        return True

    except Exception as e:
        print(f"\n✗ ERRO: {type(e).__name__}")
        print(f"   {str(e)}")
        print("\nStacktrace completo:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
