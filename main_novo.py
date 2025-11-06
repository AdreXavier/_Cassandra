#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point para executar a estratégia BioAlpha de ponta a ponta.
"""

from data_ingestion import load_data
from feature_engineering import engineer_features
from modeling import ModelingPipeline
from portfolio_construction import generate_target_portfolio
from backtester import Backtester
from report_generator import generate_full_report
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path ANTES dos imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Imports da estratégia


def main():
    """
    Executa o pipeline completo: dados -> features -> modelo -> backtest -> relatório.
    """
    print("="*80)
    print("🚀 INICIANDO PIPELINE DE BACKTESTING - ESTRATÉGIA BIOALPHA")
    print("="*80)
    print()

    # ========== ETAPA 1: Carregar Dados ==========
    print("📊 [ETAPA 1] Carregando dados...")
    try:
        prices_df = load_data()
        print(f"✅ Dados carregados com sucesso")
        print(
            f"   Período: {prices_df.index[0].date()} a {prices_df.index[-1].date()}")
        print(f"   Ativos: {list(prices_df.columns)}")
        print()
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return

    # ========== ETAPA 2: Engenharia de Features ==========
    print("⚙️  [ETAPA 2] Gerando features...")
    try:
        features_df = engineer_features(prices_df)
        print(f"✅ Features criadas: {features_df.shape[1]} variáveis")
        print(f"   Períodos de treinamento: {features_df.shape[0]}")
        print()
    except Exception as e:
        print(f"❌ Erro ao criar features: {e}")
        return

    # ========== ETAPA 3: Treinamento Walk-Forward ==========
    print("🤖 [ETAPA 3] Treinando modelo em regime walk-forward...")
    try:
        pipeline = ModelingPipeline(prices=prices_df, features=features_df)
        predictions_df = pipeline.run_walk_forward()
        print(f"✅ Modelo treinado com sucesso")
        print(f"   Períodos com previsões: {predictions_df.shape[0]}")
        print()
    except Exception as e:
        print(f"❌ Erro ao treinar modelo: {e}")
        return

    # ========== ETAPA 4: Geração de Portfólio ==========
    print("💼 [ETAPA 4] Gerando pesos do portfólio...")
    try:
        target_weights = generate_target_portfolio(predictions_df, prices_df)
        print(f"✅ Pesos do portfólio gerados")
        print(f"   Alocações ativas: {target_weights.shape[0]} dias")
        print()
    except Exception as e:
        print(f"❌ Erro ao gerar pesos: {e}")
        return

    # ========== ETAPA 5: Backtesting ==========
    print("📈 [ETAPA 5] Executando backtest...")
    try:
        backtester = Backtester(
            prices_df=prices_df,
            target_weights_df=target_weights,
            transaction_cost_bps=5,
            slippage_bps=2
        )

        # Executar backtest
        portfolio_returns = backtester.run_backtest()
        print(f"✅ Backtest concluído")

        # Calcular métricas
        print("\n   Calculando métricas de performance...")
        metrics_df = backtester.calculate_performance_metrics()

        # Exibir principais métricas
        print("\n   📊 PRINCIPAIS MÉTRICAS:")
        print(f"   ├─ CAGR: {backtester.metrics.get('CAGR', 0)*100:.2f}%")
        print(
            f"   ├─ Sharpe Ratio: {backtester.metrics.get('Sharpe Ratio', 0):.4f}")
        print(
            f"   ├─ Max Drawdown: {backtester.metrics.get('Maximum Drawdown (MDD)', 0)*100:.2f}%")
        print(
            f"   ├─ Annual Turnover: {backtester.metrics.get('Average Annual Turnover', 0)*100:.2f}%")
        print(
            f"   └─ Transaction Costs: {backtester.metrics.get('Total Transaction Costs', 0)*100:.4f}%")
        print()

    except Exception as e:
        print(f"❌ Erro no backtest: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== ETAPA 6: Gerar Relatório PDF ==========
    print("📄 [ETAPA 6] Gerando relatório em PDF...")
    try:
        # Coletar todos os resultados
        backtest_results = backtester.get_backtest_results_for_report()

        # Gerar PDF
        report_path = os.path.join(
            os.path.dirname(__file__),
            'notebooks',
            f'relatorio_estrategia_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        generate_full_report(backtest_results, report_path)
        print(f"✅ Relatório gerado com sucesso!")
        print(f"   📁 Localização: {report_path}")
        print()

    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== CONCLUSÃO ==========
    print("="*80)
    print("✨ PIPELINE COMPLETADO COM SUCESSO!")
    print("="*80)
    print(f"Relatório disponível em: {report_path}")
    print()


if __name__ == '__main__':
    main()
