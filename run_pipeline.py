#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estratégia de Backtesting BioTech - Pipeline Completo
"""

from report_generator import generate_full_report
from backtester import Backtester
from portfolio_construction import generate_target_portfolio
from modeling import ModelingPipeline
from feature_engineering import engineer_features
from data_ingestion import load_data
from datetime import datetime
import numpy as np
import pandas as pd
import sys
import os

# IMPORTANTE: Adicionar src ao path ANTES de fazer imports dos módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


# Agora importar os módulos estratégia


def main():
    """Executa pipeline completo de backtesting."""
    print("="*80)
    print("🚀 INICIANDO PIPELINE DE BACKTESTING - ESTRATÉGIA CASSANDRA")
    print("="*80)
    print()

    # ETAPA 1: Dados
    print("📊 [ETAPA 1] Carregando dados...")
    try:
        prices_df = load_data()
        print(f"✅ Dados carregados")
        print(
            f"   Período: {prices_df.index[0].date()} a {prices_df.index[-1].date()}")
        print(f"   Ativos: {list(prices_df.columns)}")
        print()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return

    # ETAPA 2: Features
    print("⚙️  [ETAPA 2] Gerando features...")
    try:
        features_df = engineer_features(prices_df)
        print(f"✅ Features criadas: {features_df.shape}")
        print()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return

    # ETAPA 3: Modelo
    print("🤖 [ETAPA 3] Treinando modelo...")
    try:
        pipeline = ModelingPipeline(prices=prices_df, features=features_df)
        predictions_df = pipeline.run_walk_forward()
        print(f"✅ Modelo treinado: {predictions_df.shape}")
        print()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return

    # ETAPA 4: Portfólio
    print("💼 [ETAPA 4] Gerando pesos...")
    try:
        target_weights = generate_target_portfolio(predictions_df, prices_df)
        print(f"✅ Pesos gerados: {target_weights.shape}")
        print()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return

    # ETAPA 5: Backtest
    print("📈 [ETAPA 5] Executando backtest...")
    try:
        backtester = Backtester(
            prices_df=prices_df,
            target_weights_df=target_weights,
            transaction_cost_bps=5,
            slippage_bps=2
        )

        portfolio_returns = backtester.run_backtest()
        metrics_df = backtester.calculate_performance_metrics()

        print(f"✅ Backtest concluído")
        print(f"\n   📊 MÉTRICAS:")
        print(f"   ├─ CAGR: {backtester.metrics.get('CAGR', 0)*100:.2f}%")
        print(f"   ├─ Sharpe: {backtester.metrics.get('Sharpe Ratio', 0):.4f}")
        print(
            f"   ├─ MDD: {backtester.metrics.get('Maximum Drawdown (MDD)', 0)*100:.2f}%")
        print(
            f"   ├─ Turnover: {backtester.metrics.get('Average Annual Turnover', 0)*100:.2f}%")
        print(
            f"   └─ Custos: {backtester.metrics.get('Total Transaction Costs', 0)*100:.4f}%")
        print()

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return

    # ETAPA 6: Relatório
    print("📄 [ETAPA 6] Gerando relatório PDF...")
    try:
        backtest_results = backtester.get_backtest_results_for_report()

        report_path = os.path.join(
            os.path.dirname(__file__),
            'notebooks',
            f'relatorio_estrategia_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        generate_full_report(backtest_results, report_path)

        print(f"✅ Relatório gerado!")
        print(f"   📁 {report_path}")
        print()

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return

    print("="*80)
    print("✨ PIPELINE COMPLETADO COM SUCESSO!")
    print("="*80)


if __name__ == '__main__':
    main()
