if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from data_ingestion import get_unified_dataset as load_data
    from feature_engineering import create_feature_set as engineer_features
    from modeling import WalkForwardModel
    from portfolio_construction import generate_target_portfolio
    from backtester import Backtester
    from report_generator import generate_full_report
    from datetime import datetime

    print("="*80)
    print("🚀 GERANDO RELATÓRIO - ESTRATÉGIA CASSANDRA")
    print("="*80)
    print()
    print("📊 Dados...")
    prices_df = load_data()
    print(f"   ✓ {prices_df.shape}")
    print("⚙️  Features...")
    features_df = engineer_features(prices_df)
    print(f"   ✓ {features_df.shape}")
    print("🤖 Modelo...")
    pipeline = WalkForwardModel()
    predictions_df = pipeline.generate_predictions(features_df)
    print(f"   ✓ {predictions_df.shape}")
    print("💼 Portfólio...")
    target_weights = generate_target_portfolio(predictions_df, prices_df)
    print(f"   ✓ {target_weights.shape}")
    print("📈 Backtest...")
    bt = Backtester(prices_df=prices_df, target_weights_df=target_weights)
    bt.run_backtest()
    bt.calculate_performance_metrics()
    m = bt.metrics
    print(
        f"   ✓ CAGR: {m.get('CAGR', 0)*100:.2f}%  |  Sharpe: {m.get('Sharpe Ratio', 0):.4f}")
    print("📄 PDF...")
    results = bt.get_backtest_results_for_report()
    os.makedirs('notebooks', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = os.path.join(
        'notebooks', f'relatorio_estrategia_{timestamp}.pdf')
    generate_full_report(results, pdf_path)
    tamanho_kb = os.path.getsize(pdf_path) / 1024
    print(f"   ✓ {pdf_path} ({tamanho_kb:.1f} KB)")
    print("\n" + "="*80)
    print("✨ SUCESSO!")
    print("="*80)
