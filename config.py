# src/config.py

# Período de análise para o backtest
START_DATE = "2018-12-07"
END_DATE = "2023-12-29"

# Universo de Ativos
ASSET_UNIVERSE = {
    # Core da Estratégia Biotech
    'IBB': 'thematic',      # ETF iShares Biotechnology (principal)
    'MRNA': 'thematic',     # Moderna Inc (principal)
    'REGN': 'thematic',     # Regeneron (backup/hedge)
    'AMGN': 'thematic',     # Amgen (backup/hedge)

    # Hedge Macroeconômico
    'GLD': 'hedge',         # Ouro - Hedge para risco sistêmico
    'BND': 'hedge',         # Renda Fixa - Hedge para volatilidade
    'UUP': 'hedge',         # Dólar - Hedge para risco Brasil/global
}

# Features a serem removidas apenas se causarem problemas
VIF_HIGH_CORRELATION_FEATURES_TO_EXCLUDE = [
    # Grupo 1: Remover se VIF > 5000
    'dxy',          # Alta correlação com UUP

    # Grupo 2: Remover se VIF > 1000
    'UUP',          # Manter apenas se dxy for removido
    'BND',          # Manter para hedge, remover se VIF muito alto

    # Grupo 3: Features macro (manter ao menos uma de cada)
    'fed_rate',     # Taxa de juros - remover apenas se instável
    'vix',          # Volatilidade - tentar manter

    # Grupo 4: Ativos Core (nunca remover ambos)
    'IBB',          # Remover apenas se MRNA for mais estável
    'MRNA'          # Remover apenas se IBB for mais estável
]

# Configurações de Backtesting
REBALANCE_THRESHOLD = 0.05  # 5% de tolerância para rebalanceamento
TRANSACTION_COST_BPS = 5    # 5 bps de custo de transação

# Benchmark para cálculo de performance relativa e comparação
BENCHMARK_TICKER = '^GSPC'  # S&P 500

# Tickers para indicadores macroeconômicos do FRED
MACRO_INDICATORS = {
    'DFF': 'fed_rate',      # Federal Funds Effective Rate
    'VIXCLS': 'vix',        # CBOE Volatility Index
    'DTWEXBGS': 'dxy'       # Trade Weighted U.S. Dollar Index
}

# CORREÇÃO FINAL APLICADA AQUI:
# A sintaxe correta para adicionar o ticker do benchmark à lista de ativos.
# Somamos uma lista com outra lista.
ALL_TICKERS = list(ASSET_UNIVERSE.keys()) + [BENCHMARK_TICKER]

# Parâmetros de modelagem (ajustáveis para walk-forward)
MODEL_TRAIN_WINDOW = 252          # ~1 ano de dados para o treinamento base
MODEL_MIN_TRAIN_WINDOW = 126      # janela mínima aceitável para iniciar previsões
MODEL_RETRAIN_FREQUENCY = 63      # retreinamento aproximado trimestral

# Definições para avaliação de multicolinearidade
VIF_THRESHOLD = 10.0

# Parâmetros auxiliares para fallback baseado em momentum
MOMENTUM_LOOKBACK = 63            # ~três meses
MOMENTUM_FALLBACK_PROB = 0.58
MOMENTUM_FALLBACK_PROB_LOW = 0.42
