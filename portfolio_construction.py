# src/portfolio_construction.py
import pandas as pd
import numpy as np
import config


def calculate_confidence_score(predictions_df):
    """
    Calcula o Score de Confiança a partir das probabilidades de previsão.
    Score = P(Alta) - 0.5
    """
    confidence_scores = predictions_df - 0.5
    return confidence_scores


def calculate_volatility_scaled_weights(confidence_scores, prices_df, vol_window=60):
    """
    Calcula os pesos preliminares escalonados pela volatilidade inversa.
    Peso_Preliminar = Score_Confiança / Volatilidade_Recente
    """
    # Calcula os retornos diários
    returns = prices_df.pct_change()

    # Calcula a volatilidade histórica (desvio padrão dos retornos)
    volatility = returns.rolling(window=vol_window).std()

    # Alinha os dataframes de scores e volatilidade
    scores_aligned, volatility_aligned = confidence_scores.align(
        volatility, join='inner', axis=0)

    # Evita divisão por zero
    volatility_aligned[volatility_aligned == 0] = np.nan

    # Calcula os pesos brutos
    raw_weights = scores_aligned / volatility_aligned

    return raw_weights


def generate_target_portfolio(predictions_df, prices_df):
    """
    Gera os pesos do portfólio final aplicando a lógica de alocação.
    """
    # 1. Calcular o Score de Confiança
    confidence_scores = calculate_confidence_score(predictions_df)

    # Garantir alinhamento com os ativos disponíveis em prices_df
    asset_columns = [
        col for col in prices_df.columns if col in confidence_scores.columns]
    confidence_scores = confidence_scores.reindex(columns=asset_columns)

    # Overlay de momentum para reforçar sinais fracos
    momentum_window = getattr(config, 'MOMENTUM_LOOKBACK', 63)
    returns = prices_df[asset_columns].pct_change()
    momentum_returns = prices_df[asset_columns].pct_change(
        momentum_window).reindex(confidence_scores.index)
    momentum_vol = returns.rolling(momentum_window).std().reindex(
        confidence_scores.index)
    momentum_signal = (momentum_returns / (momentum_vol *
                       np.sqrt(momentum_window))).replace([np.inf, -np.inf], 0)
    momentum_signal = momentum_signal.fillna(0)
    confidence_scores = confidence_scores.add(
        0.15 * np.tanh(momentum_signal), fill_value=0)

    # 2. Calcular os pesos brutos com escalonamento por volatilidade
    raw_weights = calculate_volatility_scaled_weights(
        confidence_scores, prices_df)

    # 3. Aplicar a regra tática long-only
    # Apenas consideramos pesos positivos
    positive_weights = raw_weights.where(raw_weights > 0, 0)

    # 4. Normalizar os pesos para que somem 1
    row_sum = positive_weights.sum(axis=1)
    # Evita divisão por zero em dias sem sinais positivos
    row_sum[row_sum == 0] = 1.0

    target_weights = positive_weights.div(row_sum, axis=0)

    # 4b. Dias sem alocação (todos os scores <= 0) -> fallback para top-N
    fallback_days = target_weights.sum(axis=1).fillna(0) == 0
    if fallback_days.any():
        fallback_scores = confidence_scores.loc[fallback_days].copy()
        # Substituir NaNs por valores muito baixos para permitir ranqueamento
        fallback_scores = fallback_scores.fillna(-np.inf)

        # Número de posições máximas no fallback (1 ou 2 dependendo do universo)
        max_positions = min(3, max(1, len(asset_columns)))

        fallback_alloc = pd.DataFrame(
            0.0, index=fallback_scores.index, columns=asset_columns)
        for idx, row in fallback_scores.iterrows():
            # Seleciona os ativos com maiores probabilidades, mesmo que < 0.5
            top_assets = row.nlargest(max_positions).index
            valid_assets = [
                asset for asset in top_assets if row[asset] != -np.inf]
            if not valid_assets:
                continue
            weight = 1.0 / len(valid_assets)
            fallback_alloc.loc[idx, valid_assets] = weight

        # Substitui apenas nas datas sem alocação original
        target_weights.loc[fallback_alloc.index] = fallback_alloc

    target_weights = target_weights.fillna(0.0)

    # Garantir que os pesos somam a 1
    row_sum = target_weights[asset_columns].sum(axis=1)
    target_weights[asset_columns] = target_weights[asset_columns].div(
        row_sum, axis=0)

    # Alocar Cash como 1 - soma dos ativos
    target_weights['Cash'] = 1 - target_weights[asset_columns].sum(axis=1)

    print("Pesos do portfólio alvo gerados com sucesso.")
    return target_weights
