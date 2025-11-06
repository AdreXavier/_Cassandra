# src/modeling.py
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import config


class WalkForwardModel:
    def __init__(self, model_params=None, train_window=None, retrain_every=None, min_train_window=None):
        """Gerencia o processo de walk-forward por ativo."""
        if model_params is None:
            self.model_params = {
                'n_estimators': 40,
                'max_depth': 12,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            self.model_params = model_params

        self.train_window = train_window or getattr(
            config, 'MODEL_TRAIN_WINDOW', 252)
        self.retrain_every = retrain_every or getattr(
            config, 'MODEL_RETRAIN_FREQUENCY', 21)
        self.min_train_window = min_train_window or getattr(
            config, 'MODEL_MIN_TRAIN_WINDOW', 126)

        self.models_by_asset = {}

    def _get_features_for_asset(self, full_df, asset):
        """Seleciona o conjunto de features para um ativo específico."""
        macro_candidates = [
            'fed_rate', 'fed_rate_diff', 'vix', 'dxy', 'biotech_flow_proxy'
        ] + [f'biotech_relative_strength_{w}d' for w in (21, 63, 252)]

        asset_features = [
            col for col in full_df.columns
            if col.startswith(asset) and not col.endswith('_target')
        ]

        feature_list = [
            col for col in macro_candidates + asset_features if col in full_df.columns
        ]

        features = full_df[feature_list].copy()
        return features.dropna(axis=1, how='all')

    def _build_fallback_series(self, asset, full_dataset):
        """Cria probabilidades de fallback baseadas em momentum simples."""
        lookback = getattr(config, 'MOMENTUM_LOOKBACK', 63)
        high_prob = getattr(config, 'MOMENTUM_FALLBACK_PROB', 0.58)
        low_prob = getattr(config, 'MOMENTUM_FALLBACK_PROB_LOW', 0.42)

        if asset not in full_dataset.columns:
            return pd.Series(0.5, index=full_dataset.index, dtype=float)

        prices = pd.to_numeric(full_dataset[asset], errors='coerce')
        momentum = prices.pct_change(lookback)
        fallback = pd.Series(low_prob, index=full_dataset.index, dtype=float)
        fallback.loc[momentum > 0] = high_prob
        return fallback.fillna(0.5)

    def generate_predictions(self, full_dataset):
        """Executa o walk-forward e devolve probabilidades por ativo."""
        asset_list = list(config.ASSET_UNIVERSE.keys())
        all_predictions = {}

        for asset in asset_list:
            print(f"Iniciando processo walk-forward para o ativo: {asset}")

            target_col = f'{asset}_target'
            if target_col not in full_dataset.columns:
                print(f"  Aviso: alvo {target_col} ausente; pulando ativo.")
                continue

            X = self._get_features_for_asset(full_dataset, asset)
            if X.empty:
                print(
                    f"  Aviso: sem features disponíveis para {asset}; pulando ativo.")
                continue

            # Alinha target e features
            data = X.copy()
            data[target_col] = full_dataset[target_col]
            data = data.dropna(how='any')
            if len(data) < self.min_train_window:
                print(
                    f"  Aviso: dados insuficientes para {asset}; usando fallback.")
                fallback_series = self._build_fallback_series(
                    asset, full_dataset)
                all_predictions[asset] = fallback_series
                continue

            asset_predictions = pd.Series(
                index=full_dataset.index, dtype=float)
            clean_data = data.copy()
            dates = clean_data.index
            last_retrain_day = -1
            model = None

            for i in range(self.min_train_window, len(clean_data)):
                window_start = max(0, i - self.train_window)
                history = clean_data.iloc[window_start:i]
                if len(history) < self.min_train_window:
                    continue

                if (last_retrain_day == -1) or (i >= last_retrain_day + self.retrain_every) or model is None:
                    X_train = history.drop(columns=[target_col])
                    y_train = history[target_col]
                    if y_train.nunique() < 2:
                        continue

                    model = RandomForestClassifier(**self.model_params)
                    model.fit(X_train, y_train)
                    self.models_by_asset[asset] = model
                    last_retrain_day = i

                current_row = clean_data.drop(columns=[target_col]).iloc[[i]]
                if current_row.isna().any(axis=None):
                    continue

                try:
                    prediction = model.predict_proba(current_row)[0][1]
                    asset_predictions.loc[dates[i]] = prediction
                except Exception as exc:
                    print(
                        f"  Erro ao prever {asset} em {dates[i].date()}: {exc}")

            asset_predictions = asset_predictions.reindex(full_dataset.index)
            missing_mask = asset_predictions.isna()
            if missing_mask.any():
                fallback_series = self._build_fallback_series(
                    asset, full_dataset)
                asset_predictions.loc[missing_mask] = fallback_series.loc[missing_mask]

            all_predictions[asset] = asset_predictions.fillna(0.5)

        predictions_df = pd.DataFrame(
            all_predictions, index=full_dataset.index)
        return predictions_df
