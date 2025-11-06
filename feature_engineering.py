# src/feature_engineering.py
import pandas as pd
import numpy as np
import config
# src/feature_engineering.py
import pandas as pd
import numpy as np
import config
from statsmodels.stats.outliers_influence import variance_inflation_factor


def diagnostico_vif(df_features, threshold=10.0):
    """
    Calcula e exibe o Fator de Inflação de Variância (VIF)
    para as features, identificando multicolinearidade severa (VIF >= threshold).
    Retorna um DataFrame com os valores de VIF.
    """
    X = df_features.select_dtypes(include=np.number).replace(
        [np.inf, -np.inf], np.nan).dropna()

    # Remove colunas que podem ser targets ou não explicativas
    features_only = [col for col in X.columns if not col.endswith('_target')]
    X_features = X[features_only]

# src/feature_engineering.py


def diagnostico_vif(df_features, threshold=10.0):
    """
    Calcula e exibe o Fator de Inflação de Variância (VIF)
    para as features, identificando multicolinearidade severa (VIF >= threshold).
    Retorna um DataFrame com os valores de VIF.
    """
    X = df_features.select_dtypes(include=np.number).replace(
        [np.inf, -np.inf], np.nan).dropna()

    # Remove colunas que podem ser targets ou não explicativas
    features_only = [col for col in X.columns if not col.endswith('_target')]
    X_features = X[features_only]

    if X_features.empty:
        print("Aviso: Nenhuma feature numérica válida para calcular o VIF.")
        return pd.DataFrame()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_features.columns

    try:
        vif_data["VIF"] = [variance_inflation_factor(X_features.values, i)
                           for i in range(len(X_features.columns))]
    except Exception:
        print("Erro ao calcular VIF (matriz singular ou dados insuficientes).")
        return pd.DataFrame()

    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    problemas = vif_data[vif_data['VIF'] >= threshold]

    if not problemas.empty:
        print(
            f"\nATENÇÃO: Multicolinearidade severa detectada (VIF >= {threshold}).")
        print("As features listadas precisam ser removidas ou tratadas:")
        print(problemas)

    return vif_data


def filter_and_correct_features(df, features_to_remove=None):
    """
    Aplica correções baseadas nos diagnósticos:
    - Estacionariedade para DFF
    - Remoção de features listadas (com cuidado com assets core)
    Retorna o DataFrame corrigido (sem NaNs finais).
    """
    if features_to_remove is None:
        features_to_remove = []

    features = df.copy()

    # Estacionariedade: diferenciar DFF se presente
    if 'DFF' in features.columns:
        print("Aplicando diferença em DFF para estacionariedade.")
        features['fed_rate_diff'] = features['DFF'].diff().ffill().bfill()
        features = features.drop(columns=['DFF'], errors='ignore')

    # Remover features explicitamente listadas
    if features_to_remove:
        # Remoções por grupos (respeitar dependências)
        grupo1 = [f for f in features_to_remove if f in ['dxy']]
        if grupo1:
            features = features.drop(columns=grupo1, errors='ignore')
            print(f"Features removidas (grupo1): {grupo1}")

        grupo2 = [f for f in features_to_remove if f in ['UUP', 'BND']]
        if grupo2:
            # Se dxy foi removido, podemos manter UUP
            if 'dxy' not in features.columns and 'UUP' in grupo2:
                grupo2.remove('UUP')
            features = features.drop(columns=grupo2, errors='ignore')
            print(f"Features removidas (grupo2): {grupo2}")

        grupo3 = [f for f in features_to_remove if f in ['fed_rate', 'vix']]
        if 'fed_rate' in grupo3 and 'fed_rate_diff' in features.columns:
            grupo3.remove('fed_rate')
        features = features.drop(columns=grupo3, errors='ignore')
        print(f"Features macro removidas: {grupo3}")

        # Cuidado com ativos core: nunca remover ambos
        grupo4 = [f for f in features_to_remove if f in ['IBB', 'MRNA']]
        if len(grupo4) == 2:
            if 'IBB' in features.columns and 'MRNA' in features.columns:
                ibb_na = features['IBB'].isna().sum()
                mrna_na = features['MRNA'].isna().sum()
                if ibb_na <= mrna_na:
                    grupo4.remove('IBB')
                else:
                    grupo4.remove('MRNA')
        features = features.drop(columns=grupo4, errors='ignore')
        print(f"Ativos removidos por correlação: {grupo4}")

    # Retornar com NaNs limpos (linha inteira removida se faltar dado)
    return features.dropna()


def add_macro_features(df):
    """
    Adiciona features macroeconômicas ao DataFrame (fed_rate, vix, dxy).
    """
    features = df.copy()

    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    if 'DFF' in features.columns:
        features['fed_rate_diff'] = features['DFF'].diff().ffill().bfill()
        features = features.drop(columns=['DFF'], errors='ignore')
    if 'VIXCLS' in features.columns:
        features['vix'] = features['VIXCLS']
    if 'DTWEXBGS' in features.columns:
        features['dxy'] = features['DTWEXBGS']

    return features


def add_sectoral_features(df, biotech_proxy, benchmark='^GSPC', window=60):
    """
    Cria features setoriais a partir do proxy passado (ex: IBB, MRNA).
    """
    features = df.copy()

    if biotech_proxy not in features.columns:
        raise ValueError(
            f"Ativo proxy {biotech_proxy} não encontrado nos dados!")

    features[biotech_proxy] = pd.to_numeric(
        features[biotech_proxy], errors='coerce')
    if features[biotech_proxy].isna().all():
        raise ValueError(f"Dados do ativo {biotech_proxy} estão todos nulos!")

    # Retornos multi-período
    for period in [1, 5, 21, 63]:
        col = f"{biotech_proxy}_ret_{period}d"
        features[col] = features[biotech_proxy].pct_change(period)

    # Volatilidades
    for period in [21, 63]:
        col = f"{biotech_proxy}_vol_{period}d"
        features[col] = features[biotech_proxy].pct_change().rolling(
            window=period, min_periods=int(period*0.8)).std()

    # Momentum anual suavizado
    features[f'{biotech_proxy}_momentum_252d'] = features[biotech_proxy].pct_change(
        252).rolling(window=21, min_periods=5).mean()

    # Força relativa vs benchmark (se disponível)
    if benchmark in features.columns:
        features[benchmark] = pd.to_numeric(
            features[benchmark], errors='coerce')
        if not features[benchmark].isna().all():
            for period in [21, 63, 252]:
                features[f'biotech_relative_strength_{period}d'] = features[biotech_proxy].pct_change(
                    period) - features[benchmark].pct_change(period)
            price = features[biotech_proxy]
            ma_short = price.rolling(window=21, min_periods=5).mean()
            ma_long = price.rolling(window=63, min_periods=15).mean()
            features['biotech_flow_proxy'] = (ma_short / ma_long) - 1
        else:
            for period in [21, 63, 252]:
                features[f'biotech_relative_strength_{period}d'] = 0
            features['biotech_flow_proxy'] = 0
    else:
        for period in [21, 63, 252]:
            features[f'biotech_relative_strength_{period}d'] = 0
        features['biotech_flow_proxy'] = 0

    # Clipping simples para reduzir outliers
    sector_feats = [c for c in features.columns if c.startswith(
        biotech_proxy) or c.startswith('biotech_')]
    for c in sector_feats:
        if features[c].dtype.kind in 'fc':
            q1 = features[c].quantile(0.01)
            q99 = features[c].quantile(0.99)
            features[c] = features[c].clip(lower=q1, upper=q99)

    return features


def add_technical_features(df, asset_list):
    """
    Adiciona features técnicas para cada ativo presente em asset_list.
    """
    features = df.copy()
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    for asset in asset_list:
        if asset in features.columns:
            try:
                features[asset] = pd.to_numeric(
                    features[asset], errors='coerce')
                features[f'{asset}_SMA_50'] = features[asset].rolling(
                    window=50, min_periods=10).mean()
                features[f'{asset}_EMA_20'] = features[asset].ewm(
                    span=20, adjust=False).mean()
                ema_12 = features[asset].ewm(span=12, adjust=False).mean()
                ema_26 = features[asset].ewm(span=26, adjust=False).mean()
                features[f'{asset}_MACD'] = ema_12 - ema_26
                features[f'{asset}_MACD_Signal'] = features[f'{asset}_MACD'].ewm(
                    span=9, adjust=False).mean()
            except Exception as e:
                print(
                    f"Erro ao processar features técnicas para {asset}: {str(e)}")
        else:
            # Não é crítico — apenas logar
            print(
                f"Aviso: Ativo {asset} não encontrado nos dados para features técnicas")

    return features


def add_target_variable(df, asset_list, horizon=20):
    """
    Cria a tabela de alvos binários para os ativos informados, usando df (preços).
    Retorna DataFrame com colunas {asset}_target.
    """
    targets = pd.DataFrame(index=df.index)
    for asset in asset_list:
        if asset not in df.columns:
            raise KeyError(
                f"Ativo {asset} não encontrado no DataFrame de preços ao criar alvos")
        future_returns = df[asset].pct_change(periods=horizon).shift(-horizon)
        targets[f'{asset}_target'] = (future_returns > 0).astype(int)
    return targets


def create_feature_set(unified_data):
    """
    Orquestra a criação de features e targets.
    - usa `unified_data` como fonte de preços originais
    - aplica correções e filtros nas features
    - gera alvos a partir dos preços originais e junta com features
    """
    print("\n=== Iniciando Criação de Features ===")
    data = unified_data.copy()

    # 1) selecionar ativos com dados suficientes (>30% válidos)
    available_assets = []
    for asset in config.ASSET_UNIVERSE.keys():
        if asset in data.columns:
            data[asset] = pd.to_numeric(data[asset], errors='coerce')
            if data[asset].notna().sum() > len(data) * 0.3:
                available_assets.append(asset)
    if not available_assets:
        raise ValueError("Nenhum ativo do universo com dados suficientes")

    print(f"Ativos disponíveis para análise: {available_assets}")

    # 2) escolher proxy setorial
    biotech_proxies = ['IBB', 'MRNA', 'REGN', 'AMGN']
    biotech_proxy = None
    for p in biotech_proxies:
        if p in data.columns and data[p].notna().sum() > len(data) * 0.3:
            biotech_proxy = p
            break
    if not biotech_proxy:
        raise ValueError(
            "Nenhum proxy setorial disponível com dados suficientes")
    print(f"Usando {biotech_proxy} como proxy setorial")

    # 3) criar features
    features = add_macro_features(data)
    try:
        features = add_sectoral_features(
            features, biotech_proxy, benchmark=config.BENCHMARK_TICKER)
    except Exception as e:
        print(f"Falha ao criar features setoriais: {e}")
        # continuar com o conjunto atual
    features = add_technical_features(features, available_assets)

    # 4) remover features problemáticas (VIF)
    vif_threshold = getattr(config, 'VIF_THRESHOLD', 10.0)
    features_to_remove = []
    vif_df = diagnostico_vif(features)
    if not vif_df.empty:
        high_vif_cols = vif_df[vif_df['VIF'] >=
                               vif_threshold]['feature'].tolist()
        features_to_remove = [
            c for c in high_vif_cols if c in config.VIF_HIGH_CORRELATION_FEATURES_TO_EXCLUDE]

    if features_to_remove:
        print(f"Removendo features problemáticas: {features_to_remove}")
        features = filter_and_correct_features(
            features, features_to_remove=features_to_remove)

    # 5) gerar alvos a partir dos preços originais e juntar
    valid_assets_for_targets = [
        a for a in available_assets if a in data.columns and data[a].notna().sum() > len(data) * 0.2]
    if not valid_assets_for_targets:
        raise ValueError("Nenhum ativo com dados suficientes para criar alvos")

    targets = add_target_variable(data, valid_assets_for_targets)

    # unir (inner) para assegurar alinhamento
    final = features.join(targets, how='inner')
    final = final.dropna()

    if final.empty:
        raise ValueError("Dataset final vazio após join de features e alvos")

    print(f"Conjunto final criado: {final.shape}")
    return final
    return final
