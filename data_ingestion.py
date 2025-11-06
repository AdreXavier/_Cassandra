# src/data_ingestion.py
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from pathlib import Path
import config


def _load_cached_dataset(cache_path: Path):
    """Carrega o dataset em cache e valida se contém dados utilizáveis."""
    if not cache_path.exists():
        return None

    try:
        cached_data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    except Exception as exc:
        print(
            f"Aviso: não foi possível ler o cache local ({exc}). Ignorando cache.")
        return None

    if cached_data.empty:
        print(
            f"Aviso: cache encontrado em {cache_path}, mas está vazio. Ignorando cache.")
        return None

    asset_columns = [
        col for col in config.ASSET_UNIVERSE if col in cached_data.columns]
    if not asset_columns:
        print("Aviso: cache localizado, porém sem nenhum ativo do universo esperado. Ignorando cache.")
        return None

    return cached_data


def fetch_market_data(tickers, start_date, end_date):
    """
    Baixa dados de mercado do Yahoo Finance.
    Esta versão é robusta à mudança de default do yfinance para auto_adjust=True.
    """
    print("Baixando dados de mercado...")
    try:
        # O yfinance agora usa auto_adjust=True por padrão, o que retorna
        # os preços ajustados na coluna 'Close' e remove 'Adj Close'.
        data = yf.download(tickers, start=start_date,
                           end=end_date, progress=False)

        if data.empty:
            print(
                "Erro: Nenhum dado foi retornado pelo yfinance. Verifique os tickers e o período.")
            return None

        # Lida com a estrutura de multi-index para múltiplos tickers
        # e a ausência de 'Adj Close' com o novo padrão.
        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
        elif 'Close' in data.columns:
            print(
                "Usando a coluna 'Close' pois 'Adj Close' não foi encontrada (padrão yfinance).")
            price_data = data['Close']
        else:
            print(
                f"Erro: Não foi possível encontrar as colunas 'Adj Close' ou 'Close' nos dados baixados. Colunas disponíveis: {data.columns}")
            return None

        price_data.index = pd.to_datetime(price_data.index)

        # Se apenas um ticker for baixado, yfinance não cria um DataFrame, mas uma Series.
        # Garantimos que seja sempre um DataFrame.
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers)

        print("Dados de mercado baixados com sucesso.")
        return price_data
    except Exception as e:
        print(f"Erro inesperado ao baixar dados de mercado: {e}")
        return None


def fetch_macro_data(indicators, start_date, end_date):
    """
    Busca dados macroeconômicos do FRED.

    Args:
        indicators (dict): Dicionário com códigos do FRED e nomes desejados.
        start_date (str): Data de início.
        end_date (str): Data de fim.

    Returns:
        pd.DataFrame: DataFrame com os dados macroeconômicos.
    """
    print("Buscando dados macroeconômicos do FRED...")
    try:
        macro_data = web.DataReader(
            list(indicators.keys()), 'fred', start_date, end_date)
        macro_data = macro_data.rename(columns=indicators)
        macro_data.index = pd.to_datetime(macro_data.index)
        print("Dados macroeconômicos buscados com sucesso.")
        return macro_data
    except Exception as e:
        print(f"Erro ao buscar dados macroeconômicos: {e}")
        return None


def get_unified_dataset(cache_filename: str = "unified_dataset.csv"):
    """
    Orquestra o download e a unificação de todos os dados necessários.

    Se a requisição online falhar, tenta carregar um cache local salvo em data/unified_dataset.csv
    """
    cache_path = Path(__file__).resolve().parents[1] / "data" / cache_filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    market_data = fetch_market_data(
        config.ALL_TICKERS, config.START_DATE, config.END_DATE)
    macro_data = fetch_macro_data(
        config.MACRO_INDICATORS, config.START_DATE, config.END_DATE)

    if market_data is None or macro_data is None:
        cached_data = _load_cached_dataset(cache_path)
        if cached_data is not None:
            print(
                "Aviso: falha ao baixar dados online. Dataset carregado do cache local.")
            return cached_data
        raise ConnectionError(
            "Falha ao obter os dados necessários e nenhum cache local válido foi encontrado."
        )

    # Unifica os datasets com base no índice de datas (dias úteis de mercado)
    unified_data = market_data.join(macro_data, how='left')

    # Tratamento de valores ausentes
    # Para dados macro, que são atualizados com menos frequência, usamos forward fill.
    unified_data = unified_data.ffill()

    # Para preços de ativos, valores ausentes podem indicar feriados.
    # Por simplicidade, removemos quaisquer dias que ainda tenham NaNs após o ffill inicial.
    unified_data = unified_data.dropna()

    # Garante que o dataset final seja utilizável; caso contrário, tenta recorrer ao cache
    if unified_data.empty:
        print("Aviso: dataset unificado ficou vazio após limpeza. Tentando carregar cache local.")
        cached_data = _load_cached_dataset(cache_path)
        if cached_data is not None:
            return cached_data
        raise ValueError(
            "Dataset unificado ficou vazio e nenhum cache válido foi encontrado. Verifique as fontes de dados ou exclua o cache inválido."
        )

    asset_columns = [
        col for col in config.ASSET_UNIVERSE if col in unified_data.columns]
    if not asset_columns:
        print("Aviso: dataset baixado não contém os ativos esperados. Tentando carregar cache local.")
        cached_data = _load_cached_dataset(cache_path)
        if cached_data is not None:
            return cached_data
        raise ValueError(
            "Dataset unificado não contém nenhum dos ativos do universo configurado e nenhum cache válido está disponível."
        )

    print("Dataset unificado e limpo criado com sucesso.")
    print(f"Shape final do dataset: {unified_data.shape}")

    # Salvar cache local para uso offline futuro
    try:
        unified_data.to_csv(cache_path)
        print(f"Dataset salvo em cache para uso offline: {cache_path}")
    except Exception as exc:
        print(f"Aviso: não foi possível salvar o cache local ({exc})")

    return unified_data


# Alias para compatibilidade
load_data = get_unified_dataset
