# src/backtester.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config


class Backtester:
    def __init__(self, prices_df, target_weights_df, transaction_cost_bps=5, slippage_bps=None):
        """
        Inicializa o backtester.

        prices_df: DataFrame com preços (index datetime, colunas tickers)
        target_weights_df: DataFrame de pesos-alvo (index datetime, colunas tickers)
        transaction_cost_bps: custo por unidade de turnover em basis points (bps)
        slippage_bps: slippage estimado em bps (se None, tenta ler de config.SLIPPAGE_BPS)
        """
        # store inputs
        self.prices = prices_df.copy()
        self.target_weights = target_weights_df.copy()

        # normalize costs (bps -> decimal)
        self.transaction_cost = float(transaction_cost_bps) / 10000.0
        if slippage_bps is None:
            slippage_bps = getattr(config, 'SLIPPAGE_BPS', 0)
        self.slippage = float(slippage_bps) / 10000.0

        # aligned asset list (intersection)
        self.asset_list = [
            c for c in self.target_weights.columns if c in self.prices.columns]

        # placeholders to be filled by run_backtest
        self.results = pd.Series(dtype=float)
        self.turnover = pd.Series(dtype=float)
        self.metrics = {}

    def run_backtest(self):
        """
        Executa um backtest simples: assume rebalance diário para os pesos alvo.
        Calcula turnover como 0.5 * sum(|w_t - w_{t-1}|) e aplica custos (transaction_cost)
        na data do rebalance.
        """
        # prepare returns and align weights
        returns = self.prices[self.asset_list].pct_change().fillna(0)

        # Alinhar pesos com o índice de preços
        weights = self.target_weights[self.asset_list].reindex(returns.index)

        # Se houver NaNs no início (antes das previsões começarem),
        # usar estratégia de peso igual (equal-weight) como fallback
        initial_nan_mask = weights.isna().all(axis=1)
        if initial_nan_mask.any():
            # Preencher os primeiros NaNs com equal-weight
            equal_weight = 1.0 / len(self.asset_list)
            weights.loc[initial_nan_mask] = equal_weight

        # Preencher qualquer NaN restante para frente
        weights = weights.fillna(method='ffill').fillna(0)

        # init series/dataframes
        portfolio_returns = pd.Series(index=returns.index, dtype=float)
        turnover = pd.Series(index=returns.index, dtype=float)

        prev_weights = pd.Series(0.0, index=self.asset_list)

        for t in returns.index:
            target_w = weights.loc[t]
            # compute trades and turnover
            trades = (target_w - prev_weights).abs().sum() / 2.0
            turnover.loc[t] = trades

            # cost and slippage
            cost = trades * self.transaction_cost
            slippage_cost = trades * self.slippage

            # portfolio return uses previous day's executed weights
            gross_return = (prev_weights * returns.loc[t]).sum()
            net_return = gross_return - cost - slippage_cost

            portfolio_returns.loc[t] = net_return

            # update prev_weights
            prev_weights = target_w

        # save results
        self.results = portfolio_returns.fillna(0)
        self.turnover = turnover.fillna(0)
        print("Backtest concluído.")
        return self.results

    def calculate_performance_metrics(self):
        """
        Calcula as principais métricas de desempenho.
        """
        # Verifica se temos resultados para calcular
        if not hasattr(self, 'results') or len(self.results) == 0:
            print("Aviso: Não há resultados disponíveis para calcular métricas.")
            return pd.DataFrame()

        # Garante que o índice está em formato datetime
        if not isinstance(self.results.index, pd.DatetimeIndex):
            self.results.index = pd.to_datetime(self.results.index)

        equity_curve = (1 + self.results).cumprod()

        # Básicos: CAGR e returns
        total_return = equity_curve.iloc[-1] - 1
        n_years = (equity_curve.index[-1] -
                   equity_curve.index[0]).days / 365.25
        cagr = (equity_curve.iloc[-1])**(1 /
                                         n_years) - 1 if n_years > 0 else np.nan

        # Volatilidade Anualizada (diária -> anual)
        volatility = self.results.std(ddof=1) * np.sqrt(252)

        # Sharpe Ratio (assumindo taxa livre de risco de 0)
        sharpe_ratio = (self.results.mean() * 252) / (self.results.std(ddof=1)
                                                      * np.sqrt(252)) if volatility != 0 else np.nan

        # Maximum Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = equity_curve / rolling_max - 1
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = cagr / \
            abs(max_drawdown) if max_drawdown != 0 else np.nan

        # VaR e CVaR (95%) - usando perdas: VaR positivo representa perda
        alpha = 0.05
        var_95 = -np.percentile(self.results.dropna(), 100 *
                                alpha) if len(self.results.dropna()) > 0 else np.nan
        cvar_95 = -self.results[self.results <= np.percentile(self.results.dropna(
        ), 100 * alpha)].mean() if len(self.results.dropna()) > 0 else np.nan

        # Sortino Ratio (downside risk)
        target = 0.0
        negative_returns = self.results[self.results < target]
        downside_deviation = np.sqrt(
            (negative_returns**2).mean() * 252) if len(negative_returns) > 0 else np.nan
        sortino = ((self.results.mean() * 252) - 0.0) / \
            downside_deviation if downside_deviation and downside_deviation != 0 else np.nan

        # Rachev Ratio (tail gain / tail loss) - usar 95/5
        up_tail_pct = 95
        down_tail_pct = 5
        if len(self.results.dropna()) > 0:
            up_thresh = np.percentile(self.results.dropna(), up_tail_pct)
            down_thresh = np.percentile(self.results.dropna(), down_tail_pct)
            up_tail_mean = self.results[self.results >= up_thresh].mean()
            down_tail_mean = - \
                self.results[self.results <=
                             down_thresh].mean()  # positive loss
            rachev = (
                up_tail_mean / down_tail_mean) if (down_tail_mean and down_tail_mean != 0) else np.nan
        else:
            rachev = np.nan

        # Hit Rate (taxa de dias positivos)
        hit_rate = (self.results > 0).mean()

        # Turnover e custos
        avg_annual_turnover = self.turnover.mean() * 252
        total_turnover = self.turnover.sum()
        total_transaction_costs = (self.turnover * self.transaction_cost).sum()

        # Turnover por período (mensal / trimestral)
        try:
            turnover_monthly = self.turnover.resample('M').sum()
            avg_monthly_turnover = turnover_monthly.mean()
        except Exception:
            avg_monthly_turnover = np.nan

        try:
            turnover_quarter = self.turnover.resample('Q').sum()
            avg_quarter_turnover = turnover_quarter.mean()
        except Exception:
            avg_quarter_turnover = np.nan

        # Slippage estimado (usando slippage param em bps)
        # slippage param default definido no __init__ (self.slippage)
        estimated_total_slippage = (self.turnover * self.slippage).sum()
        estimated_annual_slippage = self.turnover.mean() * 252 * self.slippage

        # Beta e Information Ratio (comparar com benchmark se disponível)
        benchmark_ticker = getattr(config, 'BENCHMARK_TICKER', None)
        beta = np.nan
        info_ratio = np.nan
        rolling_info_ratio = {}
        if benchmark_ticker is not None and benchmark_ticker in self.prices.columns:
            # alinhar índices
            bench_returns = self.prices[benchmark_ticker].pct_change().reindex(
                self.results.index).fillna(0)
            # Beta: cov(portfolio, bench) / var(bench)
            cov = np.cov(self.results.fillna(0), bench_returns.fillna(0))[0, 1]
            var_bench = np.var(bench_returns.fillna(0))
            beta = cov / var_bench if var_bench != 0 else np.nan

            # Information Ratio: annualized mean(excess)/std(excess)
            excess = self.results.fillna(0) - bench_returns.fillna(0)
            ir = (excess.mean() * 252) / (excess.std(ddof=1) *
                                          np.sqrt(252)) if excess.std(ddof=1) != 0 else np.nan
            info_ratio = ir

            # Rolling Information Ratio for multiple windows (in days)
            windows = [21, 63, 126, 252]
            for w in windows:
                try:
                    roll_excess_mean = excess.rolling(window=w).mean() * 252
                    roll_excess_std = excess.rolling(
                        window=w).std(ddof=1) * np.sqrt(252)
                    # compute last available rolling IR
                    last_ir = (roll_excess_mean / roll_excess_std).iloc[-1]
                except Exception:
                    last_ir = np.nan
                rolling_info_ratio[f'Information Ratio {w}d'] = last_ir

        # Cumulative Return (final equity - 1)
        cumulative_return = total_return

        # Hit Rate / Accuracy (já calculado como hit_rate)

        metrics = {
            'CAGR': cagr,
            'Cumulative Return': cumulative_return,
            'Annualized Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino,
            'Maximum Drawdown (MDD)': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'VaR 95%': var_95,
            'CVaR 95%': cvar_95,
            'Rachev Ratio (95/5)': rachev,
            'Beta': beta,
            'Information Ratio': info_ratio,
            'Hit Rate': hit_rate,
            'Average Annual Turnover': avg_annual_turnover,
            'Total Turnover': total_turnover,
            'Total Transaction Costs': total_transaction_costs,
            'Avg Monthly Turnover': avg_monthly_turnover,
            'Avg Quarterly Turnover': avg_quarter_turnover,
            'Estimated Total Slippage': estimated_total_slippage,
            'Estimated Annual Slippage': estimated_annual_slippage
        }

        # add rolling IRs
        for k, v in rolling_info_ratio.items():
            metrics[k] = v

        # Armazenar e retornar como DataFrame numérico
        self.metrics = metrics
        df_metrics = pd.DataFrame.from_dict(
            metrics, orient='index', columns=['Value'])
        return df_metrics

    def get_backtest_results_for_report(self, start_date=None, end_date=None):
        """
        Retorna um dicionário com todos os resultados necessários para gerar o relatório.

        Args:
            start_date: Data inicial do período (opcional)
            end_date: Data final do período (opcional)

        Returns:
            Dict com: equity_curve, daily_returns, metrics, weights_history, trade_log
        """
        # Equity curve
        equity_curve = (1 + self.results).cumprod()

        # Daily returns
        daily_returns = self.results.copy()

        # Período
        if start_date is None:
            start_date = self.results.index[0]
        if end_date is None:
            end_date = self.results.index[-1]

        period_str = f"{start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}"

        # Métricas calculadas
        metrics = self.metrics.copy() if self.metrics else {}

        # Converter para formato mais legível
        metrics_clean = {
            'CAGR': metrics.get('CAGR', 0),
            'Total_Return': metrics.get('Cumulative Return', 0),
            'Annual_Return': daily_returns.mean() * 252,
            'Annual_Volatility': metrics.get('Annualized Volatility', 0),
            'Sharpe_Ratio': metrics.get('Sharpe Ratio', 0),
            'Sortino_Ratio': metrics.get('Sortino Ratio', 0),
            'Max_Drawdown': metrics.get('Maximum Drawdown (MDD)', 0),
            'Calmar_Ratio': metrics.get('Calmar Ratio', 0),
            'VaR_95': metrics.get('VaR 95%', 0),
            'CVaR_95': metrics.get('CVaR 95%', 0),
            'Rachev_Ratio': metrics.get('Rachev Ratio (95/5)', 0),
            'Beta': metrics.get('Beta', 0),
            'Information_Ratio': metrics.get('Information Ratio', 0),
            'Hit_Rate': metrics.get('Hit Rate', 0),
            'Annual_Turnover': metrics.get('Average Annual Turnover', 0),
            'Daily_Turnover': self.turnover.mean(),
            'Transaction_Costs': metrics.get('Total Transaction Costs', 0),
            'Num_Rebalances': (self.turnover > 0).sum(),
            'Best_Month': daily_returns.resample('M').sum().max() if len(daily_returns) > 0 else 0,
            'Worst_Month': daily_returns.resample('M').sum().min() if len(daily_returns) > 0 else 0,
        }

        return {
            'equity_curve': equity_curve,
            'daily_returns': daily_returns,
            'metrics': metrics_clean,
            'weights_history': self.target_weights,
            'trade_log': pd.DataFrame({'Turnover': self.turnover}),
            'period': period_str,
            'start_date': start_date,
            'end_date': end_date,
        }

    def plot_performance(self):
        """
        Plota a curva de capital da estratégia vs. um benchmark.
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        # Curva de capital da estratégia
        equity_curve = (1 + self.results).cumprod()
        ax.plot(equity_curve.index, equity_curve,
                label='Cassandra Strategy', color='blue', lw=2)

        # Curva de capital do benchmark (S&P 500)
        benchmark_returns = self.prices.pct_change().fillna(0)
        benchmark_equity = (1 + benchmark_returns).cumprod()
        ax.plot(benchmark_equity.index, benchmark_equity,
                label='Benchmark (S&P 500)', color='gray', ls='--')

        ax.set_title(
            'Performance da Estratégia Cassandra vs. Benchmark', fontsize=16)
        ax.set_ylabel('Crescimento do Capital (Log Scale)')
        ax.set_xlabel('Data')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        plt.show()
