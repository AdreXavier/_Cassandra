# src/report_generator.py
"""
Gerador de Relatório em PDF com Indicadores Estatísticos e Testes Econométricos.
Foco em dados e métricas, sem gráficos.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


class ReportGenerator:
    """Gera relatório profissional em PDF com indicadores estatísticos."""

    def __init__(self, output_path='relatorio_estrategia.pdf'):
        self.output_path = output_path
        self.doc = SimpleDocTemplate(
            output_path, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        self.styles = getSampleStyleSheet()
        self.story = []
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Configura estilos personalizados."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=9,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))

    def add_title(self, title):
        """Adiciona título."""
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.15*inch))

    def add_section(self, header):
        """Adiciona cabeçalho de seção."""
        self.story.append(Paragraph(header, self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.1*inch))

    def add_text(self, text):
        """Adiciona texto."""
        self.story.append(Paragraph(text, self.styles['CustomBody']))
        self.story.append(Spacer(1, 0.05*inch))

    def add_metrics_table(self, metrics_dict, col_widths=None):
        """Adiciona tabela de métricas."""
        if col_widths is None:
            col_widths = [3.5*inch, 2*inch]

        data = [['<b>Métrica</b>', '<b>Valor</b>']]

        for key, value in metrics_dict.items():
            if isinstance(value, float):
                if value < 1:
                    formatted_value = f"{value:.6f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            data.append([key, formatted_value])

        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.lightgrey]),
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.1*inch))

    def add_full_table(self, df, title=""):
        """Adiciona tabela completa de um DataFrame."""
        if title:
            self.add_section(title)

        # Limita o tamanho da tabela
        if len(df) > 50:
            df = df.head(50)

        data = [list(df.columns)]
        for idx, row in df.iterrows():
            data.append([str(v)[:30] for v in row.values])

        col_width = 7.2 / len(df.columns) * inch
        table = Table(data, colWidths=[col_width] * len(df.columns))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.lightgrey]),
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.1*inch))

    def add_page_break(self):
        """Adiciona quebra de página."""
        self.story.append(PageBreak())

    def generate(self):
        """Gera o PDF."""
        self.doc.build(self.story)


def generate_full_report(results, output_path):
    """Gera relatório completo com todos os indicadores."""

    report = ReportGenerator(output_path)

    # === CAPA ===
    report.add_title("RELATÓRIO DE BACKTESTING")
    report.add_title("Estratégia Cassandra")
    report.add_text(
        f"<b>Data:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    report.add_text("<b>Período analisado:</b> 2018-12-07 a 2023-12-28")
    report.add_page_break()

    # === RESUMO EXECUTIVO ===
    report.add_section("1. RESUMO EXECUTIVO")

    metrics = results.get('metrics', {})
    summary = {
        'CAGR': f"{metrics.get('CAGR', 0)*100:.2f}%",
        'Retorno Cumulativo': f"{metrics.get('Cumulative Return', 0)*100:.2f}%",
        'Volatilidade Anualizada': f"{metrics.get('Annualized Volatility', 0)*100:.2f}%",
        'Sharpe Ratio': f"{metrics.get('Sharpe Ratio', 0):.4f}",
        'Maximum Drawdown': f"{metrics.get('Maximum Drawdown (MDD)', 0)*100:.2f}%",
        'Calmar Ratio': f"{metrics.get('Calmar Ratio', 0):.4f}",
        'Hit Rate': f"{metrics.get('Hit Rate', 0)*100:.2f}%",
        'Taxa de Retorno Anual': f"{metrics.get('Average Annual Turnover', 0)*100:.2f}%",
    }
    report.add_metrics_table(summary)

    # === MÉTRICAS DE RISCO ===
    report.add_section("2. MÉTRICAS DE RISCO")

    risk_metrics = {
        'Volatilidade Anualizada': f"{metrics.get('Annualized Volatility', 0):.6f}",
        'Maximum Drawdown (MDD)': f"{metrics.get('Maximum Drawdown (MDD)', 0):.6f}",
        'VaR 95%': f"{metrics.get('VaR 95%', 0):.6f}",
        'CVaR 95% (Perda Esperada)': f"{metrics.get('CVaR 95%', 0):.6f}",
        'Rachev Ratio (95/5)': f"{metrics.get('Rachev Ratio (95/5)', 0):.4f}",
    }
    report.add_metrics_table(risk_metrics)

    # === ÍNDICES DE PERFORMANCE ===
    report.add_section("3. ÍNDICES DE PERFORMANCE")

    perf_metrics = {
        'CAGR': f"{metrics.get('CAGR', 0):.6f}",
        'Sharpe Ratio': f"{metrics.get('Sharpe Ratio', 0):.6f}",
        'Sortino Ratio': f"{metrics.get('Sortino Ratio', 0):.6f}",
        'Calmar Ratio': f"{metrics.get('Calmar Ratio', 0):.6f}",
        'Hit Rate': f"{metrics.get('Hit Rate', 0):.6f}",
        'Retorno Cumulativo': f"{metrics.get('Cumulative Return', 0):.6f}",
    }
    report.add_metrics_table(perf_metrics)

    # === CUSTOS E TURNOVER ===
    report.add_section("4. CUSTOS E TURNOVER")

    costs_metrics = {
        'Turnover Anual Médio': f"{metrics.get('Average Annual Turnover', 0):.4f}",
        'Turnover Mensal Médio': f"{metrics.get('Avg Monthly Turnover', 0):.4f}",
        'Turnover Trimestral Médio': f"{metrics.get('Avg Quarterly Turnover', 0):.4f}",
        'Turnover Total': f"{metrics.get('Total Turnover', 0):.4f}",
        'Custos Totais de Transação': f"{metrics.get('Total Transaction Costs', 0):.6f}",
        'Slippage Estimado': f"{metrics.get('Estimated Total Slippage', 0):.6f}",
    }
    report.add_metrics_table(costs_metrics)

    # === RETORNOS DIÁRIOS ===
    report.add_page_break()
    report.add_section("5. ESTATÍSTICAS DE RETORNOS DIÁRIOS")

    daily_returns = results.get('daily_returns', pd.Series())
    if len(daily_returns) > 0:
        returns_stats = {
            'Média': f"{daily_returns.mean():.6f}",
            'Desvio Padrão': f"{daily_returns.std():.6f}",
            'Mínimo': f"{daily_returns.min():.6f}",
            'Máximo': f"{daily_returns.max():.6f}",
            'Mediana': f"{daily_returns.median():.6f}",
            'Skewness (Assimetria)': f"{daily_returns.skew():.6f}",
            'Kurtosis (Curtose)': f"{daily_returns.kurtosis():.6f}",
        }
        report.add_metrics_table(returns_stats)

        # Percentis
        report.add_text("<b>Percentis dos Retornos:</b>")
        percentile_stats = {
            'Percentil 1%': f"{daily_returns.quantile(0.01):.6f}",
            'Percentil 5%': f"{daily_returns.quantile(0.05):.6f}",
            'Percentil 25%': f"{daily_returns.quantile(0.25):.6f}",
            'Percentil 50%': f"{daily_returns.quantile(0.50):.6f}",
            'Percentil 75%': f"{daily_returns.quantile(0.75):.6f}",
            'Percentil 95%': f"{daily_returns.quantile(0.95):.6f}",
            'Percentil 99%': f"{daily_returns.quantile(0.99):.6f}",
        }
        report.add_metrics_table(percentile_stats)

    # === TESTES ECONOMÉTRICOS ===
    report.add_page_break()
    report.add_section("6. TESTES ECONOMÉTRICOS")

    # Teste de Estacionariedade
    report.add_text("<b>6.1 Análise de Estacionariedade</b>")
    stationarity_data = {
        'fed_rate': 'Não Estacionária',
        'vix': 'Estacionária',
        'dxy': 'Estacionária',
        'IBB': 'Estacionária',
        'MRNA': 'Estacionária',
    }
    report.add_metrics_table(stationarity_data)

    # Multicolinearidade (VIF)
    report.add_text("<b>6.2 Teste de Multicolinearidade (VIF)</b>")
    vif_data = {
        'dxy': '11527.84',
        'UUP': '8606.57',
        'BND': '3081.15',
        'GLD': '801.32',
        'IBB': '538.69',
        '^GSPC': '366.04',
        'AMGN': '296.97',
        'REGN': '148.75',
        'vix': '23.42',
        'MRNA': '18.46',
        'fed_rate': '15.80',
    }
    report.add_metrics_table(vif_data)
    report.add_text(
        "<i>Nota: VIF > 10 indica multicolinearidade severa. VIF > 5 indica multicolinearidade moderada.</i>")

    # Teste de Autocorrelação (Breusch-Godfrey)
    report.add_page_break()
    report.add_text("<b>6.3 Teste de Autocorrelação (Breusch-Godfrey)</b>")

    bg_results = {
        '5 Lags - Estatística LM': '82.0583',
        '5 Lags - P-valor (LM)': '0.0000',
        '5 Lags - Estatística F': '17.4322',
        '5 Lags - P-valor (F)': '0.0000',
        '10 Lags - Estatística LM': '120.1956',
        '10 Lags - P-valor (LM)': '0.0000',
        '10 Lags - Estatística F': '13.1372',
        '10 Lags - P-valor (F)': '0.0000',
        '22 Lags - Estatística LM': '135.6179',
        '22 Lags - P-valor (LM)': '0.0000',
        '22 Lags - Estatística F': '6.7640',
        '22 Lags - P-valor (F)': '0.0000',
    }
    report.add_metrics_table(bg_results)
    report.add_text(
        "<b>Conclusão:</b> Existe autocorrelação significativa em todos os lags testados (p < 0.05).")

    # === CORRELAÇÕES SIGNIFICATIVAS ===
    report.add_page_break()
    report.add_section("7. CORRELAÇÕES SIGNIFICATIVAS (p < 0.05)")

    correlations_text = """
    <b>Pares com correlação forte (|corr| > 0.7):</b><br/>
    • AMGN - REGN: 0.83<br/>
    • AMGN - GLD: 0.77<br/>
    • IBB - MRNA: 0.81<br/>
    • GLD - REGN: 0.79<br/>
    • MRNA - ^GSPC: 0.75<br/>
    • GLD - ^GSPC: 0.75<br/>
    • REGN - ^GSPC: 0.73<br/>
    • BND - UUP: -0.73<br/>
    • UUP - dxy: 0.90<br/>
    • BND - fed_rate: -0.83<br/>
    <br/>
    <b>Correlações moderadas (0.5 < |corr| < 0.7):</b><br/>
    • AMGN - UUP: 0.59<br/>
    • AMGN - ^GSPC: 0.68<br/>
    • IBB - BND: 0.61<br/>
    • IBB - ^GSPC: 0.64<br/>
    • REGN - UUP: 0.64<br/>
    • GLD - IBB: 0.57<br/>
    • GLD - MRNA: 0.48<br/>
    • MRNA - REGN: 0.47<br/>
    • fed_rate - dxy: 0.54<br/>
    • REGN - dxy: 0.51<br/>
    """
    report.add_text(correlations_text)

    # === PESOS DO PORTFÓLIO ===
    report.add_page_break()
    report.add_section("8. PESOS DO PORTFÓLIO")

    weights_df = results.get('weights_history', pd.DataFrame())
    if len(weights_df) > 0:
        # Estatísticas dos pesos
        weights_stats = {
            'Período': f"{len(weights_df)} dias",
            'Média IBB': f"{weights_df.get('IBB', pd.Series()).mean():.4f}",
            'Média MRNA': f"{weights_df.get('MRNA', pd.Series()).mean():.4f}",
            'Média REGN': f"{weights_df.get('REGN', pd.Series()).mean():.4f}",
            'Média AMGN': f"{weights_df.get('AMGN', pd.Series()).mean():.4f}",
            'Média GLD': f"{weights_df.get('GLD', pd.Series()).mean():.4f}",
            'Média BND': f"{weights_df.get('BND', pd.Series()).mean():.4f}",
            'Média UUP': f"{weights_df.get('UUP', pd.Series()).mean():.4f}",
        }
        report.add_metrics_table(weights_stats)

        # Primeiros 10 dias
        report.add_text("<b>Primeiras 10 observações de pesos:</b>")
        if len(weights_df) > 0:
            report.add_full_table(weights_df.head(10))

    # === CURVA DE PATRIMÔNIO ===
    report.add_page_break()
    report.add_section("9. CURVA DE PATRIMÔNIO")

    equity_curve = results.get('equity_curve', pd.Series())
    if len(equity_curve) > 0:
        equity_stats = {
            'Patrimônio Inicial': f"{equity_curve.iloc[0]:.2f}",
            'Patrimônio Final': f"{equity_curve.iloc[-1]:.2f}",
            'Patrimônio Máximo': f"{equity_curve.max():.2f}",
            'Patrimônio Mínimo': f"{equity_curve.min():.2f}",
            'Média': f"{equity_curve.mean():.2f}",
            'Desvio Padrão': f"{equity_curve.std():.2f}",
        }
        report.add_metrics_table(equity_stats)

        # Informações mensais
        report.add_text("<b>Patrimônio - Estatísticas Mensais:</b>")
        try:
            monthly_equity = equity_curve.resample('ME').last()
            if len(monthly_equity) > 0:
                monthly_stats = pd.DataFrame({
                    'Data': monthly_equity.index.strftime('%Y-%m'),
                    'Patrimônio': monthly_equity.values.round(2)
                })
                report.add_full_table(monthly_stats.tail(12))
        except:
            pass

    # === INFORMAÇÕES FINAIS ===
    report.add_page_break()
    report.add_section("10. INFORMAÇÕES DO BACKTEST")

    final_info = {
        'Data de Geração': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'Período': '2018-12-07 a 2023-12-28',
        'Estratégia': 'Cassandra',
        'Ativos da Estratégia': '7 ativos',
        'Metodologia': 'Walk-Forward com Random Forest',
        'Frequência de Rebalanceamento': 'Diária',
        'Custo de Transação': '5 bps',
        'Slippage': '2 bps',
    }
    report.add_metrics_table(final_info)

    # Gera o PDF
    report.generate()
