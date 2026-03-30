"""
ANOMALY Command

Display anomaly detection for a symbol.
Usage: ANOMALY <symbol>
"""

from typing import List

from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import MLCommandHandler
from src.tui.app import CommandResult, PythiaTerminal
from src.ml.anomaly import AnomalyDetector


class AnomalyCommand(MLCommandHandler):
    """Display anomaly detection for a stock."""
    
    @property
    def name(self) -> str:
        return "ANOMALY"
    
    @property
    def aliases(self) -> List[str]:
        return ["ANOMALY", "ANO", "ANOM"]
    
    @property
    def description(self) -> str:
        return "Display anomaly detection in price and volume"
    
    @property
    def usage(self) -> str:
        return "ANOMALY <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        return await self._show_anomalies(terminal, symbol)
    
    async def _show_anomalies(self, terminal: PythiaTerminal, symbol: str) -> CommandResult:
        """Show anomaly detection analysis."""
        try:
            detector = AnomalyDetector()
            anomalies = detector.detect_all_anomalies(symbol)
            
            if 'error' in anomalies:
                return CommandResult(False, f"Error: {anomalies['error']}")
            
            summary = anomalies.get('summary', {})
            price = anomalies.get('price_anomalies', [])
            volume = anomalies.get('volume_anomalies', [])
            
            table = Table(title=f"⚠️ {symbol} Anomaly Detection")
            table.add_column("Date", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Direction", style="magenta")
            table.add_column("Magnitude", style="green", justify="right")
            table.add_column("Severity", style="red")
            
            for p in price[:5]:
                direction = "↑" if p.get('direction') == 'up' else "↓"
                table.add_row(
                    p.get('date', 'N/A'),
                    'Price',
                    direction,
                    f"{p.get('magnitude', 0):.1f}%",
                    p.get('severity', 'N/A').upper()
                )
            
            for v in volume[:5]:
                direction = "↑" if v.get('price_change', 0) > 0 else "↓"
                table.add_row(
                    v.get('date', 'N/A'),
                    'Volume',
                    direction,
                    f"{v.get('volume_ratio', 0):.1f}x",
                    v.get('severity', 'N/A').upper()
                )
            
            summary_table = Table(show_header=False)
            summary_table.add_column("Field", style="cyan")
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Total Anomalies", str(summary.get('total_anomalies', 0)))
            summary_table.add_row("Price", str(summary.get('price_anomalies', 0)))
            summary_table.add_row("Volume", str(summary.get('volume_anomalies', 0)))
            summary_table.add_row("Volatility", str(summary.get('volatility_anomalies', 0)))
            summary_table.add_row("Gaps", str(summary.get('gap_anomalies', 0)))
            
            risk_level = summary.get('risk_level', 'unknown').upper()
            risk_style = "bold red" if risk_level == "HIGH" else "bold yellow" if risk_level == "MEDIUM" else "bold green"
            
            summary_table.add_row("Risk Level", f"[{risk_style}]{risk_level}[/{risk_style}]")
            summary_table.add_row("Recent Activity", summary.get('recent_activity', 'N/A').title())
            
            panel = Panel(summary_table, title=f"[bold cyan]{symbol} Anomaly Summary[/bold cyan]")
            
            terminal.current_symbol = symbol
            
            return CommandResult(success=True, message="", panel=panel, table=table)
            
        except Exception as e:
            return CommandResult(False, f"Error detecting anomalies: {str(e)}")
