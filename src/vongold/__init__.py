"""von-gold: a dry-run (paper) gold trading system driven by a local decision model.

Design rule: the trading logic is MECHANICAL and fully specified in code. The local
von decision model is an optional OVERLAY whose contribution must be demonstrated
against a mechanical baseline by walk-forward backtest. It is never trusted blindly.
"""

__version__ = "0.1.0"
