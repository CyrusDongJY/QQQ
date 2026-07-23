"""Scope declaration for the minimal regime portfolio simulator.

The current daily NAV series only applies regime-dependent QQQ equity
allocations. It intentionally excludes option sleeve PnL and must not be
presented as a joint equity/options portfolio.
"""


PORTFOLIO_SCOPE = "equity_allocation_only"
