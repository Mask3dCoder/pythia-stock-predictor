"""
Pythia CLI Theme System

Centralized design tokens for consistent terminal UI.
Inspired by modern CLI design principles — clean, minimal, scannable.
"""

from rich.style import Style
from rich.theme import Theme
from rich.color import Color

# ── Color Palette ────────────────────────────────────────────────────────────
# Brand colors
BRAND_PRIMARY = "#7C3AED"       # Purple — logo, branding
BRAND_SECONDARY = "#A78BFA"     # Light purple — accents
BRAND_MUTED = "#C4B5FD"         # Muted purple — subtle highlights

# Semantic colors
SUCCESS = "#10B981"             # Green — success states
SUCCESS_DIM = "#059669"         # Dark green — secondary success
WARNING = "#F59E0B"             # Amber — warnings
WARNING_DIM = "#D97706"         # Dark amber
ERROR = "#EF4444"               # Red — errors
ERROR_DIM = "#DC2626"           # Dark red
INFO = "#06B6D4"                # Cyan — informational
INFO_DIM = "#0891B2"            # Dark cyan

# Data colors
PRICE_UP = "#10B981"            # Green — price increases
PRICE_DOWN = "#EF4444"          # Red — price decreases
NEUTRAL = "#6B7280"             # Gray — neutral/unchanged
HIGHLIGHT = "#FBBF24"           # Gold — important values

# Neutral palette
TEXT_PRIMARY = "#F9FAFB"        # Near-white — primary text
TEXT_SECONDARY = "#9CA3AF"      # Gray — secondary text
TEXT_MUTED = "#6B7280"          # Dark gray — muted text
BORDER = "#374151"              # Border color
BORDER_FOCUS = "#7C3AED"        # Focused border
SURFACE = "#1F2937"             # Surface/panel background
BACKGROUND = "#111827"          # App background

# ── Rich Theme ────────────────────────────────────────────────────────────────

PYTHIA_THEME = Theme(
    {
        # Brand
        "brand": Style(color=BRAND_PRIMARY, bold=True),
        "brand.dim": Style(color=BRAND_SECONDARY),
        "brand.muted": Style(color=BRAND_MUTED),
        # Semantic
        "success": Style(color=SUCCESS),
        "success.bold": Style(color=SUCCESS, bold=True),
        "warning": Style(color=WARNING),
        "warning.bold": Style(color=WARNING, bold=True),
        "error": Style(color=ERROR),
        "error.bold": Style(color=ERROR, bold=True),
        "info": Style(color=INFO),
        "info.bold": Style(color=INFO, bold=True),
        # Data
        "price.up": Style(color=PRICE_UP, bold=True),
        "price.down": Style(color=PRICE_DOWN, bold=True),
        "highlight": Style(color=HIGHLIGHT, bold=True),
        # Text
        "text.primary": Style(color=TEXT_PRIMARY),
        "text.secondary": Style(color=TEXT_SECONDARY),
        "text.muted": Style(color=TEXT_MUTED),
        # UI
        "border": Style(color=BORDER),
        "border.focus": Style(color=BORDER_FOCUS),
        # Headings
        "heading": Style(color=BRAND_PRIMARY, bold=True),
        "subheading": Style(color=TEXT_SECONDARY, bold=True),
        # Code/values
        "value": Style(color=TEXT_PRIMARY, bold=True),
        "symbol": Style(color=INFO, bold=True),
        # Status
        "status.ok": Style(color=SUCCESS),
        "status.fail": Style(color=ERROR),
        "status.pending": Style(color=WARNING),
    }
)

# ── Rich Console Styles ──────────────────────────────────────────────────────

# Panel style colors (hex strings for Rich component compatibility)
PANEL_STYLES = {
    "default": BORDER,
    "success": SUCCESS,
    "warning": WARNING,
    "error": ERROR,
    "info": INFO,
    "brand": BRAND_PRIMARY,
    "highlight": HIGHLIGHT,
}

# Table styles
TABLE_STYLE = {
    "header": Style(color=TEXT_PRIMARY, bold=True),
    "border": Style(color=BORDER),
    "row_alt": Style(bgcolor="#1F2937"),
    "highlight": Style(color=HIGHLIGHT, bold=True),
}

# Rule/divider styles
RULE_STYLES = {
    "section": Style(color=BORDER),
    "accent": Style(color=BRAND_PRIMARY),
}

# ── Layout Constants ─────────────────────────────────────────────────────────

# Panel padding
PANEL_PADDING = (1, 2)

# Maximum widths
MAX_TABLE_WIDTH = 100
MAX_PANEL_WIDTH = 90

# ── Icon Set ──────────────────────────────────────────────────────────────────

# Unicode-friendly status icons (no emoji)
ICONS = {
    "success": "✓",
    "error": "✗",
    "warning": "⚠",
    "info": "ℹ",
    "arrow": "→",
    "bullet": "•",
    "dash": "─",
    "star": "★",
    "diamond": "◆",
    "chevron": "›",
    "dot": "·",
}

# ── ASCII Logo ────────────────────────────────────────────────────────────────

LOGO = r"""
   ____        _   _     _
  |  _ \ _   _| |_| |__ (_) __ _
  | |_) | | | | __| '_ \| |/ _` |
  |  __/| |_| | |_| | | | | (_| |
  |_|    \__, |\__|_| |_|_|\__,_|
         |___/

       Stock Prediction CLI  •  v3.0.0
"""

LOGO_SMALL = r"""
   ____       _   _     _
  |  _ \ _  _| |_| |__ (_) __ _
  | |_) | | | | __| '_ \| |/ _` |
  |  __/| |_| | |_| | | | | (_| |
  |_|    \__, |\__|_| |_|_|\__,_|
         |___/
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def style_price(price: float) -> str:
    """Return the appropriate style name for a price."""
    return "price.up" if price >= 0 else "price.down"


def style_change(change_pct: float) -> str:
    """Return the appropriate style for a percentage change."""
    if change_pct > 0:
        return "price.up"
    elif change_pct < 0:
        return "price.down"
    return "text.muted"


def format_change(change_pct: float) -> str:
    """Format a percentage change with + sign."""
    if change_pct > 0:
        return f"+{change_pct:.2f}%"
    elif change_pct < 0:
        return f"{change_pct:.2f}%"
    return "0.00%"
