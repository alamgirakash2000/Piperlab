#!/usr/bin/env python3
"""Colorful, animated styling for the VLA Dataset Collector GUI."""

# Color scheme - Vibrant cyberpunk theme
COLORS = {
    # Primary backgrounds
    "bg_dark": "#0a0a1a",
    "bg_panel": "#12122a",
    "bg_card": "#1a1a3a",
    "bg_elevated": "#22224a",
    
    # Accent colors
    "cyan": "#00f5ff",
    "cyan_dark": "#00a8b5",
    "magenta": "#ff00ff",
    "magenta_dark": "#aa00aa",
    "lime": "#00ff88",
    "lime_dark": "#00aa5a",
    "orange": "#ff8800",
    "orange_dark": "#aa5a00",
    "purple": "#8855ff",
    "purple_dark": "#5533aa",
    
    # Status colors
    "success": "#00ff88",
    "warning": "#ffaa00",
    "error": "#ff4466",
    "info": "#00aaff",
    
    # Text colors
    "text_primary": "#ffffff",
    "text_secondary": "#aaaacc",
    "text_muted": "#666688",
    
    # Recording indicator
    "record_red": "#ff3344",
    "record_red_glow": "#ff6677",
}

# Global stylesheet
STYLESHEET = f"""
/* Main Window */
QMainWindow {{
    background-color: {COLORS["bg_dark"]};
}}

QWidget {{
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    font-size: 14px;
    color: {COLORS["text_primary"]};
}}

/* Panels and Cards */
QFrame#panel {{
    background-color: {COLORS["bg_panel"]};
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}}

QFrame#card {{
    background-color: {COLORS["bg_card"]};
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

/* Labels */
QLabel {{
    color: {COLORS["text_primary"]};
}}

QLabel#header {{
    font-size: 24px;
    font-weight: bold;
    color: {COLORS["cyan"]};
}}

QLabel#subheader {{
    font-size: 16px;
    font-weight: 600;
    color: {COLORS["text_secondary"]};
}}

QLabel#muted {{
    color: {COLORS["text_muted"]};
    font-size: 12px;
}}

QLabel#status_connected {{
    color: {COLORS["success"]};
    font-weight: bold;
}}

QLabel#status_disconnected {{
    color: {COLORS["error"]};
    font-weight: bold;
}}

/* Primary Button (Cyan glow) */
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLORS["cyan"]}, stop:1 {COLORS["cyan_dark"]});
    color: {COLORS["bg_dark"]};
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 15px;
    font-weight: bold;
    min-width: 120px;
}}

QPushButton#primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #33f8ff, stop:1 {COLORS["cyan"]});
}}

QPushButton#primary:pressed {{
    background: {COLORS["cyan_dark"]};
}}

QPushButton#primary:disabled {{
    background: {COLORS["text_muted"]};
    color: {COLORS["bg_panel"]};
}}

/* Teach Button (Magenta) */
QPushButton#teach {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLORS["magenta"]}, stop:1 {COLORS["magenta_dark"]});
    color: {COLORS["text_primary"]};
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 15px;
    font-weight: bold;
    min-width: 120px;
}}

QPushButton#teach:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff44ff, stop:1 {COLORS["magenta"]});
}}

QPushButton#teach:pressed {{
    background: {COLORS["magenta_dark"]};
}}

/* Play Button (Lime/Green) */
QPushButton#play {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLORS["lime"]}, stop:1 {COLORS["lime_dark"]});
    color: {COLORS["bg_dark"]};
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 15px;
    font-weight: bold;
    min-width: 120px;
}}

QPushButton#play:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #44ffaa, stop:1 {COLORS["lime"]});
}}

QPushButton#play:pressed {{
    background: {COLORS["lime_dark"]};
}}

QPushButton#play:disabled {{
    background: {COLORS["text_muted"]};
    color: {COLORS["bg_panel"]};
}}

/* Stop/Danger Button */
QPushButton#danger {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLORS["error"]}, stop:1 #cc2244);
    color: {COLORS["text_primary"]};
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 15px;
    font-weight: bold;
    min-width: 120px;
}}

QPushButton#danger:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff6688, stop:1 {COLORS["error"]});
}}

/* Secondary Button */
QPushButton#secondary {{
    background: transparent;
    color: {COLORS["text_secondary"]};
    border: 2px solid {COLORS["text_muted"]};
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 600;
}}

QPushButton#secondary:hover {{
    border-color: {COLORS["cyan"]};
    color: {COLORS["cyan"]};
}}

/* Small Icon Button */
QPushButton#icon_btn {{
    background: {COLORS["bg_elevated"]};
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 8px;
    min-width: 40px;
    min-height: 40px;
}}

QPushButton#icon_btn:hover {{
    background: {COLORS["bg_card"]};
    border-color: {COLORS["cyan"]};
}}

/* ComboBox */
QComboBox {{
    background: {COLORS["bg_card"]};
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 10px 14px;
    color: {COLORS["text_primary"]};
    min-width: 250px;
}}

QComboBox:hover {{
    border-color: {COLORS["cyan"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox QAbstractItemView {{
    background: {COLORS["bg_card"]};
    border: 2px solid {COLORS["cyan"]};
    border-radius: 8px;
    selection-background-color: {COLORS["cyan_dark"]};
}}

/* Progress Bar */
QProgressBar {{
    background: {COLORS["bg_card"]};
    border: none;
    border-radius: 8px;
    height: 16px;
    text-align: center;
    color: {COLORS["text_primary"]};
    font-weight: bold;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS["cyan"]}, stop:0.5 {COLORS["purple"]}, stop:1 {COLORS["magenta"]});
    border-radius: 8px;
}}

/* List Widget */
QListWidget {{
    background: {COLORS["bg_card"]};
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 4px;
}}

QListWidget::item {{
    background: transparent;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 2px 0;
}}

QListWidget::item:hover {{
    background: rgba(0, 245, 255, 0.1);
}}

QListWidget::item:selected {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(0, 245, 255, 0.3), stop:1 rgba(136, 85, 255, 0.3));
    border: 1px solid {COLORS["cyan"]};
}}

/* Scroll Bar */
QScrollBar:vertical {{
    background: {COLORS["bg_dark"]};
    width: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background: {COLORS["text_muted"]};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS["cyan"]};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* Tooltip */
QToolTip {{
    background: {COLORS["bg_elevated"]};
    color: {COLORS["text_primary"]};
    border: 1px solid {COLORS["cyan"]};
    border-radius: 6px;
    padding: 6px 10px;
}}

/* Status Bar */
QStatusBar {{
    background: {COLORS["bg_panel"]};
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    color: {COLORS["text_secondary"]};
}}

/* Group Box */
QGroupBox {{
    background: {COLORS["bg_card"]};
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    margin-top: 16px;
    padding-top: 16px;
    font-weight: bold;
}}

QGroupBox::title {{
    color: {COLORS["cyan"]};
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 10px;
}}
"""

# Animation keyframes (for reference in widgets)
ANIMATIONS = {
    "pulse_duration": 1000,
    "glow_duration": 2000,
    "fade_duration": 300,
}


def get_gradient_css(start_color: str, end_color: str, direction: str = "vertical") -> str:
    """Generate CSS gradient string."""
    if direction == "vertical":
        return f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {start_color}, stop:1 {end_color})"
    else:
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {start_color}, stop:1 {end_color})"
