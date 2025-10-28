"""
REPP Beat Finding Analysis Package

This package provides enhanced tapping analysis functionality for beat-finding experiments.
"""

from .enhanced_tapping_analysis import (
    create_enhanced_tapping_plots,
    enhanced_tapping_analysis,
    check_tapping_quality,
    align_taps_to_markers
)

__version__ = "1.0.0"
__author__ = "Manuel Angladatort"

__all__ = [
    "create_enhanced_tapping_plots",
    "enhanced_tapping_analysis", 
    "check_tapping_quality",
    "align_taps_to_markers"
] 