"""
Community detection (graph partition) techniques for resource group
discovery
"""

from .connected_comp import mja, mjc

__all__ = [
    'mja', 'mjc',
]
