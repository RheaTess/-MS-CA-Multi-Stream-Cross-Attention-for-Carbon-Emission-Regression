"""MS-CA: Multi-Stream Cross-Attention for carbon-emission regression.

A pure-regression pipeline that fuses two satellite proxy streams via
cross-attention to predict continuous carbon-emission values per spatial grid.

    Stream A (Environment)    : NO2, SO2, CO                              (3ch)
    Stream B (Socio-Infra)    : Nightlight, Urban Fraction, Power Plant,
                                Fossil Capacity                          (4ch)
"""

__version__ = "0.1.0"

from msca.models.msca_net import MSCANet

__all__ = ["MSCANet", "__version__"]
