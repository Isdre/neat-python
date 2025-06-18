"""Handles node and connection genes."""
import warnings
from random import random

from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import BaseGene
from neat.mc.gru.attributes import GRUAttribute, GRUCell


class GRUNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        GRUAttribute('gru_cell'),
                        FloatAttribute('response'),
                        StringAttribute('activation', options=''),
                        StringAttribute('aggregation', options='')]

    def __init__(self, key):
        assert isinstance(key, int), f"GruNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)

        try:
            if getattr(other,"gru_cell"):
                d += self.gru_cell.distance(other.gru_cell)
        except AttributeError:
            d += 4.0

        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient
