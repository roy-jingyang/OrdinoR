"""
Rule-based approaches to learning execution contexts. 

    - TreeInductionMiner

"""

from .AtomicRule import AtomicRule
from .Rule import Rule

from .rule_generator import NumericRuleGenerator, CategoricalRuleGenerator

from .score_funcs import impurity, dispersal

from .tree_induction import TreeInductionMiner
