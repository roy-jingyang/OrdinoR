from copy import deepcopy
from .AtomicRule import AtomicRule
from .Rule import Rule

class Node:
    def __init__(self, 
        label, 
        event_ids, 
        ct_label, at_label, tt_label,
        parent_label=None,
        step_rule=None, 
        parent_rule=None):

        self.label = label

        # contained events
        self.event_ids = event_ids

        # type labels
        self.ct_label = ct_label
        self.at_label = at_label
        self.tt_label = tt_label

        # TODO: tree structures
        # pointer to parent node
        #self.parent_label = parent_label
        # TODO: child nodes
        #self.children = list()

        # rule applied to create the current node from the previous
        if step_rule is None:
            self.step_rule = Rule([AtomicRule()])
        else:
            self.step_rule = step_rule

        # rules applied so far to create the current node, i.e., 
        # chained rules along the path from the root to the current
        if parent_rule is None:
            # root: empty rule
            self.composite_rule = self.step_rule
        else:
            self.composite_rule = deepcopy(parent_rule)
            self.composite_rule.extend(self.step_rule)

        # type rules
        rule_ct, rule_at, rule_tt = self.composite_rule.to_types()
        self.ct_rule = rule_ct
        self.at_rule = rule_at
        self.tt_rule = rule_tt

    def __repr__(self) -> str:
        str_node = f"Node [{self.label}]: (CT={self.ct_label}, AT={self.at_label}, TT={self.tt_label}), containing {len(self.event_ids)} events."
        rule_ct, rule_at, rule_tt = self.ct_rule, self.at_rule, self.tt_rule
        str_types = f"\tCT=[{self.ct_label}]{rule_ct}\n\tAT=[{self.at_label}]{rule_at}\n\tTT=[{self.tt_label}]{rule_tt}"
        return str_node + '\n' + str_types
        

    def append_child(self, child_node):
        self.children.append(child_node)


class Node2:
    def __init__(self, arr, events, resource_counts):
        self.arr = arr
        self.events = events
        self.resource_counts = resource_counts

    def __repr__(self) -> str:
        return f'''
            ********
            \tArray: {self.arr}
            \tEvents: {len(self.events)}
            \tResources: {self.resource_counts}
            ********
        '''
