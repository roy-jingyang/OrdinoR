class Node:
    def __init__(self, 
        label, 
        event_ids, 
        atomic_rule, 
        ct_label, at_label, tt_label):

        self.label = label

        # containing events
        self.event_ids = event_ids

        # child nodes
        self.children = list()

        # rule applied to create the node
        self.ar = atomic_rule

        # type labels (inherit from parent if not specified)
        self.ct = ct_label
        self.at = at_label
        self.tt = tt_label

    def append_child(self, child_node):
        self.children.append(child_node)
