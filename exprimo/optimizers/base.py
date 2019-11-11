
class BaseOptimizer:

    def __init__(self, colocation_heuristic):
        self.colocation_heuristic = colocation_heuristic

    def create_colocation_groups(self, layer_names):
        groups = []
        for layer in layer_names:
            for group in groups:
                if self.colocation_heuristic(layer, group[0]):
                    group.append(layer)
                    break
            else:
                groups.append([layer])
        return groups

