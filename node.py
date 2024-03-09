class Node:
    def __init__(self, id):
        self.id = id
        self.colour = -1
        self.future_colour = -1
        self.neighbors = []
        self.is_color_fixed = 0

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def count_colour_conflicts(self):
        conflict_count = 0
        for neighbor in self.neighbors:
            if neighbor.colour == self.colour:
                conflict_count += 1
        return conflict_count

    def __str__(self):
        return f'Node(Id = {self.id}, color={self.colour}, neighbors={len(self.neighbors)})'
