import random
from node import Node
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

matrix_seed = 42
colour_seed = 42
number_of_nodes = 100
colours = list(range((number_of_nodes * 2) + 1))
colours_to_randomly_assign = 2
conflict_counts = []


def create_matrix_for_round_graph_with_crosses(num_nodes, seed):
    random.seed(seed)
    np.random.seed(seed)

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        adj_matrix[i, (i + 1) % num_nodes] = 1
        adj_matrix[(i + 1) % num_nodes, i] = 1

    additional_edges = random.randint(5, num_nodes // 2)

    for _ in range(additional_edges):
        a, b = np.random.choice(num_nodes, 2, replace=False)
        while abs(a - b) == 1 or adj_matrix[a, b] == 1:
            a, b = np.random.choice(num_nodes, 2, replace=False)
        adj_matrix[a, b] = 1
        adj_matrix[b, a] = 1

    plt.figure(figsize=(10, 10))
    plt.imshow(adj_matrix, cmap='Greys', interpolation='nearest')
    plt.show()

    return adj_matrix


def create_graph_from_adj_matrix(adj_matrix):
    graph = [Node(i) for i in range(len(adj_matrix))]

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] != 0:
                graph[i].add_neighbor(graph[j])

    return graph


def update_node_colours(graph, available_colours):
    for node in graph:
        if node.is_color_fixed == 1:
            continue

        conflicts = node.count_colour_conflicts()

        max_conflicts = len(node.neighbors)
        chance_to_change_colour = conflicts / max_conflicts if max_conflicts > 0 else 0

        # random.seed(None)
        if random.random() < chance_to_change_colour:
            for colour in available_colours:
                if all(neighbor.colour != colour for neighbor in node.neighbors):
                    node.future_colour = colour
                    break

    for node in graph:
        if node.future_colour != -1:
            node.colour = node.future_colour
            node.future_colour = -1


def normalise_colours(graph):
    used_colours = sorted({node.colour for node in graph})
    colour_mapping = {colour: new_colour for new_colour, colour in enumerate(used_colours)}

    for node in graph:
        node.colour = colour_mapping[node.colour]


def plot_conflicts(conflicts):
    plt.figure(figsize=(10, 6))
    plt.plot(conflicts, marker='o')
    plt.title('Number of Conflicts Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Amount of Conflicts')
    plt.grid(True)
    plt.show()


def plot_colored_graph(graph, number_of_colours):
    g = nx.Graph()

    for node in graph:
        g.add_node(node.id, color=node.colour)
        for neighbor in node.neighbors:
            if not g.has_edge(node.id, neighbor.id):
                g.add_edge(node.id, neighbor.id)

    # pos = nx.circular_layout(g)

    if number_of_colours <= 10:
        node_colors = [f"C{data['color'] % 10}" for _, data in g.nodes(data=True)]
        labels = None
    else:
        node_colors = 'C0'
        labels = {node: data['color'] for node, data in g.nodes(data=True)}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx(g, node_color=node_colors, labels=labels, with_labels=True, node_size=500, font_size=8)
    # nx.draw_networkx(g, pos, node_color=node_colors, labels=labels, with_labels=True, node_size=500, font_size=8)
    plt.show()


def colouring_loop(graph, max_stagnant_generations=20):
    number_of_conflicts = sum(node.count_colour_conflicts() for node in graph)
    conflict_counts.append(number_of_conflicts)

    stagnant_generation_count = 0
    previous_conflict_count = None

    while any(node.count_colour_conflicts() > 0 for node in graph):
        update_node_colours(graph, colours)
        number_of_conflicts = sum(node.count_colour_conflicts() for node in graph)
        conflict_counts.append(number_of_conflicts)

        # for index, current_node in enumerate(graph):
        #     print(f'Node {index} colour: {current_node.colour}')

        if previous_conflict_count == number_of_conflicts:
            stagnant_generation_count += 1
        else:
            stagnant_generation_count = 0

        if stagnant_generation_count >= max_stagnant_generations:
            break

        previous_conflict_count = number_of_conflicts

    normalise_colours(graph)

    for index, node in enumerate(graph):
        print(f'Node {index} colour: {node.colour}')
        # print(len(current_node.neighbors))

    colours_used = max(node.colour for node in graph) + 1
    print()
    print(f'Number of colours used: {colours_used}')
    print(f'Number of conflicts: {previous_conflict_count}')

    plot_conflicts(conflict_counts)
    plot_colored_graph(graph, colours_used)
    return graph


def create_randomly_coloured_graph(adj_matrix):
    graph = create_graph_from_adj_matrix(adj_matrix)

    random.seed(colour_seed)
    for node in graph:
        node.colour = random.choice(colours[:colours_to_randomly_assign])

    return graph


def fix_node_colours(graph, percentage, seed):
    random.seed(seed)
    num_nodes_to_fix = int(len(graph) * percentage)
    fixed_indices = random.sample(range(len(graph)), num_nodes_to_fix)

    for i in fixed_indices:
        graph[i].is_color_fixed = 1
        print(f'Node {i} has fixed colour: {graph[i].colour}')


def add_additional_edges(graph, percentage, seed):
    random.seed(seed)
    num_nodes = len(graph)
    potential_new_edges = int((num_nodes * (num_nodes - 1) / 2) * percentage)

    added_edges = 0
    attempts = 0
    max_attempts = num_nodes * 10

    while added_edges < potential_new_edges and attempts < max_attempts:
        node1, node2 = random.sample(graph, 2)

        if node2 not in node1.neighbors and node1 not in node2.neighbors:
            node1.add_neighbor(node2)
            node2.add_neighbor(node1)
            added_edges += 1

        attempts += 1

    if attempts >= max_attempts:
        print(f'Added {added_edges} edges.')


def compare_graph_colours(original_graph, modified_graph):
    changed_colors = []
    for original_node, modified_node in zip(original_graph, modified_graph):
        if original_node.id == modified_node.id and original_node.colour != modified_node.colour:
            changed_colors.append((original_node.id, original_node.colour, modified_node.colour))
    return changed_colors


def copy_graph_colours(graph):
    node_colours = []
    for old_node in graph:
        new_node = Node(old_node.id)
        new_node.colour = old_node.colour
        node_colours.append(new_node)

    return node_colours


def run_step_one():
    matrix = create_matrix_for_round_graph_with_crosses(number_of_nodes, matrix_seed)
    round_graph = create_randomly_coloured_graph(matrix)
    colouring_loop(round_graph)


def run_step_two_fixed_colours():
    matrix = create_matrix_for_round_graph_with_crosses(number_of_nodes, matrix_seed)
    round_graph = create_randomly_coloured_graph(matrix)
    fix_node_colours(round_graph, 0.1, matrix_seed)
    round_graph = colouring_loop(round_graph)

    for current_node in round_graph:
        if current_node.count_colour_conflicts() > 0:
            print(current_node)


def run_step_two_additional_edges():
    matrix = create_matrix_for_round_graph_with_crosses(number_of_nodes, matrix_seed)
    round_graph = create_randomly_coloured_graph(matrix)
    round_graph = colouring_loop(round_graph)
    graph_colours = copy_graph_colours(round_graph)
    add_additional_edges(round_graph, 0.05, matrix_seed)
    colouring_loop(round_graph)
    updated_colours = compare_graph_colours(graph_colours, round_graph)
    print("Number of updated colours:", len(updated_colours))
    print("Updated colours:", updated_colours)


run_step_one()
# run_step_two_fixed_colours()
# run_step_two_additional_edges()
