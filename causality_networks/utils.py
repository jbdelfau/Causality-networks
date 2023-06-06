import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Circle
from causality_networks.component.automata import PFSA


def draw_derivative_heap(points, vertices, x0, title=None):
    """
    Displays the derivative heap.
    :param points: symbolic derivatives forming the heap.
    :param vertices: vertices of the convex hull of the heap.
    :param x0: \epsilon-synchronizing sequence
    :param title: window title
    """
    if len(points[0]) == 2:
        ax = plt.figure()
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.plot(points[:, 0], points[:, 1], "o")
        if vertices is not None:
            for _, vertex in vertices:
                plt.plot(vertex[0], vertex[1], "rx")
        if x0 is not None:
            plt.plot(x0[0], x0[1], "go")

    elif len(points[0]) == 3:
        ax = plt.figure().add_subplot(projection="3d")
        plt.plot(points[:, 0], points[:, 1], "o")
        if vertices is not None:
            for _, vertex in vertices:
                plt.plot(vertex[0], vertex[1], vertex[2], "rx")
        if x0 is not None:
            plt.plot(x0[0], x0[1], x0[2], "go")

    else:
        return

    if title is not None:
        if len(points[0]) == 2:
            plt.title(title)
        else:
            ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    plt.show()


def draw_pfsa(pfsa: PFSA, title="PFSA", window_title="PFSA", colormap="Spectral"):
    """
    Displays a given PFSA or XPFSA.
    :param pfsa: PFSA or XPFSA object
    :param title: name of the automata
    :param window_title: name of the displayed window
    :param colormap: colormap used
    """
    cmap = get_cmap(colormap)
    dx = 1 / len(pfsa.alphabet_source)
    edge_colors_dic = {symbol: cmap(dx / 2 + i * dx) for i, symbol in enumerate(pfsa.alphabet_source)}

    def get_edge_width(probability):
        return max(0.5, int(4 * probability))

    def label_position(node_i_x, node_j_x):
        x = (node_i_x + node_j_x) / 2 - 0.005
        if node_i_x != node_j_x:
            y = 1.0 * (node_i_x - node_j_x) / 2
        else:
            y = 0.135
        return x, y

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title(window_title)
    plt.xlim([0, 1])
    plt.ylim([-0.4, 0.6])
    plt.title(title)
    margins = 0.1
    nodes_distance = (1 - 2 * margins) / (len(pfsa.states) - 1) if len(pfsa.states) > 1 else 1
    nodes_pos = {state: ((i * nodes_distance + margins), 0) for i, state in enumerate(pfsa.states)}

    # Draw edges and edges labels
    for edge in pfsa.graph.edges(data=True):
        symbol = edge[2]["symbol"][0]
        edge_color = edge_colors_dic[symbol]
        current_edge = [(edge[0], edge[1])]
        curvature = "arc3,rad=1" if edge[0] != edge[1] else "angle,angleA=0,angleB=180,rad=0.9"
        symbol_pos = label_position(nodes_pos[edge[0]][0], nodes_pos[edge[1]][0])
        nodes_margin = 20
        if pfsa.type == "PFSA":
            proba = edge[2]["probability"][0]
            edge_width = get_edge_width(proba)
            ax.text(symbol_pos[0] + 0.05, symbol_pos[1], format(proba, ".2f"), fontsize=12, backgroundcolor="white",
                    bbox=dict(boxstyle="round", ec="black", fc="lightblue", alpha=0.5))
        else:
            edge_width = 2
        nx.draw_networkx_edges(pfsa.graph, nodes_pos, edge_color=edge_color, connectionstyle=curvature,
                               min_source_margin=nodes_margin, min_target_margin=nodes_margin, edgelist=current_edge,
                               width=edge_width, ax=ax)
        ax.text(symbol_pos[0], symbol_pos[1], symbol, fontsize=12, backgroundcolor="white",
                bbox=dict(boxstyle="Square", ec="black", fc="white"))

    if pfsa.type == "XPFSA":
        for node, pos in nodes_pos.items():
            probs = "\n".join(
                [f"{symbol}: {format(prob, '.2f')}" for symbol, prob in pfsa.symbols_probabilities[node].items()]
            )
            ax.text(pos[0] + 0.05, pos[1] - 0.05, probs, fontsize=12, backgroundcolor="white",
                    bbox=dict(boxstyle="round", ec="black", fc="lightyellow", alpha=0.5))

    # Draw nodes
    nx.draw_networkx_nodes(
        pfsa.graph, nodes_pos, node_size=1000, node_color="white", edgecolors="black", ax=ax, margins=0
    )
    # Draw nodes labels
    nx.draw_networkx_labels(pfsa.graph, nodes_pos, font_size=15, ax=ax)
    plt.axis("equal")
    plt.show()


def sublist_finder(mylist, pattern):
    """
    Finds specific subsequences in a large list and returns their starting indices.
    :param mylist: large list
    :param pattern: subsequence to find
    :return: starting indices of the subsequences
    """

    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i: i + len(pattern)] == pattern:
            if i + len(pattern) < len(mylist):
                matches.append(i + len(pattern))
    return matches


def test_color_scheme(colormap, n_colors):
    """
    Testing colormaps for PFSA drawing.
    :param colormap: name of the colormap
    :param n_colors: number of symbols
    """

    cmap = get_cmap(colormap)

    fig = plt.figure()
    ax = fig.add_subplot()
    dx = 1 / n_colors
    for i in range(n_colors):
        rgba = cmap(dx / 2 + i * dx)
        circle1 = Circle((i, 0.2), radius=0.5, color=rgba)
        ax.add_patch(circle1)
    ax.axis('equal')
    plt.show()


if __name__ == "__main__":

    test_color_scheme("Spectral", 5)
