import numpy as np
import networkx as nx
from itertools import product, combinations_with_replacement
from scipy.spatial import ConvexHull
from causality_networks.utils import draw_pfsa, sublist_finder
from causality_networks.component.automata import PFSA

"""
Functions to infer a Probabilistic Finite State Automaton (PFSA) or Crossed Probabilistic Finite State Automaton (XPFSA)
from discrete time series. The steps, algorithms and equations mentioned here can be found in the supplementary material
of the article 'Event-level prediction of urban crime reveals a signature of enforcement bias in US cities' by Huang et 
al.
"""


def build_pfsa_recursively(x_0, alphabet_in, alphabet_out, x_in, x_out, epsilon, min_occurence):
    """
    Iterative construction of the automaton associated with the time series x_in and x_out using the
    \epsilon-synchronizing sequence x_0.
    :param x_0: \epsilon-synchronizing sequence
    :param alphabet_in: alphabet of the source time serie x_in
    :param alphabet_out:alphabet of the target time serie x_out
    :param x_in: source time serie
    :param x_out: target time serie
    :param epsilon: hyperparameter \epsilon
    :param min_occurence: minimum number of occurences of the subsequence required to keep the symbolic derivative
    :return: automaton object
    """
    n_states, states, new_states, states_words, symb_derivatives, state_transitions = init_pfsa_graph(
        x_0, x_in, x_out, alphabet_out)
    automata_type = "PFSA" if x_in == x_out else "XPFSA"

    # Step 8: keep iterating as long as new states are found
    while len(new_states) > 0:
        states_to_process = new_states.copy()
        # Step 9 : init empty new states ensemble
        new_states = []
        # Step 10 : iterate over new words
        for from_state, symbol in product(states_to_process, alphabet_in):
            from_states_str = states_words[from_state]
            # Step 11 : define new sequence x
            x_id = from_states_str + [symbol]
            # Step 11: calculate derivative heap or cross derivative heap d
            symb_deriv_state = compute_symbolic_derivative(
                x_in, x_out, alphabet_out, x_id, min_occurence
            )
            if symb_deriv_state is None:
                state_transitions[from_state][symbol] = from_state
                continue

            # Step 12 : check if the state found is a new one
            _is_new_state = True
            for similar_state, s_der in symb_derivatives.items():
                # TODO : which distance should we use?
                # _diff = max([abs(s_der[i] - symb_deriv_state[i]) for i in range(len(alphabet))])
                _diff = np.linalg.norm(np.array(s_der) - np.array(symb_deriv_state))
                # Step 13: if the new state is close enough to an existing one, we map the sequence to this state.
                if _diff <= epsilon:
                    _is_new_state = False
                    state_transitions[from_state][symbol] = similar_state
                    break
            # Step 15 to 17 : if the new state is sufficiently different from the existing ones, we add it to
            # the automata properties
            if _is_new_state:
                n_states += 1
                states.append(f"q{n_states}")
                new_states.append(f"q{n_states}")
                states_words[f"q{n_states}"] = x_id
                symb_derivatives[f"q{n_states}"] = symb_deriv_state
                state_transitions[from_state][symbol] = f"q{n_states}"
                state_transitions[f"q{n_states}"] = {}

    symbols_probabilities = {
        from_state: {alphabet_out[i]: round(prob, 2) for i, prob in enumerate(probs)}
        for from_state, probs in symb_derivatives.items()
    }

    # Definition of the new PFSA object
    pfsa_dic = {
        "states": states,
        "states_words": states_words,
        "alphabet_source": alphabet_in,
        "alphabet_target": alphabet_out,
        "states_transitions": state_transitions,
        "symbols_probabilities": symbols_probabilities,
    }
    pfsa_model = PFSA(pfsa_dic, automata_type)

    if len(pfsa_model.states) > 1:
        pruned_pfsa = prune_pfsa(pfsa_model)
    else:
        pruned_pfsa = pfsa_model

    return pruned_pfsa


def calculate_causality_coefficient(xpfsa: PFSA, serie_i, serie_j):
    """
    Calculation of the causality coefficient \bar{\gamma}_j^i as defined in algorithm 2 of the paper
    'Causality networks' by Chattopadhyay.
    :param xpfsa: XPFSA model
    :param serie_i: time serie potentially influencing the other
    :param serie_j: time serie potentially influenced by the other
    :return: causality coefficient \bar{\gamma}_j^i
    """

    r = {symbol: serie_j.count(symbol) / len(serie_j) for symbol in set(serie_j)}
    denominator = sum([r[symbol] * np.log2(r[symbol]) for symbol in set(serie_j)])

    u = xpfsa.streamrun_word(serie_i)
    h = {
        state: sum(
            [
                xpfsa.symbols_probabilities[state][symbol] * np.log2(xpfsa.symbols_probabilities[state][symbol])
                if xpfsa.symbols_probabilities[state][symbol] != 0
                else 0
                for symbol in xpfsa.alphabet_target
            ]
        )
        for state in xpfsa.states
    }
    numerator = sum([u[state] * h[state] for state in xpfsa.states])

    return round(1 - (numerator / denominator), 3)


def calculate_pfsa(x_in, x_out, hyperparameters, sequence_max_length):
    """
    Builds an automaton compatible with the time series serie_source and serie_target (if serie_source == serie_target,
    we get a PFSA, else we get a XPFSA).
    :param x_in: source time serie
    :param x_out: target time serie
    :param hyperparameters: dictionary of hyperparameters
    :param sequence_max_length: maximum length of subsequences
    :return: automaton object
    """

    alphabet_in = list(set(x_in))
    alphabet_out = list(set(x_out))
    # Step 3: calculation of the symbolic derivative heap of cross derivative heap.
    dh_point_to_word, dh_string_to_point = get_symbolic_derivative_heap(x_in, x_out, sequence_max_length,
                                                                        hyperparameters["min_occurence"])
    # Steps 4 to 5 : calculation of the convex hull and synchronizing word x_0
    synchronizing_word, occurences, _ = get_synchronizing_word(dh_point_to_word, alphabet_out, x_in,
                                                               hyperparameters["epsilon"])
    # Steps 5 to 18: building a relevant automata recursively
    inferred_pfsa = build_pfsa_recursively(
        synchronizing_word, alphabet_in, alphabet_out, x_in, x_out, hyperparameters["epsilon"],
        hyperparameters["min_occurence"]
    )
    # Step 19 : Extraction of the strongly connected components of the network
    strongly_connected_pfsa = extract_strongly_connected_components(inferred_pfsa, x_in)
    # Steps 20 to 26 : adjust the transition probabilities between states
    strongly_connected_pfsa = find_transition_probabilities(strongly_connected_pfsa, x_in, x_out)
    return strongly_connected_pfsa


def close_pfsa(strongly_connected_components: PFSA, full_pfsa: PFSA):
    """
    After extracting the strongly connected components from a graph, some connections can be missing (each states must
    have a transition for each symbol of the alphabet). In this case, we add the missing transitions by first
    identifying the state which is the closest to the target state previously deleted and linking it to the starting
    state.
    :param strongly_connected_components: strongly connected components of the graph
    :param full_pfsa: full graph
    :return: automaton object with missing connections
    """
    for state, dic in strongly_connected_components.state_transitions.items():
        for symbol in strongly_connected_components.alphabet_source:
            if symbol not in dic.keys():
                state_name = full_pfsa.state_transitions[state][symbol]
                state_probabilities = np.array(list(full_pfsa.symbols_probabilities[state_name].values()))
                closest_state = find_closest_state(strongly_connected_components, state_name, state_probabilities)
                strongly_connected_components.state_transitions[state][symbol] = closest_state
                strongly_connected_components.symbols_probabilities[state][symbol] = -1

    return strongly_connected_components


def compute_symbolic_derivative(x_in, x_out, alphabet_out, word, min_occurence=1):
    """
    Calculates the symbolic derivative if x_in == x_out or cross symbolic derivative else for all the symbols sigma of
    x_out alphabet. We use equation (5), which is strictly similar to equation (4) if x_in == x_out.
    :param x_in: source time serie x_in
    :param x_out: target time serie x_out
    :param alphabet_out: alphabet of the target time serie
    :param word: subsequence y in equation (5)
    :param min_occurence: minimum number of occurences of the subsequence required to keep the symbolic derivative
    :return: tuple giving the symbolic derivative \phi_{y}^{x_in} or cross symbolic derivative \phi_{y}^{x_in, x_out}
    for all the symbols sigma of x_out alphabet
    """
    indices = sublist_finder(x_in, word)
    if len(indices) < min_occurence:
        return None

    next_symbols = [x_out[i] for i in indices]
    symbolic_derivative = tuple([next_symbols.count(sigma) / len(next_symbols) for sigma in alphabet_out])
    return symbolic_derivative


def extract_strongly_connected_components(pfsa: PFSA, x_in):
    """
    Extraction of the strongly connected components of a given graph
    :param pfsa: automaton object
    :param x_in: source time serie
    :return:
    """
    strongly_connected_graphs = (pfsa.graph.subgraph(c) for c in nx.strongly_connected_components(pfsa.graph))
    graph_max_occurences = []

    for _graph in strongly_connected_graphs:
        graph_words = nx.get_node_attributes(_graph, "word").values()
        graph_occurence = max([len(sublist_finder(x_in, word)) for word in graph_words])
        graph_max_occurences.append((_graph, graph_occurence))

    most_frequent_graph = sorted(graph_max_occurences, key=lambda x: x[1], reverse=True)[0][0]

    strongly_connected_pfsa = PFSA(most_frequent_graph, pfsa.type)
    strongly_connected_pfsa = close_pfsa(strongly_connected_pfsa, pfsa)

    return strongly_connected_pfsa


def find_closest_state(pfsa: PFSA, state_name, state_probabilities):
    """
    Find the state which has the closest properties to a specific state.
    :param pfsa: automaton object
    :param state_name: name of the specific state
    :param state_probabilities: transition probabilities of the state
    :return: name of the closest state
    """
    diff = []
    for state, prob in pfsa.symbols_probabilities.items():
        if state != state_name:
            diff.append((state, np.linalg.norm(state_probabilities - np.array(list(prob.values())))))
    diff = sorted(diff, key=lambda x: x[1])
    return diff[0][0]


def find_transition_probabilities(pfsa: PFSA, x_in, x_out):
    """
    Finds the transition probabilities for all the states of a given automaton and for each symbol of the target time
    serie alphabet.
    :param pfsa: automaton object
    :param x_in: source time serie
    :param x_out: target time serie
    :return: automaton object
    """
    n_counts = {(q, symbol): 0 for q, symbol in product(pfsa.states, pfsa.alphabet_target)}
    current_state = np.random.choice(pfsa.states)

    for i in range(1, len(x_out)):
        sigma = x_in[i]
        tau = x_out[i]
        n_counts[(current_state, tau)] += 1
        current_state = pfsa.state_transitions[current_state][sigma]
    for state in pfsa.states:
        for symbol in pfsa.alphabet_target:
            pfsa.symbols_probabilities[state][symbol] = n_counts[(state, symbol)] / sum(
                n_counts[(state, sigma)] for sigma in pfsa.alphabet_target
            )

    return pfsa


def get_convex_hull_vertices(hull_dimension, points):
    """
    Identification of the vertices of a convex hull.
    :param hull_dimension: dimension of the hull
    :param points: points defining the hull
    :return: list of the hull vertices
    """
    if hull_dimension == 1:
        string_vertices = [min(points, key=lambda x: x[0]), max(points, key=lambda x: x[0])]
    else:
        hull = ConvexHull(points[:, :hull_dimension], qhull_options="Qs")
        string_vertices = [y for y in [points[vert] for vert in hull.vertices]]
    return string_vertices


def get_symbolic_derivative_heap(x_in, x_out, max_length, min_occurence):
    """
    Compute the symbolic derivative heap D_{\epsilon}^{x_{in}, x_{out}}.
    :param x_in: finite time serie generated by a QSP
    :param x_out: finite time serie generated by a QSP
    :param max_length: maximum length of the substrings used to compute the probabilities
    :param min_occurence: minimum number of occurences required to consider the substring
    :return:
    """
    initial_string = []
    alphabet_in = set(x_in)
    alphabet_out = set(x_out)
    dh_point_to_string = {}
    dh_string_to_point = {}
    n_points = 0
    n_redundant = 0
    for substring_size in range(1, max_length + 1):
        for str_suffix in product(alphabet_in, repeat=substring_size):
            subsequence = initial_string + list(str_suffix)
            # Calculating the symbolic derivative or cross symbolic derivative for all the symbols of x_out alphabet
            symbolic_derivative = compute_symbolic_derivative(x_in, x_out, alphabet_out, subsequence, min_occurence)
            if symbolic_derivative is not None:
                n_points += 1
                dh_string_to_point[",".join(subsequence)] = symbolic_derivative
                if symbolic_derivative not in dh_point_to_string.keys():
                    dh_point_to_string[symbolic_derivative] = [subsequence]
                else:
                    dh_point_to_string[symbolic_derivative] += [subsequence]
                    n_redundant += 1

    # print(f"Symbolic derivative heap size: {n_points}, Identical points: {n_redundant}")
    return dh_point_to_string, dh_string_to_point


def get_synchronizing_word(dh_point_to_sequence, alphabet_out, x_in, epsilon):
    """
    Calculation of the convex hull of the derivative heap and identification of a \epsilon-synchronizing sequence as an
    extrema.
    :param dh_point_to_sequence: derivative heap
    :param alphabet_out: alphabet of the source time serie
    :param x_in: source time serie
    :param epsilon: hyperparameter \epsilon
    :return: \epsilon-synchronizing sequence
    """
    hull_dimension = len(alphabet_out) - 1
    points = np.array([np.array(x) for x in dh_point_to_sequence.keys()])
    # Calculate convex hull
    vertices_position = get_convex_hull_vertices(hull_dimension, points)
    # Get most common word epsilon-close to vertices
    close_points = [x for x in points if any([np.linalg.norm(x - vertex) <= epsilon for vertex in vertices_position])]
    close_sequences = [y for x in close_points for y in dh_point_to_sequence[tuple(x)]]
    sequences_count = [
        (word, len(sublist_finder(x_in, list(word))), point) for word, point in zip(close_sequences, close_points)
    ]
    sorted_sequences = sorted(sequences_count, key=lambda x: x[1], reverse=True)
    # draw_derivative_heap(points, None, None)

    # Check vertices properties
    # for synchr_str in sorted(string_count, key=lambda x: x[1], reverse=True):
    #     print(synchr_str)
    #     _temp_dh_point_to_string, _ = build_derivative_heap(serie, alphabet, 5, 30, synchr_str[0])
    #     test = np.array([list(x) for x in _temp_dh_point_to_string.keys()])
    #     draw_derivative_heap(test, None, None, title=",".join(synchr_str[0]))
    #     plt.show()
    # Draw derivative heap, convex hull and most common vertex
    # draw_derivative_heap(points, string_vertices, most_common_str[2])

    return sorted_sequences[0]


def granger_network(timeseries: dict, hyperparameters: dict, sequence_max_length: int, window_title, display_pfsa=False,
                    display_xpfsa=False):
    """
    Calculation of the Crossed probabilistic finite state automaton (XPFSA) for each pair of the given time series and
    of their coefficients of causal dependence \gamma. If \gamma is greater than a given threshold, the XPFSA is
    shown on screen.
    :param timeseries: dictionary of discrete time series.
    :param hyperparameters: dictionary of hyperparameters
    :param sequence_max_length: maximum length of sub-sequences
    :param window_title: name of the displayed window
    :param display_pfsa: do we display PFSA or not
    :param display_xpfsa: do we display XPFSA or not
    :return:
    """

    automata = {"PFSA": {}, "XPFSA": {}}
    gamma = {}
    series_names = list(timeseries.keys())
    for delta in range(0, hyperparameters["max_delay"] + 1):
        for serie_i, serie_j in combinations_with_replacement(series_names, 2):
            if (serie_i == serie_j) and (delta > 0):
                continue
            if (serie_i == serie_j) and (delta == 0):
                pfsa = calculate_pfsa(timeseries[serie_i], timeseries[serie_j], hyperparameters, sequence_max_length)
                automata["PFSA"][serie_i] = pfsa
                if display_pfsa:
                    draw_pfsa(pfsa, title=f"PFSA for {serie_i}", window_title=window_title)
            else:
                serie_source = timeseries[serie_i][: len(timeseries[serie_i]) - delta]
                serie_target = timeseries[serie_j][delta:]
                xpfsa = calculate_pfsa(serie_source, serie_target, hyperparameters, sequence_max_length)
                automata["XPFSA"][(serie_i, serie_j, delta)] = xpfsa
                gamma[(serie_i, serie_j, delta)] = calculate_causality_coefficient(xpfsa, serie_source, serie_target)
                if display_xpfsa and (gamma[(serie_i, serie_j, delta)] >= hyperparameters["coefficient_threshold"]):
                    title = f"XPFSA for {serie_i} on {serie_j}, time delay {delta}: causality coefficient " \
                            f"{gamma[(serie_i, serie_j, delta)]}"
                    draw_pfsa(xpfsa, title=title, window_title=window_title)
                    print(f"Causality coefficient of {serie_i} on {serie_j}, time delay {delta}: "
                          f"causality coefficient {gamma[(serie_i, serie_j, delta)]}")

                serie_source = timeseries[serie_j][: len(timeseries[serie_j]) - delta]
                serie_target = timeseries[serie_i][delta:]
                xpfsa = calculate_pfsa(serie_source, serie_target, hyperparameters, sequence_max_length)
                automata["XPFSA"][(serie_j, serie_i, delta)] = xpfsa
                gamma[(serie_j, serie_i, delta)] = calculate_causality_coefficient(xpfsa, serie_source,
                                                                                       serie_target)
                if display_xpfsa and gamma[(serie_j, serie_i, delta)] >= hyperparameters["coefficient_threshold"]:
                    title = f"XPFSA for {serie_j} on {serie_i}, time delay {delta}: causality coefficient " \
                            f"{gamma[(serie_j, serie_i, delta)]}"
                    draw_pfsa(xpfsa, title=title, window_title=window_title)
                    print(
                        f"Causality coefficient of {serie_j} on {serie_i}, time delay {delta}: "
                        f"causality coefficient {gamma[(serie_j, serie_i, delta)]}"
                    )
    return gamma, automata


def init_pfsa_graph(x0, x_in, x_out, alphabet_out):
    """
    initialisation of the automaton.
    :param x0: \epsilon-synchronizing sequence
    :param x_in: source time serie
    :param x_out: target time serie
    :param alphabet_out: alphabet of the target time serie
    :return: properties of the automaton
    """
    n_states = 0
    states = [f"q{n_states}"]
    states_words = {f"q{n_states}": list(x0)}
    symbolic_derivative = {
        f"q{n_states}": compute_symbolic_derivative(x_in, x_out, alphabet_out, list(x0))
    }
    state_transitions = {f"q{n_states}": {}}
    new_states = states.copy()
    return n_states, states, new_states, states_words, symbolic_derivative, state_transitions


def prune_pfsa(pfsa: PFSA):
    """
    Simplifies a given graph by deleting states which transitions all point to the same state.
    :param pfsa: automaton object
    :return: simplified automaton object
    """
    states_to_delete = []
    state_dic = {state: state for state in pfsa.states}
    for from_state, dic in pfsa.state_transitions.items():
        to_states = set([x for x in dic.values()])
        if len(to_states) == 1:
            from_state_probabilities = np.array(list(pfsa.symbols_probabilities[from_state].values()))
            closest_state = find_closest_state(pfsa, from_state, from_state_probabilities)
            states_to_delete.append(from_state)
            state_dic[from_state] = closest_state

    for state in states_to_delete:
        pfsa.states.remove(state)
        pfsa.state_transitions.pop(state)
        pfsa.symbols_probabilities.pop(state)
        pfsa.state_transitions = {
            q1: {k: state_dic[v] for k, v in dic.items()} for q1, dic in pfsa.state_transitions.items()
        }
    return pfsa

# def condense_pfsa_states(pfsa):
#     epsilon = 0.1
#     merging_states = []
#     for q1, q2 in combinations(pfsa["states"], 2):
#         diff_max = max(
#             abs(pfsa["symbols_probabilities"][q1][symbol] - pfsa["symbols_probabilities"][q2][symbol])
#             for symbol in pfsa["alphabet"]
#         )
#         if diff_max <= epsilon:
#             merging_states.append(sorted((q1, q2)))
#
#     n = len(merging_states)
#     while n > 0:
#         for states_to_replace in combinations(merging_states, n):
#             transitions = []
#             state_dic = {state: state for state in pfsa["states"]}
#             for q1, q2 in states_to_replace:
#                 state_dic[q2] = q1
#             for q1, q2 in states_to_replace:
#                 transitions.append(
#                     [state_dic[pfsa["state_transition_function"][q1][symbol]] for symbol in pfsa["alphabet"]]
#                     == [state_dic[pfsa["state_transition_function"][q2][symbol]] for symbol in pfsa["alphabet"]]
#                 )
#
#             if all(transitions):
#                 print("Found simplification of states")
#                 for _, state in states_to_replace:
#                     pfsa["states"].remove(state)
#                     pfsa["state_transition_function"].pop(state)
#                     pfsa["symbols_probabilities"].pop(state)
#                 pfsa["state_transition_function"] = {
#                     q1: {k: state_dic[v] for k, v in dic.items()}
#                     for q1, dic in pfsa["state_transition_function"].items()
#                 }
#                 n = 0
#                 break
#         n += -1
#     _is_transition_for_all_states = [
#         len(x.keys()) == len(pfsa["alphabet"]) for x in pfsa["state_transition_function"].values()
#     ]
#     return pfsa
