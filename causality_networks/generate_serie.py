import json
import numpy as np


def correlated_time_series(pfsa_name, serie_length, delay, _weights, _hyperparameters):
    """
    Testing causality networks on time series obtained in the following way: the probability of a symbol to appear in
    the first time serie depends on the symbol that just appeared in the other time serie. This is equivalent to
    defining an XPFSA with a number of states equal to the number of symbols.
    :param pfsa_name: name of the PFSA file used to generate the time series
    :param serie_length: length of the time series to generate
    :param delay: time delay
    :param _weights: probabilities to get the symbols of the first time serie at time t depending on the symbol at time
    t in the other time serie
    :param _hyperparameters: dictionary of the hyperparameters
    """

    _pfsa = load_pfsa(f"data/{pfsa_name}.json")

    serie_i = serie_from_pfsa(_pfsa, serie_length)
    serie_j = [np.random.choice(_pfsa["alphabet"], p=_weights[x]) for x in serie_i]
    serie_j = [np.random.choice(_pfsa["alphabet"]) for _ in range(delay)] + serie_j[:-delay]

    return serie_i, serie_j


def load_pfsa(config_path):
    """
    Loads a PFSA json file.
    :param config_path: json file path
    :return: dictionary of the PFSA properties
    """

    with open(config_path, "r") as f:
        pfsa = json.load(f)

    return pfsa


def pattern_induction_series(pfsa_name, serie_length, patterns, delay, _hyperparameters):
    """
    Testing causality networks on time series obtained in the following way: the symbols appearing in the first serie
    are chosen purely randomly except if a specific pattern appears in the second serie. In this case, a specific
    symbol always appears in the first one.
    :param pfsa_name: name of the PFSA file used to generate the second time serie
    :param serie_length: length of the time series to generate
    :param patterns: dictionary of patterns that directly induce a specific symbol
    :param delay: time delay
    :param _hyperparameters: dictionary of the hyperparameters
    """

    _pfsa = load_pfsa(f"data/{pfsa_name}.json")

    serie_i = serie_from_pfsa(_pfsa, serie_length)
    serie_j = []
    for t in range(len(serie_i)):
        found_pattern = False
        for pattern, induced_symbol in patterns.items():
            if tuple(serie_i[max(0, t - len(pattern)):t]) == tuple(pattern):
                serie_j.append(induced_symbol)
                found_pattern = True
        if not found_pattern:
            serie_j.append(np.random.choice(_pfsa["alphabet"]))
    serie_j = [np.random.choice(_pfsa["alphabet"]) for _ in range(delay)] + serie_j[:-delay]

    return serie_i, serie_j


def same_series_with_time_delay(pfsa_name, serie_length, delay, _hyperparameters):
    """
    Testing causality networks on shifted identical time series.
    :param pfsa_name: name of the PFSA file used to generate the time series
    :param serie_length: length of the time series to generate
    :param delay: time delay
    :param _hyperparameters: dictionary of the hyperparameters
    """

    _pfsa = load_pfsa(f"data/{pfsa_name}.json")

    serie_i = serie_from_pfsa(_pfsa, serie_length)
    serie_j = [np.random.choice(_pfsa["alphabet"]) for _ in range(delay)] + serie_i[:-delay]
    return serie_i, serie_j


def serie_from_pfsa(pfsa, serie_length):
    """
    Randomly generates a discrete time serie from the dictionary of a PFSA properties.
    :param pfsa: dictionary of the PFSA properties
    :param serie_length: length of the time serie to generate
    :return: generated time serie
    """

    serie = []
    weights_per_state = {state: [pfsa["symbols_probabilities"][state][symbol] for symbol in pfsa["alphabet"]] for state
                         in pfsa["states"]}
    current_node = np.random.choice(pfsa["states"])
    for _ in range(serie_length):
        next_symbol = np.random.choice(pfsa["alphabet"], p=weights_per_state[current_node])
        current_node = pfsa["state_transition_function"][current_node][next_symbol]
        serie.append(next_symbol)

    return serie


def series_from_xpfsa(xpfsa_name, serie_length, _hyperparameters):
    """
    Randomly generates a discrete time serie from the dictionary of a XPFSA properties and a given source time serie.
    :param xpfsa_name: dictionary of the XPFSA properties
    :param serie_length: length of the time series to generate
    :param _hyperparameters: dictionary of the hyperparameters
    :return: generated time serie
    """

    xpfsa = load_pfsa(f"data/{xpfsa_name}.json")

    serie_i = []
    serie_j = []
    weights_per_state_in = {state: [xpfsa["symbols_probabilities_in"][state][symbol] for symbol in
                                    xpfsa["alphabet_in"]] for state in xpfsa["states"]}
    weights_per_state_out = {state: [xpfsa["symbols_probabilities_out"][state][symbol] for symbol in
                                     xpfsa["alphabet_out"]] for state in xpfsa["states"]}
    current_node = np.random.choice(xpfsa["states"])
    for _ in range(serie_length):
        next_symbol_i = np.random.choice(xpfsa["alphabet_in"], p=weights_per_state_in[current_node])
        next_symbol_j = np.random.choice(xpfsa["alphabet_out"], p=weights_per_state_out[current_node])
        current_node = xpfsa["state_transition_function"][current_node][next_symbol_i]
        serie_i.append(next_symbol_i)
        serie_j.append(next_symbol_j)

    return serie_i, serie_j
