from causality_networks.find_pfsa_from_serie import granger_network
from causality_networks.generate_serie import *


hyperparameters = {"epsilon": 0.05, "min_occurence": 10, "max_delay": 2, "coefficient_threshold": 0.1,
                   "inference_length": None}
serie_length = 50000


def test_time_shift_1():

    delay = 3
    window_title = f"Delay of {delay} in the same time serie, 2 symbols"
    print(window_title)
    serie_i, serie_j = same_series_with_time_delay("PFAS_2symbols_4states_config", serie_length, delay, hyperparameters)
    word_max_length = int(np.ceil(np.emath.logn(2, (1 / hyperparameters["epsilon"]))))
    gamma, _ = granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)
    assert all([abs(gamma["i", "j", delay - 1] - 1) <= 0.99, abs(gamma["j", "i", delay - 1]) <= 0.01])


def test_time_shift_2():

    delay = 2
    window_title = f"Delay of {delay} in the same time serie, 3 symbols"
    print(window_title)
    serie_i, serie_j = same_series_with_time_delay("PFAS_3symbols_4states_config", serie_length, delay, hyperparameters)
    word_max_length = int(np.ceil(np.emath.logn(3, (1 / hyperparameters["epsilon"]))))
    gamma, _ = granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)
    assert all([abs(gamma["i", "j", delay - 1] - 1) <= 0.99, abs(gamma["j", "i", delay - 1]) <= 0.01])


def test_correlations_with_weights_1():

    delay = 2
    window_title = f"Correlated time series with specific weight and time delay of {delay}, 2 symbols"
    print(window_title)
    word_max_length = int(np.ceil(np.emath.logn(2, (1 / hyperparameters["epsilon"]))))
    weights = {"0": [0.2, 0.8], "1": [0.9, 0.1]}
    serie_i, serie_j = correlated_time_series("PFAS_2symbols_4states_config", serie_length, delay, weights,
                                              hyperparameters)
    gamma, _ = granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)
    assert gamma["i", "j", delay - 1] >= 0.1


def test_correlations_with_weights_2():

    delay = 3
    window_title = f"Correlated time series with specific weight and time delay of {delay}, 3 symbols"
    print(window_title)
    word_max_length = int(np.ceil(np.emath.logn(3, (1 / hyperparameters["epsilon"]))))
    weights = {"0": [0.2, 0.7, 0.1], "1": [0.85, 0.1, 0.05], "2": [0.7, 0.1, 0.2]}
    serie_i, serie_j = correlated_time_series("PFAS_3symbols_4states_config", serie_length, delay, weights,
                                              hyperparameters)
    gamma, _ = granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)
    assert gamma["i", "j", delay - 1] >= 0.1


def test_xpfsa_1():

    window_title = f"Correlated time series, XPFSA 1st case"
    print(window_title)
    word_max_length = int(np.ceil(np.emath.logn(2, (1 / hyperparameters["epsilon"]))))
    serie_i, serie_j = series_from_xpfsa("XPFSA_2_to_3_symbols_3states_config", serie_length, hyperparameters)
    gamma, _ = granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)
    assert gamma["i", "j", 0] >= 0.1


def test_xpfsa_2():

    window_title = f"Correlated time series, XPFSA 2nd case"
    print(window_title)
    word_max_length = int(np.ceil(np.emath.logn(2, (1 / hyperparameters["epsilon"]))))
    serie_i, serie_j = series_from_xpfsa("XPFSA_2_to_4_symbols_2states_config", serie_length, hyperparameters)
    gamma, _ = granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)
    assert gamma["i", "j", 0] >= 0.1


# The following tests are either not conclusive or robust enough so we do not use asserts here.
def pattern_induction_1():

    delay = 1
    window_title = f"Correlated time series with patterns induction, max memory of 2"
    print(window_title)
    word_max_length = int(np.ceil(np.emath.logn(2, (1 / hyperparameters["epsilon"]))))
    patterns = {("0", "0"): "0", ("1"): "1"}
    serie_i, serie_j = pattern_induction_series("PFAS_2symbols_1state_config", serie_length, patterns, delay,
                                                hyperparameters)
    granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)


def pattern_induction_2():

    delay = 1
    window_title = f"Correlated time series with patterns induction, max memory of 3"
    print(window_title)
    word_max_length = int(np.ceil(np.emath.logn(2, (1 / hyperparameters["epsilon"]))))
    patterns = {("0", "0", "1"): "1"}
    serie_i, serie_j = pattern_induction_series("PFAS_2symbols_1state_config", serie_length, patterns, delay,
                                                hyperparameters)
    granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)


def xpfsa_identical_states():

    window_title = f"Correlated time series, XPFSA identical states"
    print(window_title)
    word_max_length = int(np.ceil(np.emath.logn(2, (1 / hyperparameters["epsilon"]))))
    serie_i, serie_j = series_from_xpfsa("XPFSA_2_to_3_symbols_3states_config_close_states", serie_length,
                                         hyperparameters)
    granger_network({"i": serie_i, "j": serie_j}, hyperparameters, word_max_length, window_title)
