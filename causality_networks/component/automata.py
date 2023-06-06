import random
import networkx as nx


class PFSA:
    """
    Class object for a Probabilistic Finite State Automaton (PFSA) or Crossed Probabilistic Finite State Automaton
    (XPFSA).
    """

    def __init__(self, inputs: [dict, nx.DiGraph], automata_type):

        self.type = automata_type
        if isinstance(inputs, nx.DiGraph):
            self.graph = inputs
            _pfsa_dic = self.extract_properties_from_graph()
            self.states = _pfsa_dic["states"]
            self.states_words = _pfsa_dic["states_words"]
            self.alphabet_source = _pfsa_dic["alphabet_source"]
            self.alphabet_target = _pfsa_dic["alphabet_target"]
            self.state_transitions = _pfsa_dic["states_transitions"]
            self.symbols_probabilities = _pfsa_dic["symbols_probabilities"]
        else:
            self.states = inputs["states"]
            self.states_words = inputs["states_words"]
            self.alphabet_source = inputs["alphabet_source"]
            self.alphabet_target = inputs["alphabet_target"]
            self.state_transitions = inputs["states_transitions"]
            self.symbols_probabilities = inputs["symbols_probabilities"]
            self.graph = self.build_graph()

    def __str__(self):

        _states_str = f"{len(self.states)} states : " + ", ".join(self.states) + "\n"
        if self.type == "PFSA":
            _alphabet_str = f"{len(self.alphabet_source)} symbols : " + ", ".join(self.alphabet_source) + "\n"
        else:
            _alphabet_str = f"Source: {len(self.alphabet_source)} symbols : " + ", ".join(self.alphabet_source) + "\n"
            _alphabet_str += f"Target: {len(self.alphabet_target)} symbols : " + ", ".join(self.alphabet_target) + "\n"
        _transitions_str = "transitions:\n"
        for from_state, dic in self.state_transitions.items():
            for symbol, to_state in dic.items():
                _transitions_str += \
                    f"\t{from_state} ---({symbol})---> {to_state}\t{self.symbols_probabilities[from_state][symbol]}\n"
        return _states_str + _alphabet_str + _transitions_str

    def build_graph(self):
        """
        Builds a networkx graph from the properties of the automaton.
        :return: networkx DiGraph object
        """

        graph = nx.DiGraph()
        for state in self.states:
            if self.type == "XPFSA":
                graph.add_node(state, word=",".join(self.states_words[state]),
                               probabilities=self.symbols_probabilities[state])
            else:
                graph.add_node(state, word=",".join(self.states_words[state]))

        for from_state, state_dic in self.state_transitions.items():
            for symbol, to_state in state_dic.items():
                if not graph.has_edge(from_state, to_state):
                    if self.type == "XPFSA":
                        graph.add_edge(from_state, to_state, symbol=[symbol])
                    else:
                        prob = self.symbols_probabilities[from_state][symbol]
                        graph.add_edge(from_state, to_state, symbol=[symbol], probability=[prob])
                else:
                    _edge_symbols = nx.get_edge_attributes(graph, "symbol")[(from_state, to_state)]
                    if self.type == "XPFSA":
                        graph.add_edge(from_state, to_state, symbol=[symbol])
                        graph.remove_edge(from_state, to_state)
                        graph.add_edge(from_state, to_state, symbol=_edge_symbols + [symbol])
                    else:
                        prob = self.symbols_probabilities[from_state][symbol]
                        _edge_probability = nx.get_edge_attributes(graph, "probability")[(from_state, to_state)]
                        graph.remove_edge(from_state, to_state)
                        graph.add_edge(from_state, to_state, symbol=_edge_symbols + [symbol],
                                       probability=_edge_probability + [prob])
        return graph

    def extract_properties_from_graph(self):
        """
        Extracts the dictionary of the properties of the automaton from its graph.
        :return: dictionary of the automaton properties
        """

        states = list(self.graph.nodes)
        states_words = {k: v.split(",") for k, v in nx.get_node_attributes(self.graph, "word").items()}
        alphabet_source = list(set([y for x in nx.get_edge_attributes(self.graph, "symbol").values() for y in x]))
        if self.type == "XPFSA":
            alphabet_target = list(set([y for x in nx.get_node_attributes(self.graph, "probabilities").values() for y
                                        in x]))
        else:
            alphabet_target = alphabet_source.copy()
        state_transitions = {q: {} for q in states}
        symbols_probabilities = {q: {} for q in states}
        _transitions = nx.get_edge_attributes(self.graph, "symbol")
        for (s1, s2), symbols in _transitions.items():
            for symbol in symbols:
                state_transitions[s1][symbol] = s2
        for (s1, s2), probabilities in nx.get_edge_attributes(self.graph, "probability").items():
            for i, proba in enumerate(probabilities):
                symbols_probabilities[s1][_transitions[s1, s2][i]] = proba
        return {"states": states, "states_words": states_words, "alphabet_source": alphabet_source,
                "alphabet_target": alphabet_target, "states_transitions": state_transitions,
                "symbols_probabilities": symbols_probabilities}

    def streamrun_word(self, sequence):
        """
        Computation of the streamrun function for a given word as implemented in algorithm 1 of the paper
        'Causality networks' by Chattopadhyay.
        :param sequence: word to streamrun
        :return: probability of being in a given state for the given word
        """

        u = {state: 0 for state in self.states}
        current_state = self.states[random.randint(0, len(self.states) - 1)]
        u[current_state] += 1
        for symbol in sequence:
            current_state = self.state_transitions[current_state][symbol]
            u[current_state] += 1

        u = {state: x / (len(sequence) + 1) for state, x in u.items()}
        return u
