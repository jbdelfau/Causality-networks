# Causality networks

## Author
Jean-Baptiste Delfau

## Description
This library is a Python implementation of the algorithms GenESeSS and xGenESeSS published and described by I. Chattopadhyay et al in the following articles:
- <em>Causality networks</em>, Chattopadhyay, arXiv, <strong>2014</strong>.
- <em>Abductive learning of quantized stochastic processes with probabilistic finite automata</em>, Chattopadhyay et al, Phil Trans R Soc A, <strong>2016</strong>
- <em>Event-level prediction of urban crime reveals a signature of enforcement bias in US cities</em>, Rotaru et al, Nature human Behaviour, <strong>2022</strong>

Its main purpose is to infer a Probabilistic Finite State Automaton (PFSA) or Crossed Probabilistic Finite State Automaton (XPFSA) from discrete time series. The analysis of these automata can then bring into light the existence of causal dependence - in the sense of Granger - between the events of the time series. 

## Usage
Examples can be found in file <em>test_algo.py</em> of the tests folder. In short, one needs to call the function granger_network() with the following arguments:
- a dictionary of time series likely to be correlated
- a dictionary of hyperparameters
- the maximum length of the sequences computed by the algorithm
- the title of the experiment


