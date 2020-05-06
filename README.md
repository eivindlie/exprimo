# Exprimo

A performance modelling system for deep neural networks. Exprimo builds on [Paleo](https://github.com/TalwalkarLab/paleo) (Qi et al., 2017),
combining it with the technique described in the paper about [Placeto](https://arxiv.org/abs/1906.08879) (Addanki et al., 2019)
to accomodate for more complex model parallelism configurations. Exprimo runs entirely without relying on access to training hardware,
instead relying on FLOP calculations carried out by Paleo to simulate any network on any hardware.

Also included are a number of optimization techniques for finding the best configuration for a given network to run on a given device graph.

## Optimizing a network
The script `optimize.py` handles optimization of a network. It takes a configuration file as an argument, which contains
information about the network itself, the hardware that it should be optimized for, and the optimizer algorithm to be
used along with its parameters. A set of example configuration files can be cound in the `configs` folder. The script
can be run as a module from the command line: `python -m exprimo.optimize <PATH_TO_CONFIG_FILE>`. Generally, the
MAP-Elites algorithm provides the best results, and can be considered a sensible default.

Note that this project was mainly written as support for a Master's Thesis. The script therefore also produces a lot
of artifacts that are mainly useful in such a context, and the configuration files may contain a lot of options that are
not directly relevant for producing the end result.

## References
* Addanki,  R.,  Venkatakrishnan,  S.  B.,  Gupta,  S.,  Mao,  H.,  and  Alizadeh,  M.
(2019).  Placeto:  Learning generalizable device placement algorithms for distributed machine learning.
arXiv preprint arXiv:1906.08879
* Qi, H., Sparks, E. R., and Talwalkar, A. (2016). Paleo:  A performance model for
deep neural networks.
