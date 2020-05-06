# Exprimo

A performance modelling system for deep neural networks. Exprimo builds on [Paleo](https://github.com/TalwalkarLab/paleo) (Qi et al., 2017),
combining it with the technique described in the paper about [Placeto](https://arxiv.org/abs/1906.08879) (Addanki et al., 2019)
to accomodate for more complex model parallelism configurations. Exprimo runs entirely without relying on access to training hardware,
instead relying on FLOP calculations carried out by Paleo to simulate any network on any hardware.

Also included are a number of optimization techniques for finding the best configuration for a given network to run on a given device graph.

## Installation
The project can be installed by running the following steps. It is recommended to do this in a virtual environment in
order to separate it from your global Python installation.

1. Clone this repository by running `git clone --recurse-submodules <URL>`, which ensures that the Paleo submodule is
also cloned.
2. Install requirements with `pip install -r requirements.txt`.
3. Install Paleo by navigating to the `paleo` subdirectory and running `pip install . `.
4. You are now ready to use `exprimo`!

## Optimizing a network
The script `optimize.py` handles optimization of a network. It takes a configuration file as an argument, which contains
information about the network itself, the hardware that it should be optimized for, and the optimizer algorithm to be
used along with its parameters. A set of example configuration files can be cound in the `configs` folder. The script
can be run as a module from the command line: `python -m exprimo.optimize <PATH_TO_CONFIG_FILE>`. Generally, the
MAP-Elites algorithm provides the best results, and can be considered a sensible default.

Note that this project was mainly written as support for a Master's Thesis. The script therefore also produces a lot
of artifacts that are mainly useful in such a context, and the configuration files may contain a lot of options that are
not directly relevant for producing the end result.

## Running the experiments
The source code for all experiments from my Master's thesis are included in the `exprimo.experiments` directory. Each
experiment can be run by executing the corresponding module with `python -m <MODULE_PATH>` from the
root directory. For example, `python -m exprimo.experiments.e1_bandwidth` will run the bandwidth benchmarking 
experiment. Note that you may have to make changes to configuration files in order to make the experiments fit your
system.

## References
* Addanki,  R.,  Venkatakrishnan,  S.  B.,  Gupta,  S.,  Mao,  H.,  and  Alizadeh,  M.
(2019).  Placeto:  Learning generalizable device placement algorithms for distributed machine learning.
arXiv preprint arXiv:1906.08879
* Qi, H., Sparks, E. R., and Talwalkar, A. (2016). Paleo:  A performance model for
deep neural networks.
