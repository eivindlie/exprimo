# Exprimo

A performance modelling system for deep neural networks. Exprimo builds on [Paleo](https://github.com/TalwalkarLab/paleo) (Qi et al., 2017),
combining it with the technique described in the paper about [Placeto](https://arxiv.org/abs/1906.08879) (Addanki et al., 2019)
to accomodate for more complex model parallelism configurations. Exprimo runs entirely without relying on access to training hardware,
instead relying on FLOP calculations carried out by Paleo to simulate any network on any hardware.

## References
* Addanki,  R.,  Venkatakrishnan,  S.  B.,  Gupta,  S.,  Mao,  H.,  and  Alizadeh,  M.
(2019).  Placeto:  Learning generalizable device placement algorithms for distributed machine learning.
arXiv preprint arXiv:1906.08879
* Qi, H., Sparks, E. R., and Talwalkar, A. (2016). Paleo:  A performance model for
deep neural networks.
