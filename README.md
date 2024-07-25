# K-mouflage background solver

This repo contains test notebooks and a python solver for the linear theory quantities (background expansion, scalar field evolution, growth factor) in K-mouflage gravity inspired to the solver of [Hernàndez-Aguayo et al](https://arxiv.org/pdf/2110.00566) but with added flexibility of tuning the free parameter $\lambda$ to reproduce a desired value fot the background expansion in the Einstein or in the Jordan frame. For more details on frame differences and the syntax used in the solver documentation see the companion paper [Sen Gupta et al](https://arxiv.org/abs/2407.00855). 

After comparing the results of several solver approaches with solutions from other authors in [LinearTheory-beta0p2-k1.ipynb](LinearTheory-beta0p2-k1.ipynb) (see [Bose et al](https://arxiv.org/pdf/2406.13667)) we wrapped our solver of choice in [solver.py](solver.py) and provided an example notebook with some usage cases, [Test.ipynb](Test.ipynb). We made good use of our solver by studying the parameter dependence of the linear theory enhancement of the matter power spectrum with respect to a $\Lambda$CDM counterpart (often referred to as the *boost factor*) in [ParameterDependence.ipynb](ParameterDependence.ipynb). The notebook [prepare_sims_folder.ipynb](prepare_sims_folder.ipynb) uses the K-mouflage background solver in congjunction with the Boltzmann solver CLASS to initialise a modified version of the [FML-COLA](https://github.com/HAWinther/FML) N-body code, hosted in the repo [Hi-COLA](https://github.com/Hi-COLACode/Hi-COLA/tree/kmou_forward/HiCOLA), allowing the production of an arbitrary number of simulations with the options of using different initial conditions, cosmologies, and K-mouflage parameters which makes it ideal to produce simulation suites for emulators. 

The other branches in this repo are experimental and not polished, but provide an alternative solver capable of finding solutions also for n>2 (*Solver* branch) and an adaptation of the solver module to a different choice of normalisation of the Plank mass in the Jordan frame ((*M_pl-normalisation* branch)).

If you use the tools in this folder for your research, please cite the companion paper [Sen Gupta et al](https://arxiv.org/abs/2407.00855).
