# SeaPy Tutorials:
The following list of tutorials are prepared to demonstrate some of the key functionalities of the SeaPy package. The main goal is to familiarize the user with the syntax, notation and the conventions that have been used in developing the SeaPy. We have included some of the derivations that we think are handy for everyday use or might help in checking the answers. Step by step calculations are provided along with the results of the code so that the user can follow the steps and troubleshoot the possible inconsistencies. Please refer to the SeaPy documentation for a detailed list of functions, variables and definitions.

## Educational Example: Dislocations in two dimensions

![Dislocation structure](/Tutorials/511burg2.gif)

For demonstrating SeaPy's functionalities, we will utilize a very simple and educational example, that of dislocation pairs nucleating or annihilating in a thin film. It is a characteristic example, given that it can be clearly seen what the produced "fingerprints" really are, and why they represent a wealth of more information than typical imaging approaches in micromechanics.

Edge dislocations in two dimensions have an exactly solvable strain field. If one considers only the first strain invariant, for every dislocation, one considers the dislocation pressure:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;I_1(\epsilon)=\frac{(1+\nu)b}{3\pi(1-\nu)}\frac{y}{x^2+y^2}" title="First strain invariant / Pressure" />

Dislocation pressure in edge dislocation systems is related to the first strain invariant in space, for isotropic elastic materials. In particular,

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sigma_{zz}=2G\epsilon_{zz}+\lambda\sum_i\epsilon_{ii}" title="Isotropic / Elasticity" />

Then, the first strain invariant can be found analytically,

<img src="https://latex.codecogs.com/svg.latex?\Large&space;I^{+}_\epsilon(r)=\sigma_{zz}/\lambda=-b\frac{(1-2\nu)\nu}{2\pi(1-\nu)}\frac{y}{x^2+y^2}" title="First strain invariant / Pressure" />

for a positive dislocation of Burgers vector b.

If one assumes a strain loading profile ``ε(t)=f(t)``, then the desired evolution of the first strain invariant becomes,

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{dI^{\pm}_\epsilon(r)}{dt}=\mp\frac{2f'(t)(x-x_0+f(t))}{(y-y_0)^2+(x-x_0+f(t))^2}I_\epsilon(r) (1)" title="First strain invariant / Evolution" />

The numerical evolution of the strain invariant is shown below:

![MovieDisl2Glide](/Images/disl2_fast.gif)

## Microstructural fingerprinting

As one can see from Eq.(1), the evolution of the strain invariant has no spatial or temporal (short) correlations. Naturally, one should be able to infer the properties of the evolution matrix analytically, distinguishing two regimes, depending which time interval is considered.

The numerical calculation of SEA provides an automatic way to infer the property of the evolution matrix:

![MovieDislEIM_Mode](/Images/Fig5_0th-StrainInvariantMode.png)


## Convolutional Neural Networks For small deformations: Distinguishing Dislocation Glide from Dislocation Nucleation

The principal reason why convolutional neural networks are the most appropriate approach in strain fingerprint recognition is the Small Deformations' Superposition principle that states the ability of reconstructing the deformation features of a solid by considering appropriate superpositions of co-existing strain solutions. Weighted superpositions are the principal building components of convolutional neural network methods:

![NeuronDescription](/Images/figures_ArtificialNeuron.png)

![ResultToBeSoughtAfterRunningTheTutorial](/Images/Plot_WeightsIndex.png)

### Copyright (c) 2019, Stefanos Papanikolaou.
