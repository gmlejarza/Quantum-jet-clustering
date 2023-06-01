# Quantum jet clustering and jet reconstruction at the LHC
Repository with all the code employed for the paper: "Jorge J. Martínez de Lejarza, Leandro Cieri, and Germán Rodrigo Phys. Rev. D 106, 036021 (2022)": https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.036021

Abstract:

Clustering is one of the most frequent problems in many domains, in particular, in
particle physics where jet reconstruction is central in experimental analyses. Jet
clustering at the CERN’s Large Hadron Collider (LHC) is computationally expensive
and the difficulty of this task will increase with the upcoming High-Luminosity LHC
(HL-LHC). In this paper, we study the case in which quantum computing algorithms
might improve jet clustering by considering two novel quantum algorithms which may
speed up the classical jet clustering algorithms. The first one is a quantum subroutine
to compute a Minkowski-based distance between two data points, whereas the second
one consists of a quantum circuit to track the maximum into a list of unsorted data.
The latter algorithm could be of value beyond particle physics, for instance in statistics.
When one or both of these algorithms are implemented into the classical versions of
well-known clustering algorithms (K-means, Affinity Propagation and kT -jet) we obtain
efficiencies comparable to those of their classical counterparts. Even more, exponential
speed-up could be achieved, in the first two algorithms, in data dimensionality and data
length when the distance algorithm or the maximum searching algorithm are applied.
