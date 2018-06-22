# Chicharro_PID: Chicharro Trivariate Partial Information Decomposition

This Python module implements the Chicharro trivariate partial information decomposition (Daniel Chicharro, *Quantifying multivariate redundancy with maximum entropy decompositions of mutual information*, 2018; [arXiv 1708.03845](https://arxiv.org/pdf/1708.03845.pdf).).

It uses the Exponential Cone Programming approach described in

* Abdullah Makkeh's PhD thesis (forthcoming (2018))

The software has a similar features with [BROJA_2PID](). So, for now, the following refrence can be used to check some details of the implementation:

* A. Makkeh, D.O. Theis, R. Vicente, *BROJA-2PID: A cone programming based Partial Information Decomposition estimator*, Entropy 2018, 20(4), 271-291; [doi:10.3390/e20040271](http://dx.doi.org/10.3390/e20040271).

#### If you use this software...
...we ask that you give proper reference.
If you use it with only small modifications (note the Apache 2.0 license), use 
```
@Article{makkeh2018broja,
  author =       {Makkeh, Abdullah and Theis, Dirk Oliver and Vicente, Raul},
  title =        {BROJA-2PID: A robust estimator for Bertschinger et al.'s bivariate partial information decomposition},
  journal =      {Entropy},
  volume =    {20},
  number =    {4},
  pages =     {271},
  year =         2018,
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
If you make significant modifications but stick to the approach based on the Exponential Cone Programming model, use
```
@PhdThesis{makkeh:phd:2018,
  author =       {Abdullah Makkeh},
  title =        {Applications of Optimization in Some Complex Systems},
  school =       {University of Tartu},
  year =         {forthcoming}
}
```

#### Files
