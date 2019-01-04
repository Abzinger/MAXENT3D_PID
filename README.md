# MAXENT3D_PID: Maximum Entropy Trivariate Partial Information Decomposition

This Python module implements the maximum entropy trivariate partial information decomposition (Daniel Chicharro, *Quantifying multivariate redundancy with maximum entropy decompositions of mutual information*, 2018; [arXiv 1708.03845](https://arxiv.org/pdf/1708.03845.pdf).).

It uses the Exponential Cone Programming approach described in
* Abdullah Makkeh's PhD thesis, *Applications of Optimization in Some Complex Systems*, 2018; [ISBN 978-9949-77-781-5](https://dspace.ut.ee/handle/10062/61143).

The software has similar features with [BROJA_2PID](https://github.com/Abzinger/BROJA_2PID/). So, for now, the following reference can be used to check some details of the implementation
* A. Makkeh, D.O. Theis, R. Vicente, *BROJA-2PID: A cone programming based Partial Information Decomposition estimator*, Entropy 2018, 20(4), 271-291; [doi:10.3390/e20040271](http://dx.doi.org/10.3390/e20040271).

To **get started** have a look at the [documentation](http://maxent3d-pid.rtfd.io). For further discussions, contact abdullah`dot`makkeh`at`gmail`dot`com.

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
  title =        {Applications of Optimization in Some Complex Systems},
  author =       {Abdullah Makkeh},
  year =         {2018},
  school =       {Ph. D. Thesis, University of Tartu, Tartu, Estonia}
}
```
#### Contributors

* [Abdullah Makkeh](https://www.theory.cs.ut.ee/people/abdullah-makkeh), Algorithms & theory, Institute of Computer Science, University of Tartu, Tartu, Estonia.

* [Dirk Oliver Theis](https://www.theory.cs.ut.ee/people/dot), Algorithms & theory, Institute of Computer Science, University of Tartu, Tartu, Estonia.

* [Raul Vicente](https://neuro.cs.ut.ee/people/), Computational Neuroscience Lab, Institute of Computer Science, University of Tartu, Tartu, Estonia.

* [Daniel Chicharro](https://www.iit.it/advanced-robotics-people/daniel-chicharro), Cognitive Neuropsychology Laboratory, Department of Phsycology, Harvard University, Massachusetts, United States.
#### Files

