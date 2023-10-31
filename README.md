# phiml: A Machine Learning approach to Potential Dynamics

## TODO

* Remove conditionals in simulate_forward
* No bias for signal mapping (Completed)
* Ensure noise is not too great
* Constrained landscape. Repellor at infinity.

## Ideas

* How to handle noise.
* Infer noise parameter.
* Incorporate the number of desired fixed points
* Precompute summary stat for $x_1$ data
* Adjoint method.
* Symmetry in the order of the cells within the population: $\boldsymbol{x}_i$ (Completed?)
* Parallelize batches. (Completed).
* Parallelize individual simulations. (Completed)
* Customizable layer architecture
* batch normalization and dropout
* Autocorrelation time of individual cells to determine transitioning flag. 
* Softmax activation prevents super-linear growth in the potential function.
* Normalize data beforehand?

# References
