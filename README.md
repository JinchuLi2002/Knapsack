# Knapsack (Gatech CS6140 Course Project)
Description: 4 Implementations for the NP-Complete Knapsack Problem
* Branch and Bound
* Approximation Algorithm
  * An approximation ratio of 0.1 is used
* Local Search: Genetic Algorithm
  * A population_size of 300, generations of 100 and mutation_rate of 0.01 is used
* Local Search: Simulated Annealing
  * A temperature of 15000 and cooling_rate of 0.0001 is used

# Usage
```python knapsack.py -inst <filename> -alg [BnB|Approx|LS1|LS2] -time <cutoff in seconds> -seed <random seed>```