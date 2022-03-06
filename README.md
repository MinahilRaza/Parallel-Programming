# Parallel-Programming

A set of assignments for parallel programming in multi-core, many-core and multi-gpu systems. 

# Galaxy Problem
Many cosmological calculations require independent calculations of the same quantity for all data points, which makes them ideal candidates for parallelization (SIMD). One such problem is the calculation of **Two-point angular correlation function** between real and synthetic galaxies. The mathematical formula presented in [1] uses three quantities- DD (data-data), RR (random-random), and DR (data-random) for the calculation of angular correlation.
 

 $w(\theta) =  \frac{DD -2DR + RR}{RR}$

DD, DR and RR are calculated from four angles for each point in the real and synthetic galaxy. Thus there are two angles, $\alpha and \beta $ representing the values for ascention and declination respectively. For example, in order to calculate DR, 

$\cos(\theta) = \sin(\beta_1)*\sin(\beta_2)+\cos(\beta_1)*\cos(\beta_2)*\cos(\alpha_1-\alpha_2)$

# Single Instruction, Multiple Data
The calculation of DD, DR and RR can be accelerated using multiple cores or GPUs. For a catalogue of n galaxies, there are $\frac{n(n-1)}{2}$ calculations. The full calculation of DD, RR, and DR is effectively an $O(n^2)$ operation. The same formula is applied over all $n$ elements of the array of angles for each galaxy. Thus, the calculations can be split easily.

# Parallelization
In the provided files, the same galaxy problem has been accelerated using mutliple methods

* OpenMP (for multiple cores on the same node)
* MPI (for multiple cores on different nodes)
* CUDA (for single and multi-gpu nodes)

In addition, a hello_world file has been provided for multi-gpu execution as that was a challenge I faced while learning CUDA.

# Execution Times and Speed-ups

| Parallelism Technique         | Execution time (seconds)      | # Speedup |
|--------------|-----------|------------|
|No parallelism | 1960.1      | *1*        |
| OpenMP (40 cores)      | 171.3  | 11.44       |
| MPI (40 ranks)      | 84.54  | 23.18       |
| CUDA (1 GPU)      | 6.18  | 317.16       |
| CUDA (3 GPUs)      | 3.05  | **642.66**       |

# Data
To access the data, contact me via email: minahilrz@gmail.com

# References
[1] D. Bard, M. Bellis, M.T.Allen, H. Yepremyan, J.M. Kratochvil,
_“Cosmological calculations on the GPU”_, Astronomy and
Computing, Vol. 1 (2013) 17-22