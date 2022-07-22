# TMG_samplers
Numpy and C++ based samplers for drawing random numbers from multivariate normal distributions having hard cuts (truncations) on each variable

Sampling from a truncated distribution using rejection sampling can be a pain, especially when acceptance rate is very low.
A python-numpy based implentation can take upto 4 minutes for drawing $10^5$ samples in 10D when acceptance rates are $\sim0.5$%.
A C++ version can do this in 7 seconds!

This repo contains some functions written in python-numpy and C++ that implement rejection sampling as well as the Gibbs sampling technique.
Here is a quick summary:
- Numpy based Kotecha-Djuric method is the fastest for sampling from a truncated Gaussian in 1D
- For multivariate normal sampling, Kotecha-Djuric-Gibbs method (the way I have implemented it) is not reliable
- C++ based rejection sampling is the fastest and most reliable method for sampling from a truncated multidimensional gaussian

Requirements:
- Python, Numpy, Scipy
- C++
- Eigen <https://eigen.tuxfamily.org/index.php?title=Main_Page>

Installation (may write better instructions later):
- Download the repo
- Open `TMG_samplers.py` in edit mode
- At the top, edit the two fields `Eigen_path` and `TMG_path`
  * `Eigen_path` should point to the directory `Eigen` (do `$ pwd` from inside the extracted tarball)
  * `TMG_path` should point to the directory where all the C++ files are kept (do `$ pwd` from inside the cloned repo)
- Copy the `TMG_samplers.py` in your current working directory and then you can do `from TMG_samplers import TMG` in your code to start using it
- You may need to add the flag `-std=c++11` and compile separately if you get any errors during compilation of the C++ codes

Running it for the first time may be slow because the program will first compile the C++ codes.

Take a look at `testing_TMG_samplers.ipynb` for examples.

References:
* Gibbs Sampling Approach For Generation Of Truncated Multivariate Gaussian Random Variables - Jayesh H. Kotecha and Petar M. Djuric  
* https://en.wikipedia.org/wiki/Normal_distribution  
* https://en.wikipedia.org/wiki/Gibbs_sampling  
* https://en.wikipedia.org/wiki/Multivariate_normal_distribution  
