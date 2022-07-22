#include <iostream>
#include <random>
#include <math.h>
#include <fstream>
#include <Eigen/Dense>

#include "TMG_utils.cpp"

int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    
    double mu = std::stod(argv[1]);
    double sig = std::stod(argv[2]);
    double x_min = std::stod(argv[3]);
    double x_max = std::stod(argv[4]);
    long N = std::stol(argv[5]);
    
    double y[N];
    double* yptr = T1DG_KD(rng, y, N, mu, sig, x_min, x_max);
    
    std::ofstream fp("./T1DG_KD.samples");
    for(long i=0; i < N; i++)
    {
        fp << yptr[i] << std::endl;
    }
    fp.close();
}
