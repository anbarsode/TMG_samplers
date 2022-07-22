#include <iostream>
#include <random>
#include <math.h>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "TMG_utils.cpp"

int main(int argc, char **argv)
{
    // Read Inputs
    int Ndim = std::stoi(argv[1]);
    Eigen::MatrixXd cov(Ndim,Ndim);
    Eigen::VectorXd mu(Ndim,1), x_min(Ndim,1), x_max(Ndim,1);
    for(int i=0; i < Ndim; i++)
        mu(i) = std::stod(argv[1+1 + i]);
    for(int i=0; i < Ndim; i++)
        for(int j=0; j < Ndim; j++)
            cov(i,j) = std::stod(argv[1+1 + Ndim + Ndim*i+j]);
    for(int i=0; i < Ndim; i++)
        x_min(i) = std::stod(argv[1+1 + Ndim + Ndim*Ndim + i]);
    for(int i=0; i < Ndim; i++)
        x_max(i) = std::stod(argv[1+1 + Ndim + Ndim*Ndim + Ndim + i]);
    long N = std::stol(argv[1+1 + Ndim + Ndim*Ndim + Ndim + Ndim]);
    
    normal_random_variable sample {mu, cov};
    
    Eigen::VectorXd x_now(Ndim,1);
    bool valid;
    std::ofstream fp("./TMG_rejection.samples");
    
    long i = 0, rej = 0;
    while(i < N)
    {
        x_now = sample();
        valid = true;
        for(int j=0; j < Ndim; j++)
            if(x_now(j) < x_min(j) || x_now(j) > x_max(j)) valid = false;
        if(valid)
        {
            for(int j=0; j < Ndim; j++)
                fp << x_now(j) << " ";
            fp << std::endl;
            i++;
        }
        else rej++;
    };
    fp.close();
    std::cout << N / float(N + rej) << std::endl;
}
