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
    double x_init[Ndim];
    for(int i=0; i < Ndim; i++)
        mu(i) = std::stod(argv[1+1 + i]);
    for(int i=0; i < Ndim; i++)
        for(int j=0; j < Ndim; j++)
            cov(i,j) = std::stod(argv[1+1 + Ndim + Ndim*i+j]);
    for(int i=0; i < Ndim; i++)
        x_min(i) = std::stod(argv[1+1 + Ndim + Ndim*Ndim + i]);
    for(int i=0; i < Ndim; i++)
        x_max(i) = std::stod(argv[1+1 + Ndim + Ndim*Ndim + Ndim + i]);
    for(int i=0; i < Ndim; i++)
        x_init[i] = std::stod(argv[1+1 + Ndim + Ndim*Ndim + Ndim + Ndim + i]);
    long N = std::stol(argv[1+1 + Ndim + Ndim*Ndim + Ndim + Ndim + Ndim]);
    int burnin = std::stoi(argv[1+1 + Ndim + Ndim*Ndim + Ndim + Ndim + Ndim + 1]);
    int hop = std::stoi(argv[1+1 + Ndim + Ndim*Ndim + Ndim + Ndim + Ndim + 1 + 1]);
    
    // Precalculate certain quantities
    Eigen::VectorXd M1(Ndim,1), C11(Ndim,1), C_eff(Ndim,1);
    Eigen::MatrixXd M2(Ndim,Ndim-1), C12(Ndim,Ndim-1);
    Eigen::MatrixXd *C22_inv = new Eigen::MatrixXd[Ndim];
    for(int i = 0; i < Ndim; i++)
        C22_inv[i]= Eigen::MatrixXd::Zero(Ndim-1,Ndim-1);
    
    for(int i=0; i < Ndim; i++)
    {
        M1(i) = mu(i);
        C11(i) = cov(i,i);
    }
    
    for(int i=0; i < Ndim; i++)
    {
        int skip = 0;
        for(int j=0; j < Ndim; j++)
            if(i != j)
            {
                M2(i,j-skip) = mu(j);
                C12(i,j-skip) = cov(i,j);
            }
            else skip++;
    }
    
    for(int k=0; k < Ndim; k++)
    {
        int skipi = 0;
        for(int i=0; i < Ndim; i++)
        {
            int skipj = 0;
            if(k != i)
                for(int j=0; j < Ndim; j++)
                    if(k != j) C22_inv[k](i-skipi,j-skipj) = cov(i,j);
                    else skipj++;
            else skipi++;
        }
        C22_inv[k] = C22_inv[k].inverse();
    }
    
    for(int i=0; i < Ndim; i++)
        C_eff(i) = sqrt(C11(i) - C12(i,Eigen::all) * C22_inv[i] * C12(i,Eigen::all).transpose());
    
    // Initialize rng
    std::random_device rd;
    std::mt19937 rng(rd());
    
    double x_now[Ndim], mu_eff, sig_eff;
    for(int i=0; i < Ndim; i++) x_now[i] = x_init[i];
    double y[1], *yptr;
    int iters = 0;
    long i=0;
    
    std::ofstream fp("./TMG_KDGibbs.samples");
    while(i < N)
    {
        for(int ind=0; ind < Ndim; ind++)
        {
            mu_eff = mu_effective(ind, Ndim, x_now, M1, M2, C11, C12, C22_inv);
            sig_eff = C_eff(ind);
            yptr = T1DG_KD(rng, y, 1, mu_eff, sig_eff, x_min(ind), x_max(ind));
            x_now[ind] = yptr[0];
        }
        
        iters++;
        
        if(iters > burnin && (iters - burnin) % hop == 0)
        {
            for(int j=0; j < Ndim; j++)
                fp << x_now[j] << " ";
            fp << std::endl;
            i++;
        }
    };
    fp.close();
}
