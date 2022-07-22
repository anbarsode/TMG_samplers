#include <iostream>
#include <random>
#include <fstream>

int main(int argc, char **argv)
{
    double mu = std::stod(argv[1]);
    double sig = std::stod(argv[2]);
    double x_min = std::stod(argv[3]);
    double x_max = std::stod(argv[4]);
    long N = std::stol(argv[5]);
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<double> G1D(mu, sig);
    
    double x;
    std::ofstream fp("./T1DG_rejection.samples");
    long i = 0, rej = 0;
    while(i < N)
    {
        x = G1D(rng);
        if(x >= x_min && x <= x_max)
        {
            fp << x << std::endl;
            i++;
        }
        else rej++;
    };
    fp.close();
    std::cout << N / float(N + rej) << std::endl;
}
