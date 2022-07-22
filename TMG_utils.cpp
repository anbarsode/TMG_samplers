// From https://github.com/abrari/block-cipher-testing/blob/master/stats.c
double erfinv(double x) {

    // beware that the logarithm argument must be
    // commputed as (1.0 - x) * (1.0 + x),
    // it must NOT be simplified as 1.0 - x * x as this
    // would induce rounding errors near the boundaries +/-1
    double w = - log((1.0 - x) * (1.0 + x));
    double p;

    if (w < 6.25) {
        w -= 3.125;
        p =  -3.6444120640178196996e-21;
        p =   -1.685059138182016589e-19 + p * w;
        p =   1.2858480715256400167e-18 + p * w;
        p =    1.115787767802518096e-17 + p * w;
        p =   -1.333171662854620906e-16 + p * w;
        p =   2.0972767875968561637e-17 + p * w;
        p =   6.6376381343583238325e-15 + p * w;
        p =  -4.0545662729752068639e-14 + p * w;
        p =  -8.1519341976054721522e-14 + p * w;
        p =   2.6335093153082322977e-12 + p * w;
        p =  -1.2975133253453532498e-11 + p * w;
        p =  -5.4154120542946279317e-11 + p * w;
        p =    1.051212273321532285e-09 + p * w;
        p =  -4.1126339803469836976e-09 + p * w;
        p =  -2.9070369957882005086e-08 + p * w;
        p =   4.2347877827932403518e-07 + p * w;
        p =  -1.3654692000834678645e-06 + p * w;
        p =  -1.3882523362786468719e-05 + p * w;
        p =    0.0001867342080340571352 + p * w;
        p =  -0.00074070253416626697512 + p * w;
        p =   -0.0060336708714301490533 + p * w;
        p =      0.24015818242558961693 + p * w;
        p =       1.6536545626831027356 + p * w;
    } else if (w < 16.0) {
        w = sqrt(w) - 3.25;
        p =   2.2137376921775787049e-09;
        p =   9.0756561938885390979e-08 + p * w;
        p =  -2.7517406297064545428e-07 + p * w;
        p =   1.8239629214389227755e-08 + p * w;
        p =   1.5027403968909827627e-06 + p * w;
        p =   -4.013867526981545969e-06 + p * w;
        p =   2.9234449089955446044e-06 + p * w;
        p =   1.2475304481671778723e-05 + p * w;
        p =  -4.7318229009055733981e-05 + p * w;
        p =   6.8284851459573175448e-05 + p * w;
        p =   2.4031110387097893999e-05 + p * w;
        p =   -0.0003550375203628474796 + p * w;
        p =   0.00095328937973738049703 + p * w;
        p =   -0.0016882755560235047313 + p * w;
        p =    0.0024914420961078508066 + p * w;
        p =   -0.0037512085075692412107 + p * w;
        p =     0.005370914553590063617 + p * w;
        p =       1.0052589676941592334 + p * w;
        p =       3.0838856104922207635 + p * w;
    } else if (w != INFINITY) {
        w = sqrt(w) - 5.0;
        p =  -2.7109920616438573243e-11;
        p =  -2.5556418169965252055e-10 + p * w;
        p =   1.5076572693500548083e-09 + p * w;
        p =  -3.7894654401267369937e-09 + p * w;
        p =   7.6157012080783393804e-09 + p * w;
        p =  -1.4960026627149240478e-08 + p * w;
        p =   2.9147953450901080826e-08 + p * w;
        p =  -6.7711997758452339498e-08 + p * w;
        p =   2.2900482228026654717e-07 + p * w;
        p =  -9.9298272942317002539e-07 + p * w;
        p =   4.5260625972231537039e-06 + p * w;
        p =  -1.9681778105531670567e-05 + p * w;
        p =   7.5995277030017761139e-05 + p * w;
        p =  -0.00021503011930044477347 + p * w;
        p =  -0.00013871931833623122026 + p * w;
        p =       1.0103004648645343977 + p * w;
        p =       4.8499064014085844221 + p * w;
    } else {
        // this branch does not appear in the original code, it
        // was added because the previous branch does not handle
        // x = +/-1 correctly. In this case, w is positive infinity
        // and as the first coefficient (-2.71e-11) is negative.
        // Once the first multiplication is done, p becomes negative
        // infinity and remains so throughout the polynomial evaluation.
        // So the branch above incorrectly returns negative infinity
        // instead of the correct positive infinity.
        p = INFINITY;
    }

    return p * x;
}

const double SQRT_TWO = sqrt(2.0);

double G1DCDF(double x, double mu, double sig)
{
    return 0.5 * (1.0 + erf((x - mu) / sig / SQRT_TWO));
}


double G1DCDF_inv(double y, double mu, double sig)
{
    double x = 2.0 * y - 1.0;
    x = erfinv(x);
    x *= sig * SQRT_TWO;
    x += mu;
    return x;
}

double* T1DG_KD(std::mt19937 &rng, double* y, long N, double mu, double sig, double minval, double maxval)
{
    double x;
    std::normal_distribution<double> G1D(mu, sig);
    for(long i=0; i < N; i++)
    {
        x = G1D(rng);
        //std::cout << mu << " " << sig << " " << x << std::endl;
        //std::cout << G1DCDF(x, mu, sig) << " " << G1DCDF(maxval, mu, sig) << " " << G1DCDF(minval, mu, sig) << std::endl;
        x = G1DCDF(x, mu, sig) * (G1DCDF(maxval, mu, sig) - G1DCDF(minval, mu, sig)) + G1DCDF(minval, mu, sig);
        //std::cout << x << std::endl;
        //std::cout << G1DCDF_inv(x, mu, sig) << " #" << std::endl;
        y[i] = G1DCDF_inv(x, mu, sig);
    }
    return y;
}

double mu_effective(int ind, int Ndim, double* x, Eigen::VectorXd M1, Eigen::MatrixXd M2, Eigen::VectorXd C11, Eigen::MatrixXd C12, Eigen::MatrixXd* C22_inv)
{
    Eigen::VectorXd x_cond(Ndim-1,1);
    double mu_eff;
    int skip = 0;
    for(int i=0; i < Ndim; i++)
        if(i != ind) x_cond(i-skip) = x[i];
        else skip++;
    mu_eff = M1(ind) + C12(ind,Eigen::all) * C22_inv[ind] * (x_cond - M2(ind,Eigen::all).transpose());
    return mu_eff;
}


// From https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
struct normal_random_variable
{
    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};

