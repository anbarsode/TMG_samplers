import numpy as np
from scipy.special import erf, erfinv
import os

Eigen_path = '/home/ankur/My_Projects/eigen-3.4.0'
TMG_path = '/home/ankur/astrorel/MACHO_GO_Lensing_project/TMG_sampling'

class TMG(object):
    def __init__(self):
        self.mu = None
        self.sig = None
        self.cov = None
        self.minval = None
        self.maxval = None
        self.initial = None
        self.fname_current = None
        self.samples_current = None
        self.acceptance = None
    
    
    ################################# Analytical functions #################################
    
    
    def Gaussian1DPDF(self, x, mu, sig):
        G = np.exp(-(x - mu)**2.0 / 2.0 / sig**2.0)
        G /= np.sqrt(2.0 * np.pi * sig)
        return G

    def Gaussian1DCDF(self, x, mu, sig):
        return 0.5 * (1.0 + erf((x - mu) / sig / 2.0**0.5))

    def Gaussian1DCDF_inv(self, y, mu, sig):
        x = 2.0 * y - 1.0
        x = erfinv(x)
        x *= sig * 2.0**0.5
        x += mu
        return x

    def TruncatedGaussian1DPDF(self, x, mu, sig, minval, maxval):
        TG = self.Gaussian1DPDF(x, mu, sig)
        TG /= sig**0.5 * (self.Gaussian1DCDF(maxval, mu, sig) - self.Gaussian1DCDF(minval, mu, sig))
        return TG

    
    ################################# Utility functions #################################
    
    
    def save_samples(self, outfile, asASCII=True):
        assert self.samples_current is not None
        if self.fname_current is None and asASCII==True: np.savetxt(outfile + '.txt', self.samples_current)
        elif self.fname_current is None: np.save(outfile + '.npy', self.samples_current)
        else: os.rename(self.fname_current, outfile)
    
    def precalculate_quantities(self, mu, cov):
        Ndim = cov.shape[0]
        M1 = np.empty((Ndim))
        M2 = np.empty((Ndim, Ndim-1, 1))

        C11 = np.empty((Ndim))
        C12 = np.empty((Ndim, 1, Ndim-1))
        C22_inv = np.empty((Ndim, Ndim-1, Ndim-1))
        C_eff = np.empty((Ndim))

        for ind in range(Ndim):
            nonind = np.delete(np.arange(Ndim),ind,0)
            M1[ind] = mu[ind,0].copy()
            M2[ind,:,:] = np.delete(mu,ind,0)
            C11[ind] = cov[ind,ind].copy()
            C12[ind,:,:] = np.delete(np.delete(cov,nonind,0),ind,1)
            C22_inv[ind,:,:] = np.linalg.inv(np.delete(np.delete(cov,ind,0),ind,1))
            C_eff[ind] = np.sqrt(C11[ind] - C12[ind,:,:] @ C22_inv[ind,:,:] @ C12[ind,:,:].T)
        return M1, M2, C11, C12, C22_inv, C_eff

    def mu_effective(self, ind, x, M1, M2, C11, C12, C22_inv):
        # ind: int, index corresponding to random variable whose conditional PDF is to be found
        # x: arr of shape (Ndim,1), all values other than x[ind,0] are conditions
        # mu: arr of shape (Ndim,1)
        # cov: arr of shape (Ndim,Ndim)

        x_cond = np.delete(x,ind,0)
        mu_eff = M1[ind] + C12[ind,:,:] @ C22_inv[ind,:,:] @ (x_cond - M2[ind,:,:])
        return mu_eff
    
    
    ################################# Numpy based codes #################################
    
    
    def T1DG_rejection_numpy(self, N, mu, sig, minval, maxval, maxrejini = 1000, showprogress=False):
        x = np.empty((N))
        rej = 0
        i = 0
        while i < N:
            num = np.random.normal(mu, sig, 1)
            if num >= minval and num <= maxval:
                x[i] = num
                i += 1
            else:
                rej += 1
                if showprogress: print('Sampled: %d, acceptance: %.2f %%' % (i+1, (i+1) / (rej + (i+1)) * 100.0), end='\r')
                if rej == maxrejini and len(x) == 0:
                    x = None
                    break
        if showprogress: print('\n')
        
        self.fname_current = None
        self.acceptance = N / (N + rej)
        self.samples_current = x
        return self.samples_current

    def T1DG_KD_numpy(self, N, mu, sig, minval, maxval):
        x = np.random.normal(mu, sig, N)
        y = self.Gaussian1DCDF(x, mu, sig) * \
            (self.Gaussian1DCDF(np.array([maxval]), mu, sig) - self.Gaussian1DCDF(np.array([minval]), mu, sig)) + \
            self.Gaussian1DCDF(np.array([minval]), mu, sig)
        
        self.fname_current = None
        self.samples_current = self.Gaussian1DCDF_inv(y, mu, sig)
        return self.samples_current
    
    def TMG_rejection_numpy(self, N, mu, cov, minvals, maxvals, maxrejini = 1000, showprogress=False):
        x = np.empty((N, mu.shape[0]))
        rej = 0
        i = 0
        while i < N:
            sample = np.random.multivariate_normal(mu, cov)
            if np.prod((sample >= minvals) & (sample <= maxvals)) != 0:
                x[i,:] = sample
                i += 1
            else:
                rej += 1
                if showprogress: print('Sampled: %d, acceptance: %.2f %%' % (i+1, (i+1) / (rej + (i+1)) * 100.0), end='\r')
                if rej == maxrejini and len(x) == 0:
                    x = None
                    break
        if showprogress: print('\n')
        
        self.fname_current = None
        self.acceptance = N / (N + rej)
        self.samples_current = x
        return self.samples_current
    
    def TMG_KDGibbs_numpy(self, N, mu, cov, minvals, maxvals, x0=None, burnin = 100, hop = 1):
        if x0 is None:
            if np.prod((mu >= minvals) & (mu <= maxvals)) != 0: x0 = mu.copy()
            else: x0 = np.array([np.random.uniform(minvals[i],maxvals[i]) for i in range(mu.shape[0])])
        M1, M2, C11, C12, C22_inv, Cov_eff = self.precalculate_quantities(np.reshape(mu,(mu.shape[0],1)), cov)
        x_now = x0.copy()
        x = []
        iters = 0
        while len(x) < N:
            for ind in range(cov.shape[0]):
                mu_eff = self.mu_effective(ind, np.reshape(x_now, (x_now.shape[0],1)), M1, M2, C11, C12, C22_inv)
                C_eff = Cov_eff[ind]
                x_now[ind] = self.T1DG_KD_numpy(None, mu_eff, C_eff, minvals[ind], maxvals[ind])
            iters += 1
            if iters > burnin and (iters - burnin) % hop == 0: x.append(x_now.copy())
        
        self.fname_current = None
        self.samples_current = np.array(x)
        return self.samples_current
    
    
    ################################# C++ based codes #################################
    
    
    def T1DG_rejection(self, N, mu, sig, minval, maxval, recompile=False, UsePythonIfNan = True):
        if not os.path.exists('%s/T1DG_rejection' % TMG_path) or recompile:
            os.system('g++ -I %s %s/T1DG_rejection.cpp -o %s/T1DG_rejection' % (Eigen_path, TMG_path, TMG_path))
        
        ip_str = '%e %e %e %e %d' % (mu, sig, minval, maxval, N)
        os.system('%s/T1DG_rejection %s > %s/TMGacc.temp' % (TMG_path, ip_str, TMG_path))
        self.acceptance = float(open('%s/TMGacc.temp' % TMG_path,'r').read())
        self.fname_current = '%s/T1DG_rejection.samples' % TMG_path
        self.samples_current = np.loadtxt(self.fname_current)
        
        if np.isnan(self.samples_current).sum() != 0 and UsePythonIfNan:
            print('C++ returned NaNs. Defaulting to numpy version')
            self.T1DG_rejection_numpy(N, mu, sig, minval, maxval)
        return self.samples_current
    
    def T1DG_KD(self, N, mu, sig, minval, maxval, recompile=False, UsePythonIfNan = True):
        if not os.path.exists('%s/T1DG_KD' % TMG_path) or recompile:
            os.system('g++ -I %s %s/T1DG_KD.cpp -o %s/T1DG_KD' % (Eigen_path, TMG_path, TMG_path))
        
        ip_str = '%e %e %e %e %d' % (mu, sig, minval, maxval, N)
        os.system('%s/T1DG_KD %s' % (TMG_path, ip_str))
        self.fname_current = '%s/T1DG_KD.samples' % TMG_path
        self.samples_current = np.loadtxt(self.fname_current)
        
        if np.isnan(self.samples_current).sum() != 0 and UsePythonIfNan:
            print('C++ returned NaNs. Defaulting to numpy version')
            self.T1DG_KD_numpy(N, mu, sig, minval, maxval)
        return self.samples_current
    
    def TMG_rejection(self, N, mu, cov, minvals, maxvals, recompile=False, UsePythonIfNan = True):
        if not os.path.exists('%s/TMG_rejection' % TMG_path) or recompile:
            os.system('g++ -I %s %s/TMG_rejection.cpp -o %s/TMG_rejection' % (Eigen_path, TMG_path, TMG_path))

        Ndim = mu.shape[0]
        ip_str = '%d ' % Ndim
        for i in range(Ndim): ip_str += '%e ' % mu[i]
        for i in range(Ndim):
            for j in range(Ndim):
                ip_str += '%e ' % cov[i,j]
        for i in range(Ndim): ip_str += '%e ' % minvals[i]
        for i in range(Ndim): ip_str += '%e ' % maxvals[i]
        ip_str += '%d' % N

        os.system('%s/TMG_rejection %s > %s/TMGacc.temp' % (TMG_path, ip_str, TMG_path))
        self.acceptance = float(open('%s/TMGacc.temp' % TMG_path,'r').read())
        self.fname_current = '%s/TMG_rejection.samples' % TMG_path
        self.samples_current = np.loadtxt(self.fname_current)
        
        if np.isnan(self.samples_current).sum() != 0 and UsePythonIfNan:
            print('C++ returned NaNs. Defaulting to numpy version')
            self.TMG_rejection_numpy(N, mu, cov, minvals, maxvals)
        return self.samples_current
    
    def TMG_KDGibbs(self, N, mu, cov, minvals, maxvals, x_init=None, burnin = 100, hop = 1, recompile=False, UsePythonIfNan = True):
        if not os.path.exists('%s/TMG_KDGibbs' % TMG_path) or recompile:
            os.system('g++ -I %s %s/TMG_KDGibbs.cpp -o %s/TMG_KDGibbs' % (Eigen_path, TMG_path, TMG_path))
        
        if x_init is None:
            if np.prod((mu >= minvals) & (mu <= maxvals)) != 0: x_init = mu.copy()
            else: x_init = np.array([rng.uniform(minvals[i],maxvals[i]) for i in range(mu.shape[0])])
        
        Ndim = mu.shape[0]
        ip_str = '%d ' % Ndim
        for i in range(Ndim): ip_str += '%e ' % mu[i]
        for i in range(Ndim):
            for j in range(Ndim):
                ip_str += '%e ' % cov[i,j]
        for i in range(Ndim): ip_str += '%e ' % minvals[i]
        for i in range(Ndim): ip_str += '%e ' % maxvals[i]
        for i in range(Ndim): ip_str += '%e ' % x_init[i]
        ip_str += '%d ' % N
        ip_str += '%d ' % burnin
        ip_str += '%d' % hop
        
        os.system('%s/TMG_KDGibbs %s' % (TMG_path, ip_str))
        self.fname_current = '%s/TMG_KDGibbs.samples' % TMG_path
        self.samples_current = np.loadtxt(self.fname_current)
        
        if np.isnan(self.samples_current).sum() != 0 and UsePythonIfNan:
            print('C++ returned NaNs. Defaulting to numpy version')
            self.TMG_KDGibbs_numpy(N, mu, cov, minvals, maxvals, x_init=x_init, burnin = burnin, hop = hop)
        return self.samples_current