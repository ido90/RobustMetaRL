
import numpy as np
from scipy import stats
import cross_entropy_method as cem


def get_cem_sampler(env_name, seed, oracle=False):
    if env_name == 'HalfCheetahVel-v0':
        if oracle:
            return CEM_HCV_Oracle(
                0.5, ref_alpha=0.05, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_vel_oracle_{seed:d}')
        else:
            return CEM_Beta(
                0.5, ref_alpha=0.05, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_vel_{seed:d}')
    elif env_name == 'HalfCheetahMass-v0':
        if oracle:
            return CEM_HCM_Oracle(
                0.5, ref_alpha=0.05, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_mass_oracle_{seed:d}')
        else:
            return LogBeta1D(
                0.5, ref_alpha=0.05, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_mass_{seed:d}')
    elif env_name == 'HalfCheetahBody-v0':
        if oracle:
            raise NotImplementedError(
                f'No oracle-CEM implemented for HalfCheetahBody-v0.')
        else:
            return LogBeta(
                0.5*np.ones(3), ref_alpha=0.05, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_body_{seed:d}')
    raise ValueError(env_name)


class CEM_Beta(cem.CEM):
    '''CEM for 1D Beta distribution.'''

    def __init__(self, *args, vmax=7, eps=0.02, **kwargs):
        super(CEM_Beta, self).__init__(*args, **kwargs)
        self.default_dist_titles = 'beta_mean'
        self.default_samp_titles = 'sample'
        self.vmax = vmax
        self.eps = eps

    def do_sample(self, phi):
        return self.vmax * np.random.beta(2*phi, 2-2*phi)

    def pdf(self, x, phi):
        return stats.beta.pdf(np.clip(x/self.vmax,self.eps,1-self.eps),
                              2*phi, 2-2*phi)

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples) / self.vmax
        return np.clip(np.mean(w*s)/np.mean(w), self.eps, 1-self.eps)

class CEM_HCV_Oracle(cem.CEM):
    def __init__(self, *args, vmax=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_dist_titles = 'unif_mean'
        self.default_samp_titles = 'sample'
        self.vmax = vmax
        self.phi_goal = 1 - 0.5 * self.ref_alpha
    def do_sample(self, phi):
        a = np.clip(1-2*(1-phi), 0, 1)
        b = 1
        return self.vmax * np.random.uniform(a, b)
    def pdf(self, x, phi):
        a = np.clip(1-2*(1-phi), 0, 1)
        b = 1
        return 1 / (b-a)
    def update_sample_distribution(self, samples, weights):
        return self.phi_goal


class LogBeta1D(cem.CEM):
    '''x ~ 2**Beta'''

    def __init__(self, *args, log_range=(-1, 1), eps=0.02, **kwargs):
        super(LogBeta1D, self).__init__(*args, **kwargs)
        self.default_dist_titles = 'beta_mean'
        self.default_samp_titles = 'sample'
        self.log_range = log_range
        self.eps = eps

    def do_sample(self, phi):
        x = np.random.beta(2*phi, 2-2*phi)
        x = self.log_range[0] + (self.log_range[1]-self.log_range[0]) * x
        return np.power(2, x)

    def pdf(self, x, phi):
        x = np.log2(x)
        x = (x - self.log_range[0]) / (self.log_range[1]-self.log_range[0])
        return stats.beta.pdf(np.clip(x, self.eps, 1-self.eps),
                              2*phi, 2-2*phi)

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        s = (np.log2(s) - self.log_range[0]) / (self.log_range[1]-self.log_range[0])
        return np.clip(np.mean(w*s)/np.mean(w), self.eps, 1-self.eps)

class CEM_HCM_Oracle(cem.CEM):
    def __init__(self, *args, log_range=(-1, 1), **kwargs):
        super().__init__(*args, **kwargs)
        self.default_dist_titles = 'unif_mean'
        self.default_samp_titles = 'sample'
        self.log_range = log_range
        self.phi_goal = 1 - 0.5 * self.ref_alpha
    def do_sample(self, phi):
        a = np.clip(1-2*(1-phi), 0, 1)
        b = 1
        x = np.random.uniform(a, b)
        x = self.log_range[0] + (self.log_range[1] - self.log_range[0]) * x
        return np.power(2, x)
    def pdf(self, x, phi):
        a = np.clip(1-2*(1-phi), 0, 1)
        b = 1
        return 1 / (b-a)
    def update_sample_distribution(self, samples, weights):
        return self.phi_goal


class LogBeta(cem.CEM):
    '''x ~ 2**Beta, multi-dimensional'''

    def __init__(self, *args, log_range=(-1, 1), eps=0.02, **kwargs):
        super(LogBeta, self).__init__(*args, **kwargs)
        self.default_dist_titles = ['mass_mean', 'inertia_mean', 'head_size_mean']
        self.default_samp_titles = ['mass', 'inertia', 'head_size']
        self.log_range = log_range
        self.eps = eps

    def do_sample(self, phi):
        x = np.random.beta(2*phi, 2-2*phi)
        x = self.log_range[0] + (self.log_range[1]-self.log_range[0]) * x
        return np.power(2, x)

    def pdf(self, x, phi):
        x = np.log2(x)
        x = (x - self.log_range[0]) / (self.log_range[1]-self.log_range[0])
        return stats.beta.pdf(np.clip(x, self.eps, 1-self.eps),
                              2*phi, 2-2*phi).prod()

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        wmean = np.mean(w)
        w = np.repeat(w[:, np.newaxis], s.shape[1], axis=1)
        s = (np.log2(s) - self.log_range[0]) / (self.log_range[1]-self.log_range[0])
        return np.clip(np.mean(w*s, axis=0)/wmean, self.eps, 1-self.eps)
