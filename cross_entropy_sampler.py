
import numpy as np
from scipy import stats
import cross_entropy_method as cem


def get_cem_sampler(env_name, seed, oracle=False, alpha=0.05, cem_type=1):
    sfx = f'{seed:d}'
    if cem_type == 2:
        sfx = f'soft_{sfx}'

    if env_name == 'HalfCheetahVel-v0':
        if oracle:
            return CEM_HCV_Oracle(
                0.5, ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_vel_oracle_{sfx}')
        else:
            return CEM_Beta(
                0.5, ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_vel_{sfx}')
    elif env_name == 'HalfCheetahMass-v0':
        if oracle:
            return CEM_HCM_Oracle(
                0.5, ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_mass_oracle_{sfx}')
        else:
            return LogBeta1D(
                0.5, ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_mass_{sfx}')
    elif env_name == 'HalfCheetahBody-v0':
        if oracle:
            raise NotImplementedError(
                f'No oracle-CEM implemented for HalfCheetahBody-v0.')
        else:
            return LogBeta(
                0.5*np.ones(3), ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hc_body_{sfx}',
                titles=('mass', 'damping', 'head_size'))
    elif env_name == 'HumanoidVel-v0':
        if oracle:
            raise NotImplementedError(
                f'No oracle-CEM implemented for HumanoidVel-v0.')
        else:
            return CEM_Beta(
                0.5, vmax=2.5, ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hum_vel_{sfx}')
    elif env_name == 'HumanoidMass-v0':
        if oracle:
            raise NotImplementedError(
                f'No oracle-CEM implemented for HumanoidMass-v0.')
        else:
            return LogBeta1D(
                0.5, ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hum_mass_{sfx}')
                # titles=('butt_mass', 'foot_mass', 'hand_mass'))
    elif env_name == 'HumanoidBody-v0':
        if oracle:
            raise NotImplementedError(
                f'No oracle-CEM implemented for HumanoidBody-v0.')
        else:
            return LogBeta(
                0.5 * np.ones(3), ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'hum_body_{sfx}',
                titles=('mass', 'damping', 'head_size'))
                # titles=('strength', 'mass'))
                # titles=('butt_mass', 'foot_mass', 'hand_mass', 'butt_len', 'foot_len', 'hand_len'))
                # (butt = either butt or pelvis)
    elif env_name == 'AntGoal-v0':
        if oracle:
            raise NotImplementedError(
                f'No oracle-CEM implemented for AntGoal-v0.')
        else:
            return CemCircle(
                0.5*np.ones(2), ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'ant_goal_{sfx}')
    elif env_name == 'AntMass-v0':
        if oracle:
            raise NotImplementedError(
                f'No oracle-CEM implemented for AntMass-v0.')
        else:
            return LogBeta1D(
                0.5, ref_alpha=alpha, batch_size=8*16,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'ant_mass_{sfx}')
    elif env_name == 'KhazadDum-v0':
        if oracle:
            raise NotImplementedError(
                f'No oracle-CEM implemented for KhazadDum-v0.')
        else:
            return CemExp(
                0.1, ref_alpha=alpha, batch_size=8*16, min_batch_update=0.05,
                n_orig_per_batch=0.2, soft_update=0.5, title=f'kd_{sfx}')
    raise ValueError(env_name)


class CEM_Beta(cem.CEM):
    '''CEM for 1D Beta distribution.'''

    def __init__(self, *args, vmax=7.0, eps=0.02, **kwargs):
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

    def __init__(self, *args, log_range=(-1, 1), eps=0.02, titles=None, **kwargs):
        super(LogBeta, self).__init__(*args, **kwargs)
        if titles is not None:
            self.default_dist_titles = [f'{t}_mean' for t in titles]
            self.default_samp_titles = [t for t in titles]
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

class CemExp(cem.CEM):
    def __init__(self, *args, **kwargs):
        super(CemExp, self).__init__(*args, **kwargs)
        self.default_dist_titles = 'exp_mean'
        self.default_samp_titles = 'sample'

    def do_sample(self, phi):
        return np.random.exponential(phi)

    def pdf(self, x, phi):
        return stats.expon.pdf(x, 0, phi)

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        return np.mean(w*s)/np.mean(w)

class CemCircle(cem.CEM):
    def __init__(self, *args, eps=0.0, **kwargs):
        super(CemCircle, self).__init__(*args, **kwargs)
        self.eps = eps
        self.default_dist_titles = ('mean_r', 'mean_theta')
        self.default_samp_titles = ('r', 'theta')

    def do_sample(self, phi):
        r0, theta0 = np.random.beta(2*phi, 2-2*phi)
        theta = 2*np.pi*theta0
        r = 5*np.sqrt(r0)
        return np.array((r*np.cos(theta), r*np.sin(theta)))

    def xy2rt(self, x, y):
        return (x**2+y**2)/(5**2), \
               (np.arctan2(y, x)+np.pi) / (2*np.pi)

    def pdf(self, x, phi):
        x = self.xy2rt(*x)
        return stats.beta.pdf(np.clip(x, self.eps, 1-self.eps),
                              2*phi, 2-2*phi).prod()

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.stack([self.xy2rt(*ss) for ss in samples])

        wmean = np.mean(w)
        w = np.repeat(w[:, np.newaxis], s.shape[1], axis=1)
        return np.clip(np.mean(w*s, axis=0)/wmean, self.eps, 1-self.eps)
