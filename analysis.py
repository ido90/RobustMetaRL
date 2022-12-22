import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import cross_entropy_sampler as cem
import general_utils as utils

TRAIN_FILE = 'res'
TEST_FILE = 'test_res'

def get_base_path(env_name, base='logs'):
    return f'{base}/logs_{env_name}'

def get_dir(base_path, env_short, method, seed):
    def get_month(s):
        if ':' in s:
            return int(s[s.find(':') + 1:s.find(':') + 3])
        return int(s[s.find('__')+ 5:s.find('__')+7])

    paths = [s for s in os.listdir(base_path)
             if s.startswith(f'{env_short}_{method}_{seed}__')]
    last_month = np.max([get_month(s) for s in paths])
    paths = [s for s in paths if get_month(s)==last_month]
    try:
        return sorted(paths)[-1]
    except:
        print(f'{base_path}/{env_short}_{method}_{seed}__')
        raise

def get_task_dim(dd):
    return len([col for col in dd.columns
                if col.startswith('task') and col != 'task_id'])

# aggregate over episodes within task
def agg_eps(d):
    d = d.copy()
    d['ret'] = d.ret.mean()
    d.drop('ep', axis=1, inplace=True)
    if 'info' in d.columns:
        d.drop('info', axis=1, inplace=True)
    return d.head(1)

# aggregate over tasks within seed
def agg_tasks(d, fun):
    d = d.copy()
    d['ret'] = fun(d.ret.values)
    for col in d.columns:
        if col.startswith('task') and col != 'task_id':
            d.drop(col, axis=1, inplace=True)
    if 'task_id' in d.columns:
        d.drop('task_id', axis=1, inplace=True)
    return d.head(1)

def get_cvar_fun(alpha):
    def cvar(x):
        return np.mean(sorted(x)[:int(np.ceil(alpha*len(x)))])
    return cvar

def load_train_data(env_name, env_short, methods, seeds, alpha,
                    align_progress=False, nm_map=None):
    base_path = get_base_path(env_name)
    cvar = get_cvar_fun(alpha)
    dd = pd.DataFrame()

    for method in methods:
        for seed in seeds:
            e = get_dir(base_path, env_short, method, seed)
            print(e)
            d = pd.read_pickle(f'{base_path}/{e}/{TRAIN_FILE}.pkl')
            d['method'] = method if nm_map is None else nm_map[method]
            d['seed'] = seed
            dd = pd.concat((dd, d))

    dd.reset_index(drop=True, inplace=True)
    task_dim = get_task_dim(dd)
    print('Task-space dimension:', task_dim)
    print('Validation points:', len(pd.unique(dd.iter)))

    if align_progress:
        # clip all trainings to the currently-shortest one
        n_steps = dd.groupby(['method', 'seed']).apply(
            lambda d: d.iter.max()).min()
        print(f'Removing iterations beyond {n_steps} '
              f'({100 - 100 * (dd.iter <= n_steps).mean():.0f}% of the data).')
        dd = dd[dd.iter <= n_steps]

    # aggregate episodes per task
    dda = dd.groupby(['method', 'seed', 'iter', 'task_id'], sort=False
                     ).apply(agg_eps)
    dda.reset_index(drop=True, inplace=True)

    # aggregate tasks per seed
    ddm = dda.groupby(['method', 'seed', 'iter'], sort=False
                      ).apply(lambda d: agg_tasks(d, np.mean))
    ddc = dda.groupby(['method', 'seed', 'iter'], sort=False
                      ).apply(lambda d: agg_tasks(d, cvar))
    ddm.reset_index(drop=True, inplace=True)
    ddc.reset_index(drop=True, inplace=True)

    # only first seed
    dd0 = dd[dd.seed == seeds[0]]
    dda0 = dda[dda.seed == seeds[0]]

    return dd, dda, ddm, ddc, dd0, dda0, task_dim

def load_test_data(env_name, env_short, methods, seeds, alpha,
                   model='best_cvar', fname=TEST_FILE, base_path='logs',
                   nm_map=None):
    base_path = get_base_path(env_name, base_path)
    cvar = get_cvar_fun(alpha)

    rr = pd.DataFrame()

    for method in methods:
        fnm = fname
        if model is not None:
            if isinstance(model, str):
                fnm = f'{fname}_{model}'
            elif callable(model):
                fnm = f'{fname}_{model(method)}'
            else:
                warnings.warn(f'Invalid model type: {type(model)}, {model}')

        for seed in seeds:
            e = get_dir(base_path, env_short, method, seed)
            try:
                d = pd.read_pickle(f'{base_path}/{e}/{fnm}.pkl')
            except:
                # backward compatibility: saved names were changed
                print(f'Cannot load file: {base_path}/{e}/{fnm}.pkl', end='')
                if fnm.endswith('best'):
                    ext = 'mean' if 'varibad' in e else 'cvar'
                    print(f'; trying to load best_{ext} instead')
                    d = pd.read_pickle(f'{base_path}/{e}/{fnm}_{ext}.pkl')
                else:
                    print()
                    raise
            d['method'] = method if nm_map is None else nm_map[method]
            d['seed'] = seed
            rr = pd.concat((rr, d))

    rr.reset_index(drop=True, inplace=True)
    print('Test tasks:', len(d[d.ep == 0]))

    # aggregate episodes per task
    rra = rr.groupby(['method', 'seed', 'task0'], sort=False).apply(agg_eps)
    rra.reset_index(drop=True, inplace=True)

    # aggregate tasks per seed
    rrm = rra.groupby(['method', 'seed'], sort=False).apply(
        lambda d: agg_tasks(d, np.mean))
    rrc = rra.groupby(['method', 'seed'], sort=False).apply(
        lambda d: agg_tasks(d, cvar))
    rrm.reset_index(drop=True, inplace=True)
    rrc.reset_index(drop=True, inplace=True)

    # only first seed
    rr0 = rr[rr.seed == seeds[0]]
    rra0 = rra[rra.seed == seeds[0]]

    return rr, rra, rrm, rrc, rr0, rra0

def show_task_distribution(dda0, rra0=None, tasks=None):
    task_dim = get_task_dim(dda0)
    axs = utils.Axes(2 * task_dim, 4, fontsize=15)
    a = 0

    if tasks is None:
        tasks = [f'task_{i:d}' for i in range(task_dim)]

    for i in range(task_dim):

        utils.qplot(dda0, f'task{i:d}', f'task', 'method', axs[a])
        axs.labs(a, ylab=tasks[i], title='Validation tasks, first seed')
        a += 1

        if rra0 is not None:
            utils.qplot(rra0, f'task{i:d}', f'task', 'method', axs[a])
            axs.labs(a, ylab=tasks[i], title='Test tasks, first seed')
        a += 1

    plt.tight_layout()
    return axs

def show_validation_vs_tasks(dda, tasks=None, xbins=11, fbins=7):
    print('Validation returns vs. task - over all seeds aggregated:')
    task_dim = get_task_dim(dda)
    if tasks is None:
        tasks = [f'task_{i:d}' for i in range(task_dim)]

    for i in range(task_dim):
        axs = utils.compare_quantiles(dda, f'task{i:d}', 'ret', 'method',
                                      'iter', xbins=xbins, fbins=fbins)
        for j in range(len(axs)):
            axs.labs(j, tasks[i], 'return')
    return axs

def show_test_vs_tasks(rra, rra0=None, title=None, tasks=None, xbins=11, **kwargs):
    print('Test returns vs. task - over all seeds aggregated:')
    task_dim = get_task_dim(rra)
    axs = utils.Axes(task_dim, min(3, task_dim), fontsize=15)
    a = 0

    if tasks is None:
        tasks = [f'task_{i:d}' for i in range(task_dim)]

    # n_tasks = len(pd.unique(rra0[rra0.method==rra0.method.values[0]].task0))
    # tit0 = f'First seed, {n_tasks} test tasks'
    tit1 = f'Test tasks, {len(pd.unique(rra.seed))} seeds'
    if title:
        # tit0 = f'{title}\n({tit0})'
        tit1 = f'{title}\n({tit1})'

    for i in range(task_dim):
        # utils.compare_quantiles(rra0, f'task{i:d}', 'ret', 'method',
        #                         xbins=xbins, lab_rotation=40, axs=axs, a0=a)
        # axs.labs(a, tasks[i], 'return', tit0)
        # a += 1

        utils.compare_quantiles(rra, f'task{i:d}', 'ret', 'method',
                                xbins=xbins, lab_rotation=40, axs=axs, a0=a, **kwargs)
        axs.labs(a, tasks[i], 'return', tit1)
        a += 1

    plt.tight_layout()
    return axs

def cem_analysis(env_name, task_dim, transformation=None, ylim=None,
                 smooth=20, seed=0, title=None, tasks=None, soft=False):
    if tasks is None:
        tasks = [f'task_{i:d}' for i in range(task_dim)]

    ce = cem.get_cem_sampler(env_name, seed, cem_type=2 if soft else 1)
    try:
        ce.load(f'logs/models/{ce.title}')
    except:
        # tmp exception for old naming format
        ce.title = ce.title[:-2]
        ce.load(f'logs/models/{ce.title}')

    c1, c2 = ce.get_data()
    if transformation is not None:
        trns = transformation
        for i in range(task_dim):
            if isinstance(transformation, (tuple, list)):
                trns = transformation[i]
            c1.iloc[:, -task_dim + i] = trns(
                c1.iloc[:, -task_dim + i])

    axs = utils.Axes(2, 2, fontsize=15)

    for i in range(task_dim):
        tau = c1.iloc[:, -task_dim + i].values
        axs[0].plot(c1.batch, tau, label=tasks[i])
        axs[1].plot(c1.batch, utils.smooth(tau, min(smooth, len(c1))),
                    label=tasks[i])

    if ylim is not None:
        axs[0].set_ylim(ylim)
        axs[1].set_ylim(ylim)

    if task_dim > 1:
        axs.labs(0, 'CE iteration', 'E[task]', title)
        axs.labs(1, 'CE iteration', 'E[task] (smoothed)', title)
        axs[0].legend(fontsize=13)
        axs[1].legend(fontsize=13)
    else:
        axs.labs(0, 'CE iteration', f'E[{tasks[0]}]', title)
        axs.labs(1, 'CE iteration', f'E[{tasks[0]}] (smoothed)', title)

    plt.tight_layout()

    axs2 = ce.show_summary(ylab='return')
    if title:
        axs2[0].set_title(title, fontsize=15)

    return ce, c1, c2, axs, axs2

def show_validation_results(dda0, alpha, ci=None):
    axs = utils.Axes(2, 2, (7, 4), fontsize=15)
    a = 0

    sns.lineplot(data=dda0, x='iter', hue='method', y='ret',
                 estimator='mean', ax=axs[a], ci=ci)
    axs.labs(a, 'iteration', 'validation mean', 'first seed')
    a += 1

    # cvar over tasks
    cvar = get_cvar_fun(alpha)
    sns.lineplot(data=dda0, x='iter', hue='method', y='ret',
                 estimator=cvar, ax=axs[a], ci=ci)
    axs.labs(a, 'iteration', f'validation CVaR$_{{{alpha}}}$',
             'first seed')
    a += 1

    plt.tight_layout()
    return axs

def show_validation_results_over_seeds(ddm, ddc, alpha, title=None, ci='sd', axsize=(7, 4)):
    axs = utils.Axes(2, 2, axsize, fontsize=15)
    a = 0

    tit = f'{len(pd.unique(ddm.seed))} seeds'
    if title:
        tit = f'{title} ({tit})'

    sns.lineplot(data=ddm, x='iter', hue='method', y='ret', ax=axs[a], ci=ci)
    axs.labs(a, 'iteration', 'validation mean', tit)
    a += 1

    # cvar over tasks, mean over seeds
    sns.lineplot(data=ddc, x='iter', hue='method', y='ret', ax=axs[a], ci=ci)
    axs.labs(a, 'iteration', f'validation CVaR$_{{{alpha}}}$', tit)
    a += 1

    plt.tight_layout()
    return axs

def summarize_test(rra0, rr0, alpha):
    cvar = get_cvar_fun(alpha)
    n_episodes = rr0.ep.values[-1] + 1
    axs = utils.Axes(5+n_episodes, 3, fontsize=15)
    a = 0

    sns.barplot(data=rra0, x='method', y='ret', ci=95, capsize=0.1,
                ax=axs[a])
    axs.labs(a, 'method', 'mean return', 'first seed')
    a += 1

    sns.barplot(data=rra0, x='method', y='ret', ci=95, capsize=0.1,
                estimator=cvar, ax=axs[a])
    axs.labs(a, 'method', f'$CVaR_{{{alpha}}}$ return', 'first seed')
    a += 1

    utils.qplot(rra0, 'ret', 'task', 'method', q=np.linspace(0,1,100),
                ax=axs[a])
    axs[a].set_ylabel('return')
    a += 1

    sns.lineplot(data=rr0, x='ep', hue='method', y='ret', ax=axs[a])
    axs.labs(a, 'episode', 'mean return', 'first seed')
    a += 1

    sns.lineplot(data=rr0, x='ep', hue='method', y='ret', estimator=cvar,
                 ax=axs[a])
    axs.labs(a, 'episode', f'$CVaR_{{{alpha}}}$ return', 'first seed')
    a += 1

    for ep in range(n_episodes):
        utils.qplot(rr0[rr0.ep==ep], 'ret', 'task', 'method',
                    q=np.linspace(0,1,100), ax=axs[a])
        axs.labs(a, title=f'episode {ep} (first seed)', ylab='return')
        a += 1

    plt.tight_layout()
    return axs

def summarize_test_over_seeds(rrm, rrc, alpha, title=None, barplot=False):
    n_seeds = len(pd.unique(rrm.seed))

    axs = utils.Axes(4, 4, fontsize=15)
    a = 0

    meanprops = dict(marker='o', markerfacecolor='w',
                     markeredgecolor='brown', markeredgewidth=2, markersize=10)
    dotprops = dict(color='y', edgecolor='k', size=8, linewidth=1)

    if barplot:
        sns.barplot(data=rrm, x='method', y='ret', ci=95, capsize=0.1, ax=axs[a])
    else:
        sns.boxplot(data=rrm, x='method', y='ret', ax=axs[a], showmeans=True,
                    meanprops=meanprops)
        sns.stripplot(data=rrm, x='method', y='ret', ax=axs[a], **dotprops)
    axs.labs(a, 'method', 'mean return', title)
    a += 1

    if barplot:
        sns.barplot(data=rrc, x='method', y='ret', ci=95, capsize=0.1, ax=axs[a])
    else:
        sns.boxplot(data=rrc, x='method', y='ret', ax=axs[a], showmeans=True,
                    meanprops=meanprops)
        sns.stripplot(data=rrc, x='method', y='ret', ax=axs[a], **dotprops)
    axs.labs(a, 'method', f'$CVaR_{{{alpha}}}$ return', title)
    a += 1

    utils.qplot(rrm, 'ret', 'seed', 'method', q=np.linspace(0,1,n_seeds),
                dots=True, ax=axs[a])
    axs.labs(a, f'quantile over {n_seeds} seeds [%]', 'mean return', title)
    a += 1

    utils.qplot(rrc, 'ret', 'seed', 'method', q=np.linspace(0,1,n_seeds),
                dots=True, ax=axs[a])
    axs.labs(a, f'quantile over {n_seeds} seeds [%]', f'$CVaR_{{{alpha}}}$ return',
             title)
    a += 1

    plt.tight_layout()
    return axs
