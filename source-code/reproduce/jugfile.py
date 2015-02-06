import numpy as np
from jug import TaskGenerator

from nets.pixel import surf_grid, pixel_features
from nets.regions import hypersegmented_features
from nets.learn import RegionRegressor, Combined, evaluate
from utils import mkdir_p

import plotinfo

import ncpus
N_JOBS = ncpus.get_lsb_cpus()

def listfiles():
    from os import listdir
    for f in sorted(listdir('data/')):
        if not '_dna' in f:
            continue
        group = f[len('image_'):]
        group = int(group[:2])
        yield f,group

def load_image(f):
    import mahotas as mh
    f = 'data/'+f
    imread = TaskGenerator(mh.imread)
    return imread(f.replace('dna', 'protein')), \
            imread(f), \
            imread(f.replace('dna', 'rois'))

evaluate = TaskGenerator(evaluate)

@TaskGenerator
def fraction_for(r):
    return np.mean(r > 0)

@TaskGenerator
def run_loo(features_labels, fractions, origins, orig):
    train = (origins != orig)
    features = np.array([f for f,_ in features_labels], dtype=object)
    labels = np.array([ells for _,ells in features_labels], dtype=object)
    reg = RegionRegressor(random_state=1, n_jobs=N_JOBS)
    reg.fit(features[train], labels[train], fractions[train])
    rs = []
    finals = []
    vss = []
    for fs,frac in zip(features[~train], fractions[~train]):
        r = reg.predict(fs, return_partials=True, clip01=False)
        rraw = reg.predict(fs, fix_line=False, clip01=False)
        vss.append(r.partials)
        finals.append(r.final)
        rs.append(rraw)
    return np.array(finals), np.array(rs), vss


def train_test_combined(features_labels, fractions, train, test, mode):
    features_labels = np.array(features_labels, dtype=object)
    reg = Combined(random_state=1, mode=mode, n_jobs=N_JOBS)
    reg.fit(features_labels[train], fractions[train])
    rs = []
    for fs,frac in zip(features_labels[test], fractions[test]):
        fs = [f for f,_ in fs]
        r = reg.predict(fs, clip01=False)
        rs.append(r)
    return np.array(rs)

@TaskGenerator
def run_loo_combined(features_labels, fractions, origins, orig, mode):
    train = (origins != orig)
    return train_test_combined(features_labels, fractions, train, ~train, mode)

def kfold_by_origins(origins, nfolds, shuffle):
    import random
    random.seed(shuffle)
    valid = sorted(set(origins))
    random.shuffle(valid)
    for fold in range(nfolds):
        used = [v for i,v in enumerate(valid) if ((i % nfolds) != fold)]
        train = np.array([(orig in used) for orig in origins])
        yield np.where(train)[0], np.where(~train)[0]

@TaskGenerator
def build_standard(fractions, origins, nfolds, shuffle):
    res = []
    for _,test in kfold_by_origins(origins, nfolds, shuffle):
        res.append(fractions[test])
    return np.concatenate(res)

@TaskGenerator
def run_cv_combined(features_labels, fractions, origins, fold, mode, nfolds, shuffle):
    for i,(train,test) in enumerate(kfold_by_origins(origins, nfolds, shuffle)):
        if i == fold:
            return train_test_combined(features_labels, fractions, train, test, mode)

@TaskGenerator
def concat_clip(results):
    results = np.concatenate(results)
    results = results.ravel()
    results = np.clip(results, 0, 1)
    return results

@TaskGenerator
def plot_results(name, results, fractions, in_figures=False):
    from matplotlib import pyplot as plt
    import seaborn.apionly as sns
    fig,ax = plt.subplots(figsize=[plotinfo.TEXTWIDTH_1_2_IN, plotinfo.TEXTWIDTH_1_2_IN*.75])

    ax.plot([0,1], [0,1], c=plotinfo.color_blue, zorder=0)
    ax.scatter(results, fractions, s=16, lw=1, edgecolor='k', c=plotinfo.color_red, zorder=1)
    ax.set_ylim(-.04,1.04)
    ax.set_xlim(-.04,1.04)
    ax.grid(False)
    ax.set_xlabel('Human labeled')
    ax.set_ylabel('Automatic prediction')
    sns.despine(ax=ax, trim=True)
    fig.tight_layout()


    directory = ('figures/' if in_figures else 'outputs/')
    mkdir_p(directory)
    fig.savefig('{}/{}.eps'.format(directory, name))
    fig.savefig('{}/{}.pdf'.format(directory, name))
    plt.close(fig)


@TaskGenerator
def output_results(results):
    for measure in ('q2','r2'):
        names = results.keys()
        names.sort()
        with open('outputs/results_{}.txt'.format(measure), 'w') as output:
            for name in names:
                v = results[name]
                v = getattr(v, measure)
                output.write('{:48} {}\n'.format(name, 100.*v))
    with open('outputs/results_cv4.txt', 'w') as output:
        q2s = np.array([results['average.{}.origins'.format(i+1)].q2 for i in range(10)])
        output.write('4CV results. Minimum: {:.2%} Average: {:.2%} +/- {:.2%}\n'.format(q2s.min(), q2s.mean(), q2s.std()))

@TaskGenerator
def plot_error_correlations(vs, fractions):
    import pandas as pd
    from pandas.tools.plotting import scatter_matrix
    from matplotlib import pyplot as plt


    vs2 = pd.DataFrame(vs)
    real = fractions
    # vs2 = vs2.subtract(real,axis=0)
    vs2 = (vs2.T - real).T
    fig,ax = plt.subplots(figsize=[plotinfo.TEXTWIDTH_IN*.5, plotinfo.TEXTWIDTH_IN*.5])
    axes = scatter_matrix(vs2, ax=ax, color='k', marker='.', s=2.)
    for ax in axes.ravel():
        ax.grid(False)
        ax.set_axis_bgcolor((1,1,1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    for a in range(vs2.shape[1]):
        for b in range(a+1, vs2.shape[1]):
            da,db = vs2[[a,b]].T.values
            r = np.corrcoef(da,db)[0,1]
            ax = axes[b,a]
            ylabel = ax.get_ylabel()
            xlabel = ax.get_xlabel()
            ax.clear()
            ax.set_ylabel(ylabel, fontsize=5)
            ax.set_xlabel(xlabel, fontsize=5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(0,1)
            ax.set_xlim(0,1)
            ax.text(.55, .4,
                    '{:.2}'.format(r),
                    fontsize=9,
                    horizontalalignment='center',
                    verticalalignment='center')
            # ax = axes[a,b]
            # ax.plot([0,0],[-.25,+.25], 'k-', lw=3)
            # ax.plot([-.25,+.25], [0,0], 'k-', lw=3)
            # ax.scatter(db, da, color='r', s=1)
    axes[0,0].set_ylabel(axes[0,0].get_ylabel(), fontsize=5)
    axes[5,5].set_xlabel(axes[5,5].get_xlabel(), fontsize=5)

    fig.tight_layout()
    fig.savefig('figures/error_scatter.eps')
    fig.savefig('figures/error_scatter.pdf')
    fig.savefig('figures/error_scatter.png', dpi=1200)

@TaskGenerator
def plot_final_results(results):
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn.apionly as sns

    results = {k:(100*v.q2) for k,v in results.iteritems()}

    methods = [m for m,_ in FEATURES]
    methods.sort(key=lambda m: results[m+'.origins'])
    methods.append('average.loo')


    color0 = plotinfo.color_red
    color1 = plotinfo.color_blue

    fig,ax = plt.subplots(figsize=[plotinfo.TEXTWIDTH_1_2_IN, plotinfo.TEXTWIDTH_1_2_IN/1.6])


    ax.bar(np.arange(len(methods))+.4, [results[m + '.origins'] for m in methods], width=.4, color=color0, label='Corrected')
    ax.bar(np.arange(len(methods)-1), [results[m + '.origins.raw'] for m in methods[:-1]], width=.4, color=color1, label='Raw')
    ax.set_ylim(40, 100)
    ax.set_xlabel('Method')
    ax.set_ylabel(r'$Q^2 (\%)$')
    ax.set_xticks(np.arange(len(methods))+.4)
    ax.set_xticklabels(methods[:-1] + ['avg'], fontsize=7)
    ax.grid(False)

    for i, m in enumerate(methods[:-1]):
        v = results[m + '.origins.raw']
        plt.text(i +.2, v + 0.5, '{}'.format(int(np.round(v))), fontsize=8, horizontalalignment='center')
    for i, m in enumerate(methods):
        v = results[m + '.origins']
        ax.text(i + .6, v + 0.5, '{}'.format(int(np.round(v))), fontsize=8, horizontalalignment='center')
    sns.despine(ax=ax)

    ax.legend(loc='upper left', fontsize=7)

    fig.tight_layout()
    fig.savefig('figures/final-plot.pdf')
    fig.savefig('figures/final-plot.svg')
    fig.savefig('figures/final-plot.eps')

@TaskGenerator
def write_table(results):
    methods = [m for m,_ in FEATURES]
    methods.sort(key=lambda m:results[m+'.origins'].q2)

    with open('outputs/table_S1.tex', 'w') as output:
        def do1(name, corr, pat):
            origins = results[pat.format('origins')]
            no_origins = results[pat.format('no-origins')]
            print >>output, '{} & {} & {:.0%} & {:.0%} & & {:.0%} & {:.0%}\\\\'.format(name, corr, origins.q2, origins.r2, no_origins.q2, no_origins.r2).replace('%',r'\%')
        for m in methods:
            do1(m, 'corrected', m+'.{}')
            do1(m, 'raw'.format(m), m+'.{}.raw')
        do1('average', '', 'average.loo.{}')
        do1('matrix', '', 'matrix.loo.{}')


surf_grid = TaskGenerator(surf_grid)

FEATURES = [
    ('pixel', TaskGenerator(pixel_features)),
    ('regions', TaskGenerator(hypersegmented_features)),
    ('surf(1)', lambda p,d,r: surf_grid(p,d,r, 1)),
    ('surf(2)', lambda p,d,r: surf_grid(p,d,r, 2)),
    ('surf(4)', lambda p,d,r: surf_grid(p,d,r, 4)),
    ('surf(8)', lambda p,d,r: surf_grid(p,d,r, 8)),
]

results = {}
vscorr = {}
vs = {'origins': [], 'no-origins' : [] }

origins = []
fractions = []
images = []
for im,g in listfiles():
    protein,dna, rois = load_image(im)
    images.append((protein, dna, rois))
    fractions.append(fraction_for(rois))
    origins.append(g)
origins = np.array(origins)
no_origins = np.arange(len(origins))
fractions = TaskGenerator(np.array)(fractions)

for name,feats in FEATURES:
    features = []
    for protein, dna, rois in images:
        features.append(feats(protein, dna, rois))

    for oname,origs in [('origins',origins),
                        ('no-origins', no_origins)]:
        rs = []
        for orig in range(origs.max()+1):
            rs.append(run_loo(features, fractions, origs, orig))

        rname = '{}.{}'.format(name, oname)
        just_rvalues = concat_clip([r[0] for r in rs])
        #plot_results(rname, just_rvalues, fractions)
        results[rname] = evaluate(just_rvalues, fractions)
        vs[oname].append(just_rvalues)
        if oname =='origins':
            vscorr[name] = just_rvalues

        just_rvalues = concat_clip([r[1] for r in rs])
        rname += '.raw'
        in_figures = (name == 'surf(1)' and oname == 'origins')
        #plot_results(rname, just_rvalues, fractions, in_figures)
        results[rname] = evaluate(just_rvalues, fractions)


features = []
for protein, dna, rois in images:
    features.append([feats(protein,dna,rois) for name,feats in FEATURES])


for oname,origs in [('origins',origins),
                    ('no-origins', no_origins)]:
    for mode in ('matrix', 'average'):
        rs = []
        for orig in range(origs.max()+1):
            rs.append(run_loo_combined(features, fractions, origs, orig, mode))
        rs = concat_clip(rs)
        name = '{}.loo.{}'.format(mode, oname)
        results[name] = evaluate(rs, fractions)
        in_figures = (oname == 'origins' and mode == 'average')
        if in_figures:
            plot_results(name, rs, fractions, in_figures)

write_table(results)
output_results(results)
plot_error_correlations(vscorr, fractions)
plot_final_results(results)
