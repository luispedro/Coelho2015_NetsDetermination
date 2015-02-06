import numpy as np
from jug import TaskGenerator
from glob import glob
import mahotas as mh


@TaskGenerator
def compare1(f):
    rois = mh.imread(f.replace('rois2', 'rois'))
    rois2 = mh.imread(f)
    rarea = (rois != 0).ravel()
    rarea2 = (rois2 != 0).ravel()
    return (rarea.mean(),
            rarea2.mean(),
            np.corrcoef(rarea, rarea2)[0,1])

@TaskGenerator
def compare_pixels(f):
    rois = mh.imread(f.replace('rois2', 'rois'))
    rois2 = mh.imread(f)
    return np.corrcoef(rois.ravel() != 0, rois2.ravel() != 0)[0,1]

@TaskGenerator
def compare_arand(f):
    from sklearn import metrics
    from scipy.spatial import distance
    rois = mh.imread(f.replace('rois2', 'rois'))
    rois2 = mh.imread(f)
    rois = (rois.ravel() != 0)
    rois2 = (rois2.ravel() != 0)
    arand = metrics.adjusted_rand_score(rois, rois2)
    # Note that scipy returns the Jaccard Distance, which is 1 - Jaccard Index
    # sklearn does not really implement jaccard, but an interpretation where
    # jaccard is just a synonym for accuracy.

    jaccard = 1. - distance.jaccard(rois, rois2)
    mcc = metrics.matthews_corrcoef(rois, rois2)
    return arand, jaccard, mcc

@TaskGenerator
def plot(rs):
    from matplotlib import pyplot as plt
    import seaborn.apionly as sns
    import plotinfo
    rh2 = np.corrcoef(rs.T[0],rs.T[1])[0,1]**2
    real,alt,_ = rs.T
    q2 = 1.-np.dot(real-alt,real-alt)/np.dot(real-real.mean(),real-real.mean())
    w = plotinfo.TEXTWIDTH_1_2_IN
    fig,ax = plt.subplots(figsize=[w, w* 0.75])
    ax.text(.2, .84, r'$R^2: {}\%$'.format(int(np.round(rh2*100))), fontsize=14)
    ax.text(.2, .70, r'$Q^2: {}\%$'.format(int(np.round(q2*100))), fontsize=14)
    ax.set_xlabel(r'${}$Human labeler 1 (fraction assigned to NET)${}$')
    ax.set_ylabel(r'${}$Human labeler 2 (fraction assigned to NET)${}$')
    ax.set_ylim(-.04,1.04)
    ax.set_xlim(-.04,1.04)


    ax.scatter(rs.T[0], rs.T[1], s=16, lw=1, edgecolor='k', c=plotinfo.color_red, zorder=1)
    ax.plot([rs.T[:2].min(), rs.T[:2].max()], [rs.T[:2].min(), rs.T[:2].max()], c=plotinfo.color_blue, zorder=0)
    fig.tight_layout()
    sns.despine(ax=ax, offset=True, trim=True)

    fig.savefig('figures/human-comparison.pdf')
    fig.savefig('figures/human-comparison.eps')
    fig.savefig('figures/human-comparison.png', dpi=1200)

@TaskGenerator
def output_results(rs, prs, ars):
    real,alt,_ = rs.T
    r2 = np.corrcoef(rs.T[0],rs.T[1])[0,1]**2
    q2 = 1.-np.dot(real-alt,real-alt)/np.dot(real-real.mean(),real-real.mean())
    with open('outputs/humans.txt', 'w') as output:
        output.write('R2(overall): {}%\n'.format(np.round(r2*1000)/10.))
        output.write('Q2(overall): {}%\n'.format(np.round(q2*1000)/10.))
        output.write('R2(pixel): {}%\n'.format(np.round(np.mean(prs**2)*1000)/10.))
        output.write('R(pixel): {}%\n'.format(np.round(np.mean(prs)*1000)/10.))
        output.write('AR(pixel): {}%\n'.format(np.round(np.mean(ars.T[0])*1000)/10.))
        output.write('JI(pixel): {}%\n'.format(np.round(np.mean(ars.T[1])*1000)/10.))
        output.write('MCC(pixel): {}%\n'.format(np.round(np.mean(ars.T[2])*1000)/10.))

files = glob('data/*rois2.png')

to_array = TaskGenerator(np.array)
rs = to_array([compare1(f) for f in files])
prs = to_array([compare_pixels(f) for f in files])
ars = to_array([compare_arand(f) for f in files])
plot(rs)
output_results(rs, prs, ars)
