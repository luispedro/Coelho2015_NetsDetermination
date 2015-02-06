import numpy as np
import mahotas.thresholding
from glob import glob
from os import path
import mahotas as mh
from jug import TaskGenerator


def load_image(f):
    import mahotas as mh
    return mh.imread(f.replace('dna', 'protein')), \
            mh.imread(f), \
            mh.imread(f.replace('dna', 'rois'))


@TaskGenerator
def bernsen(radius, contrast, take_dna):
    res = []
    for f in glob('data/*_dna.png'):
        protein,dna,rois = load_image(f)
        interest = (rois > 0)
        protein = mahotas.gaussian_filter(protein, 8).astype(np.uint8)
        dna = mahotas.gaussian_filter(dna, 8).astype(np.uint8)
        dthresh = mahotas.thresholding.bernsen(dna, radius, contrast, mahotas.thresholding.otsu(dna))
        pthresh = mahotas.thresholding.bernsen(protein, radius, contrast, mahotas.thresholding.otsu(protein))
        if take_dna:
            prediction = pthresh & ~dthresh
        else:
            prediction = pthresh
        res.append([prediction.mean(), interest.mean()])
    return np.array(res)


@TaskGenerator
def output_result(results):
    cs = []
    for v in results.values():
        cs.append(np.corrcoef(v.T[0], v.T[1])[0,1])
    with open('outputs/bernsen.txt', 'w') as output:
        print >>output, 'Max R: {}'.format(np.max(cs))
        print >>output, 'Min R: {}'.format(np.min(cs))

results = {}
for r in (4,8,16,32):
    for c in (8,16,24,32,40,48,64):
        results[r,c,True] = bernsen(r,c, True)
        results[r,c,False] = bernsen(r,c, False)
output_result(results)
