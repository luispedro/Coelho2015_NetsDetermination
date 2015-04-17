import cPickle as pickle
import gzip
import numpy as np

from nets.pixel import surf_grid, pixel_features
from nets.regions import hypersegmented_features
from nets.learn import Combined
from multiprocessing import cpu_count

N_JOBS = cpu_count()
DATADIR = '../data/'


def listfiles():
    'List data files'
    from os import listdir
    for f in sorted(listdir(DATADIR)):
        if not '_dna' in f:
            continue
        group = f[len('image_'):]
        group = int(group[:2])
        yield f,group

def load_image(f):
    'Load the triple (PROTEIN, DNA, ROIS)'
    import mahotas as mh
    f = DATADIR+f
    return mh.imread(f.replace('dna', 'protein')), \
            mh.imread(f), \
            mh.imread(f.replace('dna', 'rois'))

FEATURES = [
    ('pixel', pixel_features),
    ('regions', hypersegmented_features),
    ('surf(1)', lambda p,d,r: surf_grid(p,d,r, 1)),
    ('surf(2)', lambda p,d,r: surf_grid(p,d,r, 2)),
    ('surf(4)', lambda p,d,r: surf_grid(p,d,r, 4)),
    ('surf(8)', lambda p,d,r: surf_grid(p,d,r, 8)),
]

fractions = []
features = []
images = list(listfiles())
for im,g in images:
    protein,dna, rois = load_image(im)
    features.append([feats(protein,dna,rois) for name,feats in FEATURES])
    fractions.append(np.mean(rois > 0))
    print("Computed features for image {} (out of {}).".format(len(fractions), len(images)))

features= np.array(features, dtype=object)
reg = Combined(random_state=1,  n_jobs=N_JOBS)
print("Fitting model...")
reg.fit(features, fractions)

print("Saving model...")
pickle.dump(reg, gzip.open('model.pkl.gz', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
print("Done.")

