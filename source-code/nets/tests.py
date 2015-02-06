import numpy as np
import mahotas as mh
from . import regions
from . import pixel

histone = mh.imread('data/image_00_00_protein.png')
dna = mh.imread('data/image_00_00_dna.png')
rois = mh.imread('data/image_00_00_rois.png')

def test_rois_None():
    def test_function(f):
        features, labels = f(histone, dna, rois)
        features2 = f(histone, dna, None)
        assert np.all(features == features2)
        assert np.all(features == features2)
        assert len(features) == len(labels)
        assert np.abs(np.mean(rois > 0) - np.dot(features.T[0],labels)/np.sum(features.T[0])) < .05
    yield test_function, regions.hypersegmented_features
    yield test_function, pixel.pixel_features
    yield test_function, (lambda h,d,r : pixel.surf_grid(h,d,r,1.0))

