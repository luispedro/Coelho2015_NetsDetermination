import numpy as np
import mahotas as mh
from mahotas.features import haralick

def _segment(histone):
    markers = np.zeros_like(histone)
    markers[16::32, 16::32] = 1
    markers,_ = mh.label(markers)
    regions = mh.cwatershed(histone.max() - mh.gaussian_filter(histone, 1.2).astype(int), markers)
    sizes = mh.labeled.labeled_size(regions.astype(np.intc))
    invalid, = np.where(sizes < 512)
    return mh.labeled.relabel(mh.labeled.remove_regions(regions.astype(np.intc), invalid))

def _region_features_for(histone, dna, region):
    pixels0 = histone[region].ravel()
    pixels1 = dna[region].ravel()
    bin0 = pixels0 > histone.mean()
    bin1 = pixels1 > dna.mean()
    overlap = [
            np.corrcoef(pixels0, pixels1)[0,1],
            (bin0&bin1).mean(),
            (bin0|bin1).mean(),
            ]

    spi = mh.sobel(histone, just_filter=1)
    sp = spi[mh.erode(region)]
    sdi = mh.sobel(dna, just_filter=1)
    sd = sdi[mh.erode(region)]
    sobels = [
        np.dot(sp,sp)/len(sp),
        np.abs(sp).mean(),
        np.dot(sd,sd)/len(sd),
        np.abs(sd).mean(),
        np.corrcoef(sp,sd)[0,1],
        np.corrcoef(sp, sd)[0,1]**2,
        sp.std(),
        sd.std(),
        ]

    return np.concatenate([
            [region.sum()],
            haralick(histone * region, ignore_zeros=True).mean(0),
            haralick(dna * region, ignore_zeros=True).mean(0),
            overlap,
            sobels,
            haralick(mh.stretch(sdi * region), ignore_zeros=True).mean(0),
            haralick(mh.stretch(spi * region), ignore_zeros=True).mean(0),
            ])

def hypersegmented_features(histone, dna, rois):
    '''\
    features,labels = hypersegmented_features(histone, dna, rois)
    features = hypersegmented_features(histone, dna, None)

    Computes hyper-segmented features on (histone/dna). If ``rois`` is not
    ``None``, returns the label for each region (fraction of NETs).

    Parameters
    ----------
    histone : ndarray
    dna : ndarray
    rois : ndarray or None

    Returns
    -------
    features : ndarray
        2D array features for each region
    labels : ndarray (only if ``rois is not None``)
        for the corresponding region, returns the fraction of NETs
    '''
    if rois is not None:
        interest = (rois > 0)

    regions, n_regions = _segment(histone)
    histone = mh.stretch(histone)
    dna = mh.stretch(dna)
    features = []
    labels = []
    for ni in range(n_regions):
        region = (regions == (ni+1))
        if region.sum() < 16:
            continue
        features.append(_region_features_for(histone, dna, region))
        if rois is not None:
            fraction = interest[region].mean()
            labels.append(fraction)
    if rois is not None:
        return np.array(features), np.array(labels)
    return np.array(features)
