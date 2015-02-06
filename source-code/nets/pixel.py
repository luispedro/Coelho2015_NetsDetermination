def filter_channel(im):
    '''Compute filtered versions of an image

    Parameters
    ----------
    im : ndarray
        input image (2D)

    Returns
    -------
    fs : ndarray
        filtered version of the image (3D array)
    '''
    import numpy as np
    import mahotas as mh
    Tg = im > im.mean()
    T2g = im > (im.mean() + 2*im.std())
    zim = im-im.mean()
    zim /= im.std()
    z8 = mh.gaussian_filter(zim, 8)
    z4 = mh.gaussian_filter(zim, 4)
    z12 = mh.gaussian_filter(zim, 12)
    z16 = mh.gaussian_filter(zim, 16)
    zdiff = z16 - z8
    sob = mh.sobel(im, just_filter=True)
    return np.dstack([
            Tg, T2g, zim, z4, z8, z12, z16, zdiff, sob])


def pixel_features(histone, dna, rois):
    '''
    features,labels = pixel_features(histone, dna, rois, scale)
    features = pixel_features(histone, dna, None, scale)

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
    import numpy as np
    import mahotas as mh
    if rois is not None:
        rois = (rois > 0)
        rois = rois.astype(float)
        rois = mh.gaussian_filter(rois, 8)
    fs = np.dstack((filter_channel(histone), filter_channel(dna)))
    step = 8
    fs = fs[16::step, 16::step]
    fs = fs.reshape((-1,fs.shape[-1]))
    fs = np.hstack((np.ones([len(fs),1]), fs))
    if rois is None:
        return fs
    labels = rois[16::step, 16::step]
    labels = labels.ravel()
    return fs, labels



def surf_grid(red, green, rois, scale):
    '''
    features,labels = surf_grid(histone, dna, rois, scale)
    features = surf_grid(histone, dna, None, scale)

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
    import mahotas.features.surf
    import mahotas as mh
    import numpy as np
    locations = [(x,y,scale,1.,1.) for x in range(32,512,8) for y in range(32,512,8)]
    rfeatures = mahotas.features.surf.descriptors(red, locations, descriptor_only=False)
    gfeatures = mahotas.features.surf.descriptors(green, locations, descriptor_only=True)
    features = np.hstack((rfeatures, gfeatures))
    positions = features[:,:2].astype(int)
    features = features[:,5:]
    features.T[0] = 1

    if rois is None:
        return features
    rois = (rois > 0)
    rois = rois.astype(float)
    rois = mh.gaussian_filter(rois, 8)
    labels = np.array([rois[a,b] for a,b in positions])
    return features, labels
