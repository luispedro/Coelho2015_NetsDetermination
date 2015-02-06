=================================
Determining NET Coverage Tutorial
=================================

    Companion to the paper *Automatic Determination of NET (Neutrophil
    Extracellular Traps) Coverage in Fluorescent Microscopy Images* by Coelho
    et al. (2015, in review)

Upon publication, the code will be available under the MIT license, including
scripts which generate the figures in the paper.

The code is split into two parts:

1. A general implementation of the algorithms
2. A driver module which uses the algorithms for our data and generates the
   figures in the paper.

To reproduce the paper, you should use the script ``reproduce.sh`` bundled with
the source, which from the raw data produces all the figures (including
supplemental figures) as well as outputing tables with all the numbers
mentioned in the manuscript.

In this document, we explain how to use the general algorithms on new images.

Dependencies
------------

This software works in Python, using the following dependencies:

- numpy
- `mahotas <http://mahotas.rtfd.org>`__
- `scikit-learn <http://www.scikit-learn.org>`__

These are imported as the first step, setting up the standard abbreviations::

    import numpy as np
    import mahotas as mh

Single model prediction
-----------------------

Feature computation
~~~~~~~~~~~~~~~~~~~

For the purpose of this tutorial, we assume that the image information is
present in an array called ``image_data``::

    image_data = [
        ('data/image_00_protein.png', 'data/image_00_dna.png', 'data/image_00_rois.png'),
        ('data/image_01_protein.png', 'data/image_01_dna.png', 'data/image_01_rois.png'),
        ...
        ]

This matches format of our data as used for the paper and naturally needs to be
changed for your data. Regions of interest (ROIs) are represented as images
where 0 represents the background and positive numbers represent the different
regions (this is the same format as is returned by the ``mahotas.label`` function).

We will use the *region method* as an example; other methods have a similar
interface::

    from nets.regions import hypersegmented_features
    feature_labels = [hypersegmented_features(mh.imread(histone_path),
                                                mh.imread(dna_path),
                                                mh.imread(rois_path))
                    for histone_path, dna_path, rois_path in image_data]
    features = [fs for fs,_ in features]
    labels = [ells for _,ells in features]

We also need to compute the fraction of area that is a region of interes for
all the images. For a single image, we use the expression::

    im = mh.imread(rois_path)
    fraction = np.mean(im > 0)

We can use a list comprehension to perform the same for the whole dataset::

    fractions = np.array([np.mean(mh.imread(rois_path) > 0)
                    for _,_, rois_path in image_data])


Learning
~~~~~~~~
The learning method is encapsulated in a class ``RandomForestRegressor`` which
follows a similar interface to the objects in `scikit-learn
<http://www.scikit-learn.org>`__::

    from nets.learn import RandomForestRegressor
    learner = RandomForestRegressor()
    learner.fit(features, labels, fractions)

We need to pass the ``fractions`` argument explicitly in order to train the
linear correction.

Application to new images
~~~~~~~~~~~~~~~~~~~~~~~~~

To apply to a new image, you should compute the features again. Note that we
now pass ``None`` to the feature computation code as we do not know the
underlying gold standard for this image::

    features = hypersegmented_features(
                        mh.imread('histone.tiff'),
                        mh.imread('dna.tiff'),
                        None)

Obtaining a prediction is now trivial::

    fraction_estimate = learner.predict(features)

Combining Methods
-----------------

Computing many features
~~~~~~~~~~~~~~~~~~~~~~~

For this interface, we need to compute, for each image, several feature sets
(with their respective ROIs) ::

    features_labels = []
    fractions = []
    for histone, dna, rois in image_data:
        histone = mh.imread(histone)
        dna = mh.imread(dna)
        rois = mh.imread(rois)
        fs = [
            hypersegmented_features(histone, dna, rois),
            pixel_features(histone, dna, rois),
            surf_grid(histone, dna, rois, 1.),
            surf_grid(histone, dna, rois, 2.),
            surf_grid(histone, dna, rois, 4.),
            surf_grid(histone, dna, rois, 8.),
        ]
        features_labels.append(fs)
        fractions.append(np.mean(rois > 0))

The feature sets can extended in a natural fashion.

Learning
~~~~~~~~
We have also implemented another object ``learn.Combined``, which encapsulates
the whole process::

    learner = learn.Combined()
    learner.fit(features_labels, fractions)

Predictions are now also trivial. We compute the same feature set (need to be
in the same exact order as for training!) and call ``learner.predict``::

    histone = 'testing_histone.tiff'
    dna = 'testing_dna.tiff'
    fs = [
        hypersegmented_features(histone, dna, None),
        pixel_features(histone, dna, None),
        surf_grid(histone, dna, None, 1.),
        surf_grid(histone, dna, None, 2.),
        surf_grid(histone, dna, None, 4.),
        surf_grid(histone, dna, None, 8.),
    ]
    learner.predict(features)

Note that we passed ``None`` as the ROI channel to denote that we don't have
labels for this image.

