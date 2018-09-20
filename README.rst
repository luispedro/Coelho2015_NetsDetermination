============================  
Determination of NET Content
============================  

This package contains support material (code & data) for the paper:

    *Automatic Determination of NET (Neutrophil Extracellular Traps) Coverage
    in Fluorescent Microscopy Images* by Luis Pedro Coelho, Catarina Pato, Ana
    Friães, Ariane Neumann, Maren von Köckritz-Blickwede, Mário Ramirez, and
    João André Carriço, Bioinformatics *2015 Jul 15;31(14):2364-70*
    `DOI:10.1093/bioinformatics/btv156
    <http://doi.org/10.1093/bioinformatics/btv156>`__.

Use in academic publications should cite the paper above.

Code is provided under the MIT license.

Data
----

Original data is available in the directory ``data/``. The naming structure is
as follows:

    - prefix "image"
    - nr of the sample
    - nr of the field inside the sample
    - channel (protein, dna, rois, or rois2).

For example, the file ``image_25_00_protein.png`` is from image 25, index 0,
and is the protein channel. The ROI files are the result of human labeling.

See the manuscript for details on data acquisition.

Source code
-----------

The source code is split into two directories

- ``nets`` this is the library code, which is useful to adapt to new projects.
- ``reproduce`` in this directory, you will find all the necessary code to
  reproduce all figures in the paper (including supplemental material).

In addition, there are two helper scripts:

predict_image.py
    ``predict_image.py`` returns a prediction for a single input file. It takes
    two arguments, which should be the image files for the DNA and histone
    channels, respectively::

        python predict_image.py ../data/image_00_00_dna.png ../data/image_00_00_protein.png

    If it cannot find a model to load, then it runs the ``create_model.py`` script.

create_model.py
    This needs to be run once to learn the model from the data. It will look at
    the files in the directory ``../data/`` for its input. Running this step
    may require a lot of memory! If you do not have enough in your machine, you
    can adjust the ``--n-estimators`` parameter to a smaller value.

We only recommend that you use our model trained on our data if your images are
very similar to ours. Otherwise, you can still use our software, but we
recommend you provide our system some training data.

The simplest way to reuse the software is to replace the images in the
``data/`` directory by your own (using the same naming format:
``prefix_dna.png``, ``prefix_protein.png``, and ``prefix_rois.png`` forming a
triple). Note that the ``_rois.png`` image should be a labeled image (i.e.,
pixels with value 0 correspond to background, pixels with value 1 correspond to
the first area of interest, pixels with value 2 to the second area of interest,
...).

Alternatively, there is a detailed tutorial on how to adapt the library for new
uses in the file ``tutorial.rst``.

Dependencies
~~~~~~~~~~~~

For running on your own data:

- numpy
- scikit-learn
- mahotas

Additionally, for reproducing our experiments:

- jug
- pandas
- matplotlib
- seaborn

The file ``requirements.txt`` in the ``source-code`` directory lists all the
requirements. If you have permission to do so, running the following command
inside that directory should install all dependencies::

    pip install -r requirements.txt

If this fails, try::

    sudo pip install -r requirements.txt

Reproducing the paper
~~~~~~~~~~~~~~~~~~~~~

The results of the paper can be reproduced on a Unix-like machine by running
the ``reproduce.sh`` script inside ``source-code/reproduce`` after having
installed the the requirements as listed above.

To use multiple processors, edit this script and set the value of the
``NR_CPUS`` variable.

Files:


jugfile.py
    This is the central file which runs the whole analysis
compare.py
    This script performs the reported comparison between the two operators
bernsen_thresholding.py
    This script evaluates Bernsen thresholding for different sets of parameters
compare-example.py
    This builds a side-by-side Figure showing differences between operators.
draw-composites.py
    This draws composite images for all inputs images

