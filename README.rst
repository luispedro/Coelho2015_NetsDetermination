============================  
Determination of NET Content
============================  

This package contains support material (code & data) for the paper:

    *Automatic Determination of NET (Neutrophil Extracellular Traps) Coverage
    in Fluorescent Microscopy Images* by Luis Pedro Coelho, Catarina Pato, Ana
    Friães, Ariane Neumann, Maren von Köckritz-Blickwede, Mário Ramirez, and
    João André Carriço. Currently under review.

Upon publication, all the code will be made available under the MIT license.
Use in academic publications should cite the paper above.

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

Adapting to your own data
~~~~~~~~~~~~~~~~~~~~~~~~~

There is a detailed tutorial on how to use the library.

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
