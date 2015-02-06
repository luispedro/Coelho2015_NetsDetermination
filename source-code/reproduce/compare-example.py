from matplotlib import pyplot as plt
import numpy as np
import mahotas as mh
from glob import glob
MU_PER_PIXEL = 0.69

f = sorted(glob('data/*rois2.png'))[0]
protein = mh.imread(f.replace('rois2','protein'))
dna = mh.imread(f.replace('rois2','dna'))
rois = mh.imread(f.replace('rois2','rois'))
rois2 = mh.imread(f)

def composite(dna, protein, rois):
    '''Build composite image

    Parameters
    ----------
    dna : ndarray
    protein : ndarray
    rois : ndarray

    Returns
    -------
    comp : ndarray of shape (h,w,3)
        RGB image
    '''
    borders = 255*mh.borders(rois != 0)
    borders = np.dstack([borders, borders, borders])
    im = np.maximum(borders, mh.as_rgb(dna, protein, 0)).astype(np.uint8)
    im[492:500,400:400+50/MU_PER_PIXEL] = 255
    return im


plt.subplot(1,2,1)
plt.imshow(composite(dna, protein, rois))
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(composite(dna, protein, rois2))
plt.xticks([])
plt.yticks([])
fig = plt.gcf()
fig.tight_layout()
fig.set_figwidth(6)
fig.set_figheight(3)
plt.savefig('figures/compare-example.png', dpi=300)
plt.savefig('figures/compare-example.pdf')

rois = (rois != 0)
rois2 = (rois2 != 0)
total = np.sum(rois)
total2 = np.sum(rois2)
s = float(rois.size)
c00, c10, c01, c11 = np.bincount(rois.ravel() + 2 *rois2.ravel())

print("Total     {}      \t{}".format(total, total2))
print("Fraction  {:.1%}  \t{:.1%}".format(total/s, total2/s))
print("================================")
print("Contigency Matrix")
print("================================")
print("           Operator 1")
print("Operator 2")
print("          No           Yes")
print("No        {:6}      {:6}".format(c00,c01))
print("Yes       {:6}      {:6}".format(c10,c11))
