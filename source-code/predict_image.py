import mahotas as mh
import gzip

from nets.pixel import surf_grid, pixel_features
from nets.regions import hypersegmented_features
import multiprocessing
from os import path


p = multiprocessing.Pool(6)
MODEL_FILE = 'model.pkl.gz'
model = None

def read_2dimage(fname):
    'Read file to 2d image'
    im = mh.imread(fname)
    if len(im.shape) == 3:
        im = im.max(2)
    return im

def predict_image(dnafile, proteinfile, verbose=False):
    '''Return a prediction

    Parameters
    ----------
    dnafile : file path
    proteinfile : file path

    Returns
    -------
    fraction : float
    '''
    global model
    if model is None:
        if verbose:
            print("Loading model...")
        import cPickle as pickle
        model = pickle.load(gzip.open(MODEL_FILE))
    if verbose:
        print("Computing features...")

    dna = read_2dimage(dnafile)
    protein = read_2dimage(proteinfile)
    features = [
            p.apply_async(pixel_features, (dna, protein, None)),
            p.apply_async(hypersegmented_features, (dna, protein, None)),
            p.apply_async(surf_grid, (dna, protein, None, 1)),
            p.apply_async(surf_grid, (dna, protein, None, 2)),
            p.apply_async(surf_grid, (dna, protein, None, 4)),
            p.apply_async(surf_grid, (dna, protein, None, 8)),
            ]
    features = [f.get() for f in features]
    preds = model.predict(features)
    return preds[0]



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Classify images')
    parser.add_argument('--n_estimators', dest='n_estimators', action='store',
                               default=100, help='This option is only used if a model is to be learned. It specifies the number of estimators in each random forest.\nIn the paper, 100 was used, which requires ~10 GB of RAM. A smaller number consumes less memory, but may result in less variable results. In our (unreported) testing, even a number as small as 10 was still quite good (same average performance, but higher variance). Thus, if memory usage is an issue, try a small number.')

    parser.add_argument('--model', dest='model', action='store',
                        default=MODEL_FILE, help='Model file to use. This an option for advanced users. Normally the default is fine.')

    parser.add_argument('--quiet', dest='quiet', action='store_true',
                                default=False, help='Turn of verbose output')

    parser.add_argument('DNA_file', nargs=1, action='store')
    parser.add_argument('Histone_file', nargs=1, action='store')

    args = parser.parse_args()

    if not path.exists(args.model):
        if args.verbose:
            print("Model has not previously been created.")
            print("Creating now (this will take a long time, but only needs to be run once)...")
        import create_model
        create_model.build_model(args.n_estimators, verbose=(not args.quiet), ofilename=args.model)

    prediction = predict_image(args.DNA_file[0], args.Histone_file[0], verbose=(not args.quiet))
    print("Prediction for image DNA={[0]} & HISTONE={[0]} is {:.1%}".format(args.DNA_file, args.Histone_file, prediction))

