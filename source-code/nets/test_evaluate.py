from .learn import evaluate
import numpy as np
def test_evaluate():
    fs  = np.random.random(10)
    assert evaluate(fs, fs) == (1.,1.)
    q2,r2 = evaluate(fs*2, fs)
    assert np.abs(r2 - 1.) < 1e-4
    assert np.abs(q2 - 1.) > .1
    q2,r2 = evaluate(fs*1.2+3., fs)
    assert np.abs(q2 - 1.) > .1
    assert np.abs(r2 - 1.) < 1e-4
    assert np.abs(evaluate(fs*0 + fs.mean(), fs).q2) <  .0001
    assert evaluate(-fs, fs).q2 < 0

    for _ in range(32):
        r = evaluate(np.random.random(len(fs))*.4+fs, fs)
        assert r.q2 < r.r2

    assert np.abs(
        evaluate(np.array([0,0,.5,.5,.5,.5]),np.array([0,0,0,0,1,1])).q2
        - 0.25) < .01
