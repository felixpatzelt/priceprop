import unittest
import numpy as np
import pandas as pd
import json
import priceprop.propagator as prop

# some basic checks, not meant to replace manual testing with real data!

class TestIntegrate(unittest.TestCase):
    def test(self):
        np.testing.assert_almost_equal(prop.integrate([1,2,3]), [0,1,3])
        
class TestSmoothTailRBF(unittest.TestCase):
    def test(self):
        x = np.arange(1,100, dtype=float)**-1
        s = prop.smooth_tail_rbf(x)
        np.testing.assert_allclose(s, x, rtol=.01, atol=.01)

class TestPropagate(unittest.TestCase):
    def test_pulse(self):
        np.testing.assert_almost_equal(
            prop.propagate([1,0,0], [1,.5,0]),
            [1,.5,0]
        )
        
class TestResponse(unittest.TestCase):
    
    def test_lsr(self):
        # test simple response
        args = ([-1,0,1,0], [0,1,0,0])
        l,s,r = prop.response(*args)
        np.testing.assert_almost_equal(
            l, range(-2,3)
        )
        np.testing.assert_almost_equal(
            s, [0, -.25, 0, .25, 0]
        )
        np.testing.assert_almost_equal(
            r, [.25,.25,0,0,.25]
        )
        # test ret argument
        np.testing.assert_almost_equal(
            l, prop.response(*args, ret='l')
        )
        np.testing.assert_almost_equal(
            s, prop.response(*args, ret='s')
        )
        np.testing.assert_almost_equal(
            r, prop.response(*args, ret='r')
        )
        
    def test_grouped(seld):
        df = pd.DataFrame({
            'r':    [-2, 0, 1, 0,-4, 0, 1, 0],
            's':    [ 0, 1, 0, 0, 0, 1, 0, 0], 
            'date': [ 0, 0, 0, 0, 1, 1, 1, 1]
        })
        l, s, r = prop.response_grouped_df(df, ['r','s'])
        np.testing.assert_almost_equal(
            l, range(-2,3)
        )
        np.testing.assert_almost_equal(
            s, [ 0., -0.75, 0., 0.25, 0.]
        )
        np.testing.assert_almost_equal(
            r, [.75,.75,0,0,.25]
        )
        
        
class TestTIM1(unittest.TestCase):    
    def test_tim1(self):
        np.testing.assert_almost_equal(
            prop.tim1([1,0,0,0], [1,.5,0,0]), 
            [1,.5,0,0]
        )

    def test_calibrate_tim1(self):
        np.testing.assert_almost_equal(
            prop.calibrate_tim1([.25,0,0,0], [0,0,0,.25,.125,0,0], maxlag=4),
            [1,.5,0,0]
        )
        
class TestTIM2(unittest.TestCase):
    def test_tim2(self):
        np.testing.assert_almost_equal(
            prop.tim2([1,-1], np.array([1,0,], dtype=bool),[1,0,],[.1,0,]),
            [.1,-1]
        )
    def test_calibrate_tim2(self):
        gn, gc = prop.calibrate_tim2(
            [1,0],[1,0],[0,0],[0,0],[0,.1,0],[0,1,0], maxlag=2
        )
        np.testing.assert_almost_equal(gn, [.1,0])
        np.testing.assert_almost_equal(gc, [1,0])
        
class TestHDIM2(unittest.TestCase):
    def test_hdim2(self):
        np.testing.assert_almost_equal(
            prop.hdim2([1,-1], np.array([1,0,], dtype=bool),[1,0,],[.1,0,]),
            [.1,0]
        )
        
    def test_calibrate_hdim2(self):
        # It is somehow difficult to find tiny example like above where the 
        # three-point correlation matrix isn't singular!
        with open('tests/hdim2_test.json', 'r') as f:
            lpars = {
                k: np.array(v) if hasattr(v, '__len__') 
                else v 
                for k,v in json.load(f).iteritems()
            }
        kn_est, kc_est = prop.calibrate_hdim2(
            lpars['Cnnc'], 
            lpars['Cccc'], 
            lpars['Ccnc'], 
            lpars['Sn'], 
            lpars['Sc'], 
            maxlag=lpars['maxlag']
        )
        np.testing.assert_allclose(lpars['kn'], kn_est, rtol=.01, atol=.01)
        ## The required file can be saved as follows:
        #savepars = {
        #    'Cnnc': Cnnc[:maxlag,:maxlag], # an estimation
        #    'Cccc': Cccc[:maxlag,:maxlag],
        #    'Ccnc': Ccnc[:maxlag,:maxlag],
        #    'Sn':   Sn[:2*maxlag],
        #    'Sc':   Sc[:2*maxlag],
        #    'maxlag': maxlag,
        #    'kn':   kn, # the real kernel
        #    'kc':   kc
        #}
        #with open('hdim2_test.json', 'w') as f:
        #    json.dump(
        #        {
        #            k: v.tolist() if hasattr(v, '__len__') 
        #            else v 
        #            for k,v in savepars.iteritems()
        #        },
        #        f
        #    )