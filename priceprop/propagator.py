import numpy as np
from scipy.linalg import solve_toeplitz, solve
from scipy.signal import fftconvolve
from scipy.interpolate import Rbf
from scorr import xcorr, xcorr_grouped_df, xcorrshift, fftcrop, corr_mat


# Helpers
# =====================================================================

def integrate(x):
    "Return lag 1 sum, i.e. price from return, or an integrated kernel."
    return np.concatenate([[0], np.cumsum(x[:-1])])
    
    
def smooth_tail_rbf(k, l0=3, tau=5, smooth=1, epsilon=1):
    """Smooth tail of array k with radial basis functions"""
    # interpolate in log-lags
    l = np.log(np.arange(l0,len(k)))
    # estimate functions
    krbf = Rbf(
        l, k[l0:], function='multiquadric', smooth=smooth, epsilon=epsilon
    )
    # weights to blend with original for short lags
    w = np.exp(-np.arange(1,len(k)-l0+1)/ float(tau))
    # interpolate
    knew     = np.empty_like(k)
    knew[:l0]  = k[:l0]
    knew[l0:] = krbf(l) * (1-w) + k[l0:] * w
    #done
    return knew
    
def propagate(s, G, sfunc=np.sign):
    """Simulate propagator model from signs and one kernel.
    Equivalent to tim1, one of the kernels in tim2 or hdim2.
    """
    steps = len(s)
    s  = sfunc(s[:len(s)])
    p = fftconvolve(s, G)[:steps]
    return p

# Responses
# =====================================================================

def _return_response(ret, x, maxlag):
    """Helper for response and response_grouped_df."""
    # return what?
    ret = ret.lower()
    res = []
    for i in ret:
        if i   == 'l':
            # lags
            res.append(np.arange(-maxlag,maxlag+1))
        elif i == 's':
            res.append(
                # differential response
                np.concatenate([x[-maxlag:], x[:maxlag+1]])
            )
        elif i == 'r':    
            res.append(
            # bare response === cumulated differential response
                np.concatenate([
                    -np.cumsum(x[:-maxlag-1:-1])[::-1], 
                    [0], 
                    np.cumsum(x[:maxlag])
                ])
            )
    if len(res) > 1:
        return tuple(res)
    else:
        return res[0]

def response(r, s, maxlag=10**4, ret='lsr', subtract_mean=False):
    """Return lag, differential response S, response R.
    
    Note that this commonly used price response is a simple cross correlation 
    and NOT equivalent to the linear response in systems analysis.
    
    Parameters:
    ===========
    
    r: array-like
        Returns
    s: array-like
        Order signs
    maxlag: int
        Longest lag to calculate
    ret: str
        can include 'l' to return lags, 'r' to return response, and
        's' to return differential response (in specified order).
    subtract_mean: bool
        Subtract means first. Default: False (signal means already zero)
    """
    maxlag = min(maxlag, len(r) - 2)
    s  = s[:len(r)]
    # diff. resp.
    # xcorr == S(0, 1, ..., maxlag, -maxlag, ... -1)
    x = xcorr(r, s, norm='cov', subtract_mean=subtract_mean)
    return _return_response(ret, x, maxlag)

def response_grouped_df(
        df, cols, nfft='pad', ret='lsr', subtract_mean=False, **kwargs
    ):
    """Return lag, differential response S, response R calculated daily.
    
    Note that this commonly used price response is a simple cross correlation 
    and NOT equivalent to the linear response in systems analysis.
    
    Parameters
    ==========
    
    df: pandas.DataFrame
        Dataframe containing order signs and returns
    cols: tuple
        The columns of interest
    nfft:
        Length of the fft segments
    ret: str
        What to return ('l': lags, 'r': response, 's': incremental response).
    subtract_mean: bool
        Subtract means first. Default: False (signal means already zero)
    
    See also response, spectral.xcorr_grouped_df for more explanations
    """
    # diff. resp.
    x = xcorr_grouped_df(
        df, 
        cols,
        by            = 'date', 
        nfft          = nfft, 
        funcs         = (lambda x: x, lambda x: x), 
        subtract_mean = subtract_mean,
        norm          = 'cov',
        return_df     = False,
        **kwargs
    )[0]
    # lag 1 -> element 0, lag 0 -> element -1, ...
    #x = x['xcorr'].values[x.index.values-1]
    maxlag = len(x) / 2
    return _return_response(ret, x, maxlag)
   
# Analytical power-laws
# =====================================================================

def beta_from_gamma(gamma):
    """Return exponent beta for the (integrated) propagator decay 
        G(lag) = lag**-beta 
    that compensates a sign-autocorrelation 
        C(lag) = lag**-gamma.
    """
    return (1-gamma)/2.
    
def G_pow(steps, beta):
    """Return power-law Propagator kernel G(l). l=0...steps"""
    G = np.arange(1,steps)**-beta#+1
    G = np.r_[0, G]
    return G
    
def k_pow(steps, beta):
    """Return increment of power-law propagator kernel g. l=0...steps"""
    return np.diff(G_pow(steps, beta))
    
# TIM1 specific 
# =====================================================================

def estimate_tim1(c, Sl, maxlag=10**4):
    """Return empirical estimate TIM1 kernel
    
    Parameters:
    ===========
    
    c: array-like
        Cross-correlation (covariance).
    Sl: array-like
        Price-response. If the response is differential, so is the returned
        kernel.
    maxlag: int
        length of the kernel.
    See also: integrate, g2_empirical, tim1
 
    """
    lS = int(len(Sl) / 2)
    g = solve_toeplitz(c[:maxlag], Sl[lS:lS+maxlag])
    return g

def tim1(s, G, sfunc=np.sign):
    """Simulate Transient Impact Model 1, return price or return.
    
    Result is the price p when the bare responses G is passed
    and the 1 step ahead return p(t+1)-p(t) for the differential kernel 
    g, where G == numpy.cumsum(g).
    
    Parameters:
    ===========
    
    s: array-like
        Order signs
    G: array-like
        Kernel
    
    See also: estimate_tim1, integrate, tim2, hdim2.
    """
    return propagate(s, G, sfunc=sfunc)

# TIM2 specific
# =====================================================================

def estimate_tim2(
        nncorr, cccorr, cncorr, nccorr, Sln, Slc, maxlag=2**10
    ):
    """
    Return empirical estimate for both kernels of the TIM2.
    (Transient Impact Model with two propagators)
        
    Parameters:
    ===========
    
    nncorr: array-like
        Cross-covariance between non-price-changing (n-) orders.
    cccorr: array-like
        Cross-covariance between price-changing (c-) orders.
    cncorr: array-like
        Cross-covariance between c- and n-orders
    nccorr: array-like
        Cross-covariance between n- and c-orders.
    Sln: array-like
        (incremental) price response for n-orders
    Slc: array-like
        (incremental) price response for c-orders
    maxlag: int
        Length of the kernels.
    
    See also: estimate_tim1, estimate_hdim2
    """
    # incremental response
    lSn = int(len(Sln) / 2)
    lSc = int(len(Slc) / 2)
    S = np.concatenate([Sln[lSn:lSn+maxlag], Slc[lSc:lSc+maxlag]])
    
    # covariance matrix
    mat_fn = lambda x: corr_mat(x, maxlag=maxlag)
    C = np.bmat([
        [mat_fn(nncorr), mat_fn(cncorr)], 
        [mat_fn(nccorr), mat_fn(cccorr)]
    ])
    
    # solve
    g = solve(C, S)
    gn = g[:maxlag]
    gc = g[maxlag:]
            
    return gn, gc

def tim2(s, c, G_n, G_c, sfunc=np.sign):
    """Simulate Transient Impact Model 2
    
    Returns prices when integrated kernels are passed as arguments
    or returns for differential kernels.
    
    Parameters:
    ===========
    s: array
        Trade signs
    c: array
        Trade labels (1 = change; 0 = no change)
    G_n: array
        Kernel for non-price-changing trades
    G_c: array
        Kernel for price-changing trades
    sfunc: function [optional]
        Function to apply to signs. Default: np.sign.
        
    See also: estimate_tim2, tim1, hdim2.
    """
    assert c.dtype == bool, "c must be a boolean indicator!"
    return propagate(s * c, G_c) + propagate(s * (~c), G_n)

    
# HDIM2 specific
# =====================================================================

def estimate_hdim2(
        Cnnc, Cccc, Ccnc, Sln, Slc,
        maxlag=None, force_lag_zero=True
    ):
    """Return empirical estimate for both kernels of the HDIM2.
    (History Dependent Impact Model with two propagators).
    
    Requres three-point correlation matrices between the signs of one 
    non-lagged and two differently lagged orders.
    We distinguish between price-changing (p-) and non-price-changing (n-)
    orders. The argument names corresponds to the argument order in 
    spectral.x3corr.
    
    Parameters:
    ===========
    Cnnc: 2d-array-like
        Cross-covariance matrix for n-, n-, c- orders.
    Cccc: 2d-array-like
        Cross-covariance matrix for c-, c-, c- orders.
    Ccnc: 2d-array-like
        Cross-covariance matrix for c-, n-, c- orders.
    Sln: array-like
        (incremental) lagged price response for n-orders
    Slc: array-like
        (incremental) lagged price response for c-orders
    maxlag: int
        Length of the kernels.
        
    See also: hdim2,
    """
    maxlag = maxlag or min(len(Cccc), len(Sln))/2
    
    # incremental response
    lSn = int(len(Sln) / 2)
    lSc = int(len(Slc) / 2)
    S = np.concatenate([
        Sln[lSn:lSn+maxlag], 
        Slc[lSc:lSc+maxlag]
    ])
    
    # covariance matrix
    Cncc = Ccnc.T
    C = np.bmat([
        [Cnnc[:maxlag,:maxlag], Ccnc[:maxlag,:maxlag]], 
        [Cncc[:maxlag,:maxlag], Cccc[:maxlag,:maxlag]]
    ])
    
    if force_lag_zero:
        C[0,0] = 1
        C[0,1:] = 0
    
    # solve
    g = solve(C, S)
    gn = g[:maxlag]
    gc = g[maxlag:]
    
    return gn, gc
    
def hdim2(s, c, k_n, k_c, sfunc=np.sign):
    """Simulate History Dependent Impact Model 2, return return.
    
    Parameters:
    ===========
    s: array
        Trade signs
    c: array
        Trade labels (1 = change; 0 = no change)
    k_n: array
        Differential kernel for non-price-changing trades
    k_c: array
        Differential kernel for price-changing trades
    sfunc: function [optional]
        Function to apply to signs. Default: np.sign.
        
    See also: estimate_hdim2, tim2, tim1.
    """
    assert c.dtype == bool, "c must be a boolean indicator!"
    return c * (propagate(s * c, k_c) + propagate(s * (~c), k_n))
