# Helpers to perform batch analysis of various propagator models.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scorr
import propagator as prop

# Helpers
# ============================================================================

def complete_data_columns(
        tt,
        split_dates=True
    ):
    """Fill in columns necessary for models"""
    
    if not 'sc' in tt:
        tt['sc'] = tt['sign'] * tt['change']
    if not 'sn' in tt:
        tt['sn'] = tt['sign'] * (~tt['change'])
        
    if split_dates and not 'sample' in tt:
        # two sample groups based on date
        dt = np.zeros(len(tt), dtype=int)
        dt[1:] = (
            np.cumsum(
                np.array([t.days for t in np.diff(tt['date'])], dtype=int)
            ) % 2
        )
        tt['sample'] = dt


def get_trade_split(tt, split_by):
    "Get dict of sample key-mask pairs for splitting trades into groups."
    if split_by:
            samples = tt[split_by].unique()
            masks   = {m: (tt[split_by] == m) for m in samples}
    else:
        masks   = {'all': slice(None)}
        
    return masks
    
def shift(x, n, val=np.nan):
    """Shift array, pad with fixed value.
    
    Example: Convert r_1 = p(t+1) - p(t) to causal return p(t) - p(t-1)
             without losing timesteps with pad(r_1, 1).
    """
    
    if n == 0:
        return x
    else:
        res = np.empty_like(x)
        if n > 0:
            res[:n] = val
            res[n:] = x[:-n]
        else:
            res[n:] = val
            res[:n] = x[n:]
        return res
    
# Analyse Trades
# ============================================================================

def calibrate_models(
        tt, # true trades df
        nfft='pad', 
        group=False,
        models = ['cim','tim1','tim2','hdim2','hdim2_x2']
    ):
    """Return dict with correlations, kernels, and responses.
    Calculate sign & price change correlations, response functions, 
    and fitted propagator kernels for trades in DataFrame.
    """
    
    # store results
    res = {}
    
    # intra day mask
    mask0 = ~tt['r1'].isnull()
    
    # apply intra-day mask
    tt = tt[mask0]
        
    # get same optimal nfft used by pna functions
    nfft_opt, events_required = scorr.get_nfft(nfft, tt.groupby('date')['r1'])
    maxlag = int(min(nfft_opt/2, events_required))
    
    res['maxlag'] = maxlag
    
    # correlations and responses
    # ------------------------------------------------------------------------
    kwargs = {'subtract_mean': False, 'norm': 'cov'}
    if not group:
        # one continuous time-series
        s  = tt['sign'].values
        c  = tt['change'].values
        r  = tt['r1'].values
        sc = tt['sc'].values
        sn = tt['sn'].values
        # "normal" correlations
        res['sacorr'] = scorr.fftcrop(scorr.acorr(s, **kwargs), maxlag)
        res['cccorr'] = scorr.fftcrop(scorr.xcorr(sc, sc, **kwargs), maxlag)
        res['nncorr'] = scorr.fftcrop(scorr.xcorr(sn, sn, **kwargs), maxlag)
        res['cncorr'] = scorr.fftcrop(scorr.xcorr(sc, sn, **kwargs), maxlag)
        res['nccorr'] = scorr.fftcrop(scorr.xcorr(sn, sc, **kwargs), maxlag)
        # triple cross correlations
        if 'hdim2' in models:
            res['ccccorr'] = scorr.x3corr(
                c, sc, sc, nfft=2*maxlag, pad=maxlag, **kwargs
            )
            res['nnccorr'] = scorr.x3corr(
                c, sn, sn, nfft=2*maxlag, pad=maxlag, **kwargs
            )
            res['cnccorr'] = scorr.x3corr(
                c, sc, sn, nfft=2*maxlag, pad=maxlag, **kwargs
            )
        # responses
        signed_lags, S, R   = prop.response(r, s, maxlag=maxlag)
        signed_lags, Sc, Rc = prop.response(r, sc, maxlag=maxlag)
        signed_lags, Sn, Rn = prop.response(r, sn, maxlag=maxlag)
        res['signed_lags']  = signed_lags
        res['S'] = S 
        res['R'] = R
        res['Sc'] = Sc
        res['Rc'] = Rc
        res['Sn'] = Sn
        res['Rn'] = Rn
        res['cmean'] = tt['change'].mean()
    else:
        # treat days separately
        res['sacorr'] = scorr.acorr_grouped_df(
            tt, ['sign'], nfft=nfft, return_df=False, **kwargs)[0][:maxlag]
        res['cccorr'] = scorr.fftcrop(
            scorr.xcorr_grouped_df(
                tt, ['sc','sc'], nfft=nfft, return_df=False, **kwargs
            )[0], 
            maxlag
        )
        res['nncorr'] = scorr.fftcrop(
            scorr.xcorr_grouped_df(
                tt, ['sn', 'sn'], nfft=nfft, return_df=False, **kwargs
            )[0], 
            maxlag
        )
        res['cncorr'] = scorr.fftcrop(
            scorr.xcorr_grouped_df(
                tt, ['sc', 'sn'], nfft=nfft, return_df=False, **kwargs
            )[0], 
            maxlag
        )
        res['nccorr'] = scorr.fftcrop(
            scorr.xcorr_grouped_df(
                tt, ['sn', 'sc'], nfft=nfft, return_df=False, **kwargs
            )[0], 
            maxlag
        )
        # triple cross correlations
        if 'hdim2' in models:
            res['ccccorr'] = scorr.x3corr_grouped_df(
                tt, ['change', 'sc', 'sc'], nfft=nfft, **kwargs
            )[0]
            res['nnccorr'] = scorr.x3corr_grouped_df(
                tt, ['change', 'sn', 'sn'], nfft=nfft, **kwargs
            )[0]
            res['cnccorr'] = scorr.x3corr_grouped_df(
                tt, ['change', 'sc', 'sn'], nfft=nfft, **kwargs
            )[0]
        # responses
        signed_lags, S, R   = prop.response_grouped_df(
            tt, ['r1', 'sign'], nfft=nfft
        )
        signed_lags, Sc, Rc = prop.response_grouped_df(
            tt, ['r1', 'sc'],   nfft=nfft
        )
        signed_lags, Sn, Rn = prop.response_grouped_df(
            tt, ['r1', 'sn'],   nfft=nfft
        )
        res['signed_lags'] = signed_lags
        res['S'] = S
        res['R'] = R
        res['Sc'] = Sc
        res['Rc'] = Rc
        res['Sn'] = Sn
        res['Rn'] = Rn
        res['cmean'] = tt.groupby('date')['change'].mean().mean()
        
    # kernels
    # ------------------------------------------------------------------------

    # TIM1
    if 'tim1' in models:
        res['g'] = prop.calibrate_tim1(res['sacorr'], res['S'], maxlag=maxlag)
    
    # TIM2
    ## estimate kernels
    if 'tim2' in models:
        gn, gc = prop.calibrate_tim2(
            res['nncorr'],
            res['cccorr'],
            res['cncorr'],
            res['nccorr'],
            res['Sn'],
            res['Sc'],
            maxlag=maxlag
        )
        res['gc'] = gc
        res['gn'] = gn

    # HDIM2
    if 'hdim2' in models:
        kn, kc = prop.calibrate_hdim2(
            res['nnccorr'],
            res['ccccorr'],
            res['cnccorr'],
            res['Sn'], 
            res['Sc'], 
            maxlag=maxlag,
        )
        res['kn'] = kn
        res['kc'] = kc
    if 'hdim2_x2' in models:
        kn, kc = prop.calibrate_hdim2(
            prop.corr_mat(res['nncorr'], maxlag=maxlag),
            prop.corr_mat(res['cccorr'], maxlag=maxlag),
            prop.corr_mat(res['cncorr'], maxlag=maxlag),
            res['Sn'], 
            res['Sc'], 
            maxlag=maxlag,
        )
        res['kn_x2'] = kn
        res['kc_x2'] = kc
    return res
    
# Run Models
# ============================================================================

def calc_models(
        dbc,
        nfft              = 'pad', # also for calibration
        group             = False,      # "
        calibrate         = True,
        split_by          = 'sample',
        rshift            = 0, ## sync to causal return, not r1
        models            = ['cim','tim1','tim2','hdim2','hdim2_x2'],
        smooth_kernel     = True
    ):
    """Add propagator(-like) models to trades.  
    
    Pass a dict: calibration is added to dict, not returned.
    Pass trades DataFrame directly: calibration is returned as DataFrame(s)
    
    See also: calibrate_models, aggregate_impact.add_models_to_trades
    """
    # normalise inputs (dict / df)
    if 'tt' in dbc:
        tt = dbc['tt']
    else:
        tt = dbc
        dbc = {'tt': tt}
        
    # kernel preprocessing
    if smooth_kernel:
        if type(smooth_kernel) == dict:
            kern = lambda x: prop.smooth_tail_rbf(x **smooth_kernel)
        else:
            kern = lambda x: prop.smooth_tail_rbf(x)
    else:
        kern = lambda x: x
    
    # propagator ingredients
    complete_data_columns(tt, split_dates=split_by)

    # get masks for different samples (groups of events, normally days)
    masks   = get_trade_split(tt, split_by)
    samples = masks.keys()
    
    for i in range(len(samples)):
        # get calibration for a sample
        ## get a sample name and the corresponding mask
        s = samples[i-1]
        m = masks[s]
        
        ## calculate now or rely on existing calibration?
        if calibrate:
            cal = calibrate_models(
                tt.loc[m], nfft=nfft, group=group, models=models
            )
            if 'cal' in dbc:
                dbc['cal'][s] = cal
            else:
                dbc['cal'] = {s: cal}
        else:
            cal = dbc['cal'][s]
        
        # switch mask to a different sample
        ## keeping the above calibration which is then out-of-sample
        s = samples[i]
        m = masks[s]

        # now run the models
        ## model output -> trades df (sync to regular return, not r1)
        if not group:
            ## treat everything in the sample as one continuous series
            if 'cim' in models:
                tt.loc[m,'r_cim'] = (
                    tt.loc[m,'change'] * tt.loc[m,'sign']
                ).shift(rshift)
                #tt['r_cps'] = (
                #    np.random.permutation(tt['change']) * tt['sign']
                #).shift(1)
            if 'tim1' in models:
                tt.loc[m,'r_tim1']  = shift(
                    prop.tim1(
                        tt.loc[m,'sign'], 
                        kern(cal['g'])
                    ),
                    rshift
                )
            if 'tim2' in models:
                tt.loc[m,'r_tim2'] = shift(
                    prop.tim2(
                       tt.loc[m,'sign'], 
                       tt.loc[m,'change'], 
                       kern(cal['gn']), 
                       kern(cal['gc'])
                   ),
                   rshift
                )
            if 'hdim2' in models:
                tt.loc[m,'r_hdim2'] = shift(
                    prop.hdim2(
                        tt.loc[m,'sign'], 
                        tt.loc[m,'change'], 
                        kern(cal['kn']), 
                        kern(cal['kc'])
                    ),
                    rshift
                )
            if 'hdim2_x2' in models:
                tt.loc[m,'r_hdim2_x2'] = shift(
                    prop.hdim2(
                        tt.loc[m,'sign'], 
                        tt.loc[m,'change'], 
                        kern(cal['kn_x2']), 
                        kern(cal['kc_x2'])
                    ),
                    rshift
                )
        else:
            # simulate models for each day in the sample independently
            g = tt.loc[m].groupby('date')
            rcim      = []
            rtim1     = []
            rtim2     = []
            rhdim2    = []
            rhdim2_x2 = []
            # kernels (smoothing takes time so don't don't in the loop)
            g_tim1   = kern(cal['g'])
            gn_tim2  = kern(cal['gn'])
            gc_tim2  = kern(cal['gc'])
            kn_hdim2 = kern(cal['kn'])
            kc_hdim2 = kern(cal['kc'])
            kn_hdim2_x2 = kern(cal['kn_x2'])
            kc_hdim2_x2 = kern(cal['kc_x2'])
            # simulate for all groups
            for i, (gk, gv) in enumerate(g):
                rcim.append((gv['change'].astype(float) * gv['sign']))
                if 'tim1' in models:
                    rtim1.append(
                        shift(
                            prop.tim1(
                                gv['sign'], g_tim1
                            ),
                            rshift
                        )
                    )
                if 'tim2' in models:
                    rtim2.append(
                        shift(
                             prop.tim2(
                                gv['sign'], 
                                gv['change'], 
                                gn_tim2, 
                                gc_tim2
                            ),
                            rshift
                        )
                    )
                if 'hdim2' in models:
                    rhdim2.append(
                        shift(
                            prop.hdim2(
                                gv['sign'], 
                                gv['change'], 
                                kn_hdim2, 
                                kc_hdim2
                            ),
                            rshift
                        )
                    )
                if 'hdim2_x2' in models:
                    rhdim2_x2.append(
                        shift(
                            prop.hdim2(
                                gv['sign'], 
                                gv['change'], 
                                kn_hdim2_x2, 
                                kc_hdim2_x2
                            ),
                            rshift
                        )
                    )
            if 'cim' in models:    tt.loc[m,'r_cim']    = np.concatenate(rcim)
            if 'tim1' in models:   tt.loc[m,'r_tim1']  = np.concatenate(rtim1)
            if 'tim2' in models:   tt.loc[m,'r_tim2']  = np.concatenate(rtim2)
            if 'hdim2' in models:  tt.loc[m,'r_hdim2'] = np.concatenate(rhdim2)
            if 'hdim2_x2' in models:
                tt.loc[m,'r_hdim2_x2'] = np.concatenate(rhdim2_x2)
        # end if group
        
    # end loop over samples