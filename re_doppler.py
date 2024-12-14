# This is not our final frequency shifting codes, we use this to do test at the beginning, this is from
#https://github.com/meinardmueller/libtsm/blob/master/libtsm/tsm.py
# Our own frequency shifting code is in doppler_mul_freq3.py
import numpy as np
import scipy.signal
import scipy.interpolate
from scipy.interpolate import CubicSpline
def win(win_len, beta) -> np.ndarray:

    w = scipy.signal.hann(win_len) ** beta
    return w

def cross_corr(x, y, win_len) -> np.ndarray:
    cc = np.convolve(np.flip(x), y)
    # restrict the cross correlation result to just the relevant values
    # Values outside of this range are related to deltas bigger or smaller than our tolerance values.
    cc = cc[win_len-1:-(win_len-1)]
    return cc


def pitch_shift(x, p, t_p=None, Fs=22050, **kwargs) -> np.ndarray:
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    t_x = np.linspace(0, (len(x) - 1) / Fs, len(x))

    if not np.isscalar(p):
        if t_p is None:
            raise Exception("t must be specified if p is an array!")
        if len(p) != len(t_p):
            raise Exception("t must have the same length as p!")
        if t_p[0] != 0:  # time axis should start with 0
            t_p = t_p.astype(float)
            t_p = np.insert(t_p, 0, 0)
            p = np.insert(p, 0, 0)
        if t_p[-1] != t_x[-1]:  # time axis should end with the last time instance
            t_p = t_p.astype(float)
            t_p = np.insert(t_p, len(t_p), t_x[-1])
            p = np.insert(p, len(p), 0)

    alpha = 2 ** (-p / 1200)

    # convert pitch shift in cents to (non-linear) time-stretch function tau
    if np.isscalar(p):
        tau = np.array([[0, 0], [x.shape[0] - 1, x.shape[0] * alpha - 1]]) / Fs  # given in seconds
    else:
        # compute tau
        tau = np.zeros((len(alpha), 2))
        tau[:, 0] = t_p

        for i in range(1, len(alpha)):
            dt = tau[i, 0] - tau[i - 1, 0]
            tau[i, 1] = dt * alpha[i-1] + tau[i - 1, 1]

    # Pitch-shifting
    # (Non-linear) Resampling
    # fi = sc.interpolate.interp1d(tau[:, 0], tau[:, 1], kind='linear', fill_value="extrapolate")
    # time_input = fi(t_x)

    time_input = np.interp(t_x, tau[:,0],tau[:,1])

    # fi = sc.interpolate.interp1d(time_input, x[:, 0], kind='cubic', fill_value="extrapolate")
    # t_res = np.arange(0, tau[-1, 1] + 1 / Fs, 1 / Fs)
    # y_ps = fi(t_res)
    print("x[:, 0]",x[:, 0])
    spl = CubicSpline(time_input, x[:, 0])
    t_res = np.arange(0, tau[-1, 1] + 1 / Fs, 1 / Fs)
    y_ps = spl(t_res)

    tau_inv = np.hstack((time_input.reshape(-1, 1), t_x.reshape(-1, 1)))
    anchor_points = np.ceil(tau_inv * Fs).astype(int)
    anchor_points = np.flip(anchor_points, axis=0)
    anchor_points = anchor_points[np.unique(anchor_points[:, 0],
                                            return_index=True)[1], :]  # only keep unique indices

    # Time-Scale Modification
    y_ps = wsola_tsm(y_ps, anchor_points, **kwargs)

    # crop if pitch-shifted signal is longer than x
    y_ps = y_ps.reshape(-1, 1)[:len(x), :]

    return y_ps

def wsola_tsm(x, alpha, syn_hop=512, win_length=1024, win_beta=2, tol=512) -> np.ndarray:
        # Pre-calculations
    window = win(win_length, win_beta)

    w = window
    win_len = len(w)
    win_len_half = np.around(win_len / 2).astype(int)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    num_of_chan = x.shape[1]

    # Time-stretch function
    if np.isscalar(alpha):
        anchor_points = np.array([[0, 0], [int(x.shape[0]) - 1, int(np.ceil(alpha * x.shape[0])) - 1]])
    else:
        anchor_points = alpha.astype(int)

    output_length = anchor_points[-1, 1] + 1
    syn_win_pos = np.arange(0, output_length + win_len_half, syn_hop)  # positions of the synthesis winLenHalf
    # windows in the output
    ana_win_pos = np.interp(syn_win_pos, anchor_points[:, 1], anchor_points[:, 0])

    ana_win_pos = np.round(ana_win_pos).astype(int)  # positions of the analysis windows in the input
    ana_hop = np.append([0], ana_win_pos[1:] - ana_win_pos[:-1])  # analysis hop sizes

    # WSOLA
    y = np.zeros((output_length, num_of_chan))  # initialize output
    min_fac = np.min(syn_hop / ana_hop[1:])  # the minimal local stretching factor
    # to avoid that we access x outside its range, we need to zero pad it appropriately

    # x = np.pad(x, [(int(win_len_half + tol), int(np.ceil(1 / min_fac) * win_len + tol)), (0, 0)])
    pad_before = int(win_len_half + tol)

    pad_after = int(np.ceil(1 / min_fac) * win_len + tol)

    x = np.pad(x, [(pad_before, pad_after), (0, 0)])
    ana_win_pos += tol  # compensate for the extra 'tol' padded zeros at the beginning of x

    for c in range(num_of_chan):  # loop over channels
        x_c = x[:, c]
        y_c = np.zeros((output_length + 2 * win_len, 1))  # initialize the output signal
        ow = np.zeros((output_length + 2 * win_len, 1))  # keep track of overlapping windows
        delay = 0  # shift of the current analysis window position

        for i in range(len(ana_win_pos) - 1):
            # OLA
            curr_syn_win_ran = np.arange(syn_win_pos[i], syn_win_pos[i] + win_len, dtype=int)  # range of current
            # synthesis window
            curr_ana_win_ran = np.arange(ana_win_pos[i] + delay, ana_win_pos[i] + win_len + delay, dtype=int)  # range
            # of the current analysis window, shift by 'del' offset
            y_c[curr_syn_win_ran, 0] += x_c[curr_ana_win_ran] * w  # overlap and add
            ow[curr_syn_win_ran, 0] += w  # update the sum of overlapping windows
            natural_indices = curr_ana_win_ran + syn_hop
            nat_prog = x_c[natural_indices]  # 'natural progression' of the last copied audio segment
            a = ana_win_pos[i+1]
            next_ana_win_ran = np.arange(ana_win_pos[i + 1] - tol, ana_win_pos[i + 1] + win_len + tol, dtype=int)  #
            # range where the next analysis window could be located (including the tolerance region)
            x_next_ana_win_ran = x_c[next_ana_win_ran]  # corresponding segment in x

            # Cross Correlation
            cc = cross_corr(x_next_ana_win_ran, nat_prog, win_len)  # compute the cross correlation
            max_index = np.argmax(cc)  # pick the optimizing index in the cross correlation
            delay = tol - max_index  # infer the new 'delay'

        # process last frame
        y_c[syn_win_pos[-1]:syn_win_pos[-1] + win_len, 0] += x_c[ana_win_pos[i] + delay:ana_win_pos[i] + win_len + delay] * w
        ow[syn_win_pos[-1]:syn_win_pos[-1] + win_len, 0] += w

        # re-normalize the signal by dividing by the added windows
        ow[ow < 10 ** (-3)] = 1  # avoid potential division by zero
        y_c /= ow

        # remove zero-padding at the beginning
        y_c = y_c[win_len_half:]

        # remove zero-padding at the end
        y_c = y_c[0:output_length]

        y[:, c] = y_c[:, 0]

    return y

