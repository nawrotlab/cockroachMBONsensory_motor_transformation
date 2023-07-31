import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from numpy import nanmean
from bisect import bisect_right


def _get_tlim(spiketimes):
    try:
        tlim = [np.nanmin(spiketimes[0, :]), np.nanmax(spiketimes[0, :]) + 1]
    except:
        tlim = [0, 1]
    if np.isnan(tlim).any():
        tlim = [0, 1]
    return tlim


def cut_spiketimes(spiketimes, tlim):
    alltrials = list(set(spiketimes[1, :]))
    cut_spikes = spiketimes[:, np.isfinite(spiketimes[0])]
    cut_spikes = cut_spikes[:, cut_spikes[0, :] >= tlim[0]]

    if cut_spikes.shape[1] > 0:
        cut_spikes = cut_spikes[:, cut_spikes[0, :] < tlim[1]]
    for trial in alltrials:
        if not trial in cut_spikes[1, :]:
            cut_spikes = np.append(cut_spikes, np.array([[np.nan], [trial]]), axis=1)
    return cut_spikes

def gaussian_kernel(sigma, dt=1., nstd=3.):
    """ returns a gaussian kernel with standard deviation sigma.
        sigma:  standard deviation of the kernel
        dt:     time resolution
        nstd:   overall width of the kernel specified in multiples
                of the standard deviation.
        """
    t = np.arange(-nstd * sigma, nstd * sigma + dt, dt)
    gauss = np.exp(-t ** 2 / sigma ** 2)
    gauss /= gauss.sum() * dt
    return gauss



def exponential_kernel(sigma, dt=1., nstd=3., twosided=False):
    """ returns an exponential kernel with standard deviation sigma.
        sigma:  standard deviation of the kernel
        dt:     time resolution
        nstd:   overall width of the kernel specified in multiples
                of the standard deviation.
        """
    t = np.arange(-nstd * sigma, nstd * sigma + dt, dt)
    kernel = np.exp(-np.abs(t) / sigma)
    if twosided == False:
        kernel[t < 0] = 0
    kernel /= kernel.sum() * dt
    return kernel


def spiketimes_to_binary(spiketimes, tlim=None, dt=1.):
    """ takes a n array of spiketimes and turns it into a binary
        array of spikes.
            spiketimes:  array where - spiketimes[0,:] -> spike times in [ms]
                                     - spiketimes[1,:] -> trial indices
            tlim:        [tmin,tmax] - time limits for the binary array
                         if None (default), extreme values from spiketimes
                         are taken
            dt:          time resolution of the binary array in [ms]

        returns:
            binary:      binary array (trials,time)
            time:        time axis for the binary array (length = binary.shape[1])
        """
    if tlim is None:
        tlim = _get_tlim(spiketimes)

    time = np.arange(tlim[0], tlim[1] + dt, dt).astype(float)

    if dt <= 1:
        time -= 0.5 * float(dt)

    # trials = np.array([-1]+range(int(spiketimes[1,:].max()+1)))+0.5
    trials = np.array([-1] + list(range(int(spiketimes[1, :].max() + 1)))) + 0.5

    tlim_spikes = cut_spiketimes(spiketimes, tlim)
    tlim_spikes = tlim_spikes[:, np.isnan(tlim_spikes[0, :]) == False]

    if tlim_spikes.shape[1] > 0:
        binary = np.histogram2d(tlim_spikes[0, :], tlim_spikes[1, :], [time, trials])[0].T
    else:
        binary = np.zeros((len(trials) - 1, len(time) - 1))
    return binary, time[:-1]

def kernel_rate(spiketimes, kernel, tlim=None, dt=1., pool=True):
    """ computes a kernel rate-estimate for spiketimes.
            spiketimes:  array where - spiketimes[0,:] -> spike times [ms]
                                     - spiketimes[1,:] -> trial indices
            kernel:      1D centered kernel
            tlim:        [tmin,tmax] time-limits for the calculation
                         if None (default), extreme values from spiketimes
                         are taken
            dt:          time resolution of output [ms]
            pool:        if True, mean rate over all trials is returned
                         if False, trial wise rates are returned
        returns:
            rates:       the rate estimate s^-1 (if not pooled (ntrials,len(time)))
            time:        time axis for rate
        """
    if tlim is None:
        tlim = _get_tlim(spiketimes)

    binary, time = spiketimes_to_binary(spiketimes, tlim, dt)

    if pool:
        binary = binary.mean(axis=0)[np.newaxis, :]

    rates = convolve2d(binary, kernel[np.newaxis, :], 'same')
    kwidth = len(kernel)
    rates = rates[:, int(kwidth / 2):-int(kwidth / 2)]
    time = time[int(kwidth / 2):-int(kwidth / 2)]
    return rates * 1000., time

def alpha_kernel(sigma, dt=1., nstd=3., calibrateMass=True):
    """ returns an exponential kernel with standard deviation sigma.
        sigma:  time constant of the kernel
        dt:     time resolution
        ntau:   overall width of the kernel specified in multiples
                of the tau.
        """
    t = np.arange(-nstd * sigma, nstd * sigma + dt, dt)
    Relu = lambda x: (abs(x) + x) / 2

    if calibrateMass:
        shift = 1.6783 * sigma
    else:
        shift = 0

    kernel = 1 / (sigma ** 2) * Relu(t + shift) * np.exp(-Relu(t + shift) / sigma)
    kernel /= kernel.sum() * dt
    return kernel


def ToFiringRates(data, TW):
    Dur = TW[1] - TW[0]
    FR = len(data) / Dur
    return FR


def ToSpikeRates(data, TWOdor, kernel, dt=1.):
    SpikeTimesPerTrial = data * 1000
    Trials = np.full(len(SpikeTimesPerTrial),
                     0)  # generates array with respective trialID depending on number of spikes
    SpikeTimesPerTrial = np.hstack(
        (SpikeTimesPerTrial, TWOdor[1] * 1000 + 2))  # adds one spiketime after relevant timespan
    Trials = np.hstack((Trials, 0))
    spiketime = np.vstack([SpikeTimesPerTrial, Trials])  # merges spiketimes with respective trialID
    spikerates = kernel_rate(spiketime, kernel, dt=dt, tlim=[TWOdor[0] * 1000, TWOdor[1] * 1000],
                                        pool=True)  # calculates spikerates with moving kernel
    return spikerates[0].tolist()[0]


def ToSpikeRatesTime(data, TWOdor, kernel, dt=1.):
    SpikeTimesPerTrial = data * 1000
    Trials = np.full(len(SpikeTimesPerTrial),
                     0)  # generates array with respective trialID depending on number of spikes
    SpikeTimesPerTrial = np.hstack(
        (SpikeTimesPerTrial, TWOdor[1] * 1000 + 2))  # adds one spiketime after relevant timespan
    Trials = np.hstack((Trials, 0))
    spiketime = np.vstack([SpikeTimesPerTrial, Trials])  # merges spiketimes with respective trialID
    spikerates = kernel_rate(spiketime, kernel, dt=dt, tlim=[TWOdor[0] * 1000, TWOdor[1] * 1000],
                                        pool=True)  # calculates spikerates with moving kernel
    return spikerates[1].tolist()


def CorrectFiringRateOnset(data, sigmaK, KernelF, BL_Rate, dt=1.):
    kernel = KernelF(sigmaK, dt, nstd=6)
    kernel = kernel[len(kernel) // 2:-1]
    # kernel=kernel[1]
    kernel = BL_Rate * kernel / np.max(kernel)
    Datawidth = len(data)
    KernelWidth = len(kernel)
    if Datawidth > KernelWidth:
        kernel = np.pad(kernel, (0, Datawidth - KernelWidth), 'constant', constant_values=0)
    else:
        kernel = kernel[0:Datawidth]
    return (np.array(data) + kernel).tolist()


def CorrectFiringRateOnset_Optimized(data, kernel, BL_Rate):
    # kernel=kernel
    Datawidth = len(data)
    KernelWidth = len(kernel)
    if Datawidth > KernelWidth:
        kernel = BL_Rate * np.pad(kernel, (0, Datawidth - KernelWidth), 'constant', constant_values=0)
    else:
        kernel = BL_Rate * kernel[0:Datawidth]
    return (np.array(data) + kernel).tolist()
