import numpy as np
import pandas as pd
import sys
from Functions import Dataload, HelperFunctions
import os
import glob

try:
    import pandarallel
    import multiprocessing
except:
    pass


def GenerateSignificanceBorders(Data, Percentile=0.95):
    '''
    Generates significance borders for a given baseline distribution
    :param Data: Sampled baseline
    :param Percentile: Percentile to be used for significance borders
    :return: list [LowerLimit, UpperLimit] of significance borders
    '''
    SortedBasel = np.sort(Data)
    LowerInd = len(SortedBasel) * (1 - Percentile)
    UpperInd = len(SortedBasel) * (Percentile)
    LowerLimit = SortedBasel[int(np.floor(LowerInd)) - 1]
    UpperLimit = SortedBasel[int(np.ceil(UpperInd)) - 1]
    return [LowerLimit, UpperLimit]

def findNeuronalOnset(Border, Spikerate, Time):
    '''
    Finds the first timepoint of a given spikerate that is above a given border
    :param Border: Border to be used
    :param Spikerate: Timecourse of spikerates
    :return: Timepoint of first crossing
    '''
    Spikerate=np.array(Spikerate)
    UpperInd=np.where(Spikerate>Border)[0]
    if len(UpperInd):
        return Time[UpperInd[0]]/1000.#TWOdor[0]+UpperInd[0]/1000
    else:
        return np.nan



def Latency(df, sigmaFR, WidthFactor, TW, TW_BL, Border=0.95, MinSpike=1.0, Single_Trial_BL=True, Stims=['A', 'C', 'G'], n_processes=None):
    Edge = sigmaFR * WidthFactor
    kernel = HelperFunctions.exponential_kernel(sigmaFR, 1.0,
                                                nstd=WidthFactor)  # generates kernel for spiketools.kernel_rate
    # if n_processes is None and 'pandarallel' in loaded_modules estimate number of cores
    if 'pandarallel' in sys.modules:
        if n_processes is None:
            n_processes = multiprocessing.cpu_count()
        pandarallel.pandarallel.initialize(nb_workers=n_processes, progress_bar=True)

    # if n_processes is None and 'pandarallel' not in loaded_modules use 1 core
    else:
        n_processes = 1

    df = df.copy(deep=True)


    #limit to stims
    df = Dataload.LimitDFtoStimulus(df, Stims)


    # estimate firing rates
    df.drop(['SpikeRatesBasel', 'SpikeRatesStim'], axis=1, errors='ignore', inplace=True)
    if n_processes > 1:
        df['SpikeRatesBasel'] = df.parallel_apply(
            lambda x: HelperFunctions.ToSpikeRates(x.BaselSpikeTimes, TW_BL, kernel), axis=1)
        print("Debbuging")
        df['SpikeRatesStim'] = df.parallel_apply(
            lambda x: HelperFunctions.ToSpikeRates(x.StimSpikeTimes, [TW[0] - Edge / 1000, TW[1] + Edge / 1000],
                                                   kernel), axis=1)
        df['BLRate'] = df.parallel_apply(
            lambda x: (len(x.BaselSpikeTimes) / np.diff(TW_BL))[0], axis=1)
    else:
        df['SpikeRatesBasel'] = df.apply(
            lambda x: HelperFunctions.ToSpikeRates(x.BaselSpikeTimes, TW_BL, kernel), axis=1)
        df['SpikeRatesStim'] = df.apply(
            lambda x: HelperFunctions.ToSpikeRates(x.StimSpikeTimes, [TW[0] - Edge / 1000, TW[1] + Edge / 1000], kernel), axis=1)
        df['BLRate'] = df.apply(
            lambda x: (len(x.BaselSpikeTimes) / np.diff(TW_BL))[0], axis=1)

    Time = HelperFunctions.ToSpikeRatesTime(df['StimSpikeTimes'].iloc[0],
                                            [TW[0] - Edge / 1000, TW[1] + Edge / 1000], kernel)

    kernelCor = HelperFunctions.exponential_kernel(sigmaFR, 1.0, nstd=6)
    kernelCor /= np.max(kernelCor)
    kernelCor = kernelCor[len(kernelCor) // 2:-1]

    if Single_Trial_BL:
        if n_processes > 1:
            df['SpikeRatesStim'] = df.parallel_apply(
                lambda x: HelperFunctions.CorrectFiringRateOnset_Optimized(x.SpikeRatesStim, kernelCor,
                                                                           x.BLRate), axis=1)
        else:
            df['SpikeRatesStim'] = df.apply(
                lambda x: HelperFunctions.CorrectFiringRateOnset_Optimized(x.SpikeRatesStim, kernelCor, x.BLRate),
                axis=1)
        df[['SpikeRatesBordersLow', 'SpikeRatesBordersHigh']] = df.apply(
            lambda x: GenerateSignificanceBorders(x.SpikeRatesBasel, Percentile=Border), axis=1,
            result_type='expand')
    else:
        tmpDF = df.groupby('RealUnit').agg({'BLRate': np.mean})
        tmpDF.columns = ['BLRate']
        df.drop(columns=['BLRate'], errors='ignore', inplace=True)
        df = df.join(tmpDF, on='RealUnit')
        if n_processes > 1:
            df['SpikeRatesStim'] = df.parallel_apply(
                lambda x: df(x.SpikeRatesStim, kernelCor, x.BLRate), axis=1)
        else:
            df['SpikeRatesStim'] = df.apply(
                lambda x: df(x.SpikeRatesStim, kernelCor, x.BLRate), axis=1)

        tmpDF = df.groupby('RealUnit').agg(
            {'SpikeRatesBasel': lambda y: list(np.hstack(y)), 'BLRate': np.mean}).apply(
            (lambda x: GenerateSignificanceBorders(x.SpikeRatesBasel, Percentile=Border)), axis=1,
            result_type='expand')
        tmpDF.columns = ['SpikeRatesBordersLow', 'SpikeRatesBordersHigh']
        df.drop(columns=['SpikeRatesBordersLow', 'SpikeRatesBordersHigh', 'SpikeRatesBorders'],
                             errors='ignore', inplace=True)
        df = df.join(tmpDF, on='RealUnit')

    df['SpikeRatesBordersHigh'] = df['SpikeRatesBordersHigh'].apply(
        lambda x: x if x > (np.max(kernel) * 1000 * MinSpike) else np.max(kernel) * 1000 * MinSpike)

    df['NeuronalOnset'] = df.apply(
        lambda x: findNeuronalOnset(x.SpikeRatesBordersHigh, x.SpikeRatesStim, Time), axis=1)

    df['Latency'] = df.MLRTime - df.NeuronalOnset
    return df



def Latency_pooled(df, sigmaFR, WidthFactor, TW, TW_BL, Border=0.95, MinSpike=1.0, Stims=['A', 'C', 'G'], n_processes=None):
    Edge = sigmaFR * WidthFactor
    kernel = HelperFunctions.exponential_kernel(sigmaFR, 1.0,
                                           nstd=WidthFactor)  # generates kernel for spiketools.kernel_rate

    # if n_processes is None and 'pandarallel' in loaded_modules estimate number of cores
    if 'pandarallel' in sys.modules:
        if n_processes is None:
            n_processes = multiprocessing.cpu_count()
        pandarallel.pandarallel.initialize(nb_workers=n_processes, progress_bar=True)

    # if n_processes is None and 'pandarallel' not in loaded_modules use 1 core
    else:
        n_processes = 1

    df = df.copy(deep=True)


    # limit to stims
    df = Dataload.LimitDFtoStimulus(df, Stims)
    df = df[df['MLR'] == True].reset_index(drop=True)


    # estimate firing rates
    df.drop(['SpikeRatesBasel', 'SpikeRatesStim'], axis=1, errors='ignore', inplace=True)
    if n_processes > 1:
        df['SpikeRatesBasel'] = df.parallel_apply(
            lambda x: HelperFunctions.ToSpikeRates(x.BaselSpikeTimes, TW_BL, kernel), axis=1)
        print("Debugging")
        df['SpikeRatesStim'] = df.parallel_apply(
            lambda x: HelperFunctions.ToSpikeRates(x.StimSpikeTimes,
                                                   [TW[0] - Edge / 1000, TW[1] + Edge / 1000], kernel),
            axis=1)

        df['BLRate'] = df.parallel_apply(
            lambda x: (len(x.BaselSpikeTimes) / np.diff(TW_BL))[0], axis=1)
    else:
        df['SpikeRatesBasel'] = df.apply(
            lambda x: HelperFunctions.ToSpikeRates(x.BaselSpikeTimes, TW_BL, kernel), axis=1)
        df['SpikeRatesStim'] = df.apply(
            lambda x: HelperFunctions.ToSpikeRates(x.StimSpikeTimes, [TW[0] - Edge / 1000, TW[1] + Edge / 1000],
                                                   kernel), axis=1)
        df['BLRate'] = df.apply(
            lambda x: (len(x.BaselSpikeTimes) / np.diff(TW_BL))[0], axis=1)

    Time = HelperFunctions.ToSpikeRatesTime(df['StimSpikeTimes'].iloc[0],
                                            [TW[0] - Edge / 1000, TW[1] + Edge / 1000], kernel)

    kernelCor = HelperFunctions.exponential_kernel(sigmaFR, 1.0, nstd=6)
    kernelCor /= np.max(kernelCor)
    kernelCor = kernelCor[len(kernelCor) // 2:-1]


    if n_processes > 1:
        df['SpikeRatesStim'] = df.parallel_apply(
            lambda x: HelperFunctions.CorrectFiringRateOnset_Optimized(x.SpikeRatesStim, kernelCor, x.BLRate), axis=1)
    else:
        df['SpikeRatesStim'] = df.apply(
            lambda x: HelperFunctions.CorrectFiringRateOnset_Optimized(x.SpikeRatesStim, kernelCor, x.BLRate),
            axis=1)

    df = df.groupby(['RealUnit', 'StimID']).agg(
        {'SpikeRatesStim': lambda y: list(np.mean(np.vstack(y), axis=0)),
         'SpikeRatesBasel': lambda y: list(np.mean(np.vstack(y), axis=0)), 'MLRTime': np.mean}).reset_index()

    df[['SpikeRatesBordersLow', 'SpikeRatesBordersHigh']] = df.apply(
        lambda x: GenerateSignificanceBorders(x.SpikeRatesBasel, Percentile=Border), axis=1,
        result_type='expand')


    # df['SpikeRatesBordersHigh'] = df['SpikeRatesBordersHigh'].apply(
    #    lambda x: x if x > (np.max(kernel) * 1000 * MinSpike) else np.max(kernel) * 1000 * MinSpike)

    df['NeuronalOnset'] = df.apply(
        lambda x: findNeuronalOnset(x.SpikeRatesBordersHigh, x.SpikeRatesStim, Time), axis=1)

    df['Latency'] = df.MLRTime - df.NeuronalOnset
    return df


def PlotLatencyCDF(DF, title="CDF", color=[(0, 0.6, 0, 1), (0, 0, 0.6, 1),(0, 0, 0.6, 0.5)], TWOdor=[0,2], ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=140)

    MLROnsets = DF.MLRTime.tolist()
    MLROnsets = np.array([x for x in MLROnsets if not np.isnan(x)])
    MLROnsets = np.sort(MLROnsets)
    yNe = np.hstack((0, np.ones_like(MLROnsets)))
    MLROnsets = np.hstack((0, MLROnsets))
    MLROnsets = np.hstack((MLROnsets, TWOdor[1]))

    yNe = np.cumsum(yNe)
    if yNe[-1] > 0:
        yNe /= yNe[-1]
    yNe = np.hstack((yNe, yNe[-1]))

    ax.step(MLROnsets, yNe, where='post', label="MLR", color=color[1], linewidth=0.9)

    MLR_OnsetsPlus = MLROnsets + 0.05
    MLR_OnsetsMinus = MLROnsets - 0.05

    ax.fill_between(MLR_OnsetsMinus, yNe, step="post", color=color[2], linewidth=0)
    ax.fill_between(MLR_OnsetsPlus, yNe, step="post", color='w', linewidth=0)

    NeuronalOnsets = DF.NeuronalOnset.tolist()
    NeuronalOnsets = np.array([x for x in NeuronalOnsets if not np.isnan(x)])
    NeuronalOnsets = np.sort(NeuronalOnsets)
    yNe = np.hstack((0, np.ones_like(NeuronalOnsets)))
    NeuronalOnsets = np.hstack((0, NeuronalOnsets))
    NeuronalOnsets = np.hstack((NeuronalOnsets, TWOdor[1]))

    yNe = np.cumsum(yNe)
    if yNe[-1] > 0:
        yNe/= yNe[-1]
    yNe = np.hstack((yNe, yNe[-1]))
    ax.step(NeuronalOnsets, yNe, where='post', label="Neuronal", color=color[0],
            linewidth=0.9)


    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy

    # load data
    os.chdir("C:/Users/Cansu/Documents/Ephys_Auswertung/olfactory_visual/FinalFigure")  # go to folder with data
    Files = glob.glob("*.mat")  # open all files in folder
    file_name = "MLR_data.xlsx"  # name for Excel file
    sheet = "Roh"

    # define time windows for df generation
    T0 = 0.09
    TWOdor = [0., 2.0]
    TWBaselOdor = [-20, -0.5]
    TWMLR = [0.0, 2.0]

    OdorCodes = ['J', 'F', 'H', 'K', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
    OdorNames = ['1-Pen', '1-Hex', '1-Hep', '1-Oct', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Con']

    UseSingeTrialBL = True

    BorderLim = 0.97
    MinSpike = 2.25
    TauFR = 250
    WidthFactor = 5
    dt = 1.
    kernel = HelperFunctions.exponential_kernel(TauFR, dt=dt, nstd=WidthFactor)
    AnimalsExclude = ["CA70", "CA69", "CA71"]  # exclude because of too low frame rate


    df = Dataload.GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=T0)

    df.drop(df[[x[0:4] in AnimalsExclude for x in df['AnimalID']]].index,
            inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df_MLR = pd.read_excel(io=file_name, sheet_name=sheet)
    df, _ = Dataload.MergeNeuronal_MLR(df, df_MLR, T0=T0, TWMLR=TWMLR)

    Latencydf = Latency(df, kernel, TWOdor, TWBaselOdor, MinSpike=MinSpike, Border=BorderLim,
                        Single_Trial_BL=UseSingeTrialBL, Stims=['A', 'C', 'G'])

    Latencydf = Latencydf[Latencydf['MLR']==True]
    Odors = ['A', 'C', 'G']
    for Stim in Odors:
        PlotLatencyCDF(Latencydf[Latencydf.StimID == Stim], title=Stim)
        plt.xlim([0., 2.])
        plt.ylim([-0.05, 1.05])
        plt.show()


    PooledLatencydf = Latency_pooled(df, kernel, TWOdor, TWBaselOdor, MinSpike=MinSpike, Border=BorderLim,
                                     Stims=['A', 'C', 'G'])

    print(PooledLatencydf.groupby(['StimID']))
    print(PooledLatencydf.groupby('StimID').agg({'NeuronalOnset': [np.mean, np.min, np.max]}))

    #PooledLatencydf[['RealUnit', 'StimID', 'MLRTime', 'NeuronalOnset']].to_csv(
    #    'PopulationNeuronalOnset.csv')

    CPooled = Dataload.LimitDFtoStimulus(PooledLatencydf, ['C'])
    print(scipy.stats.wilcoxon(CPooled.NeuronalOnset, CPooled.MLRTime))

