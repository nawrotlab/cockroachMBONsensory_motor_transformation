import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys
from Functions import Dataload, HelperFunctions

try:
    import pandarallel
    import multiprocessing
except:
    pass




def PCA_Neuronal(df, kernel, TW, TW_BL, n_components=2, Stims=['A', 'C', 'G'], n_processes=None):
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
    # get animal sets with and without MLR for each stimulus
    ConsideredUnits= Dataload.FindUnits(df, 1, Odorwise=True)

    # limit to found units
    df= Dataload.LimitDFtoUnits(df, ConsideredUnits)
    print('Number of units considered: ', len(ConsideredUnits))

    # estimate firing rates
    df.drop(['SpikeRatesBasel', 'SpikeRatesStim'], axis=1, errors='ignore', inplace=True)
    if n_processes > 1:
        df['SpikeRatesBasel'] = df.parallel_apply(
            lambda x: HelperFunctions.ToSpikeRates(x.BaselSpikeTimes, TW_BL, kernel), axis=1)
        df['SpikeRatesStim'] = df.parallel_apply(
            lambda x: HelperFunctions.ToSpikeRates(x.StimSpikeTimes, TW, kernel), axis=1)
        df['BLRate'] = df.parallel_apply(
            lambda x: (len(x.BaselSpikeTimes) / np.diff(TW_BL))[0], axis=1)
    else:
        df['SpikeRatesBasel'] = df.apply(
            lambda x: HelperFunctions.ToSpikeRates(x.BaselSpikeTimes, TW_BL, kernel), axis=1)
        df['SpikeRatesStim'] = df.apply(
            lambda x: HelperFunctions.ToSpikeRates(x.StimSpikeTimes, TW, kernel), axis=1)
        df['BLRate'] = df.apply(
            lambda x: (len(x.BaselSpikeTimes) / np.diff(TW_BL))[0], axis=1)
    df['SpikeRatesStim'] = df.apply(lambda x: list(np.array(x.SpikeRatesStim) - x.BLRate),
                                                              axis=1)
    df['SpikeRatesBasel'] = df.apply(lambda x: list(np.array(x.SpikeRatesBasel) - x.BLRate),
                                                               axis=1)

    Time= HelperFunctions.ToSpikeRatesTime(df['StimSpikeTimes'].iloc[0], TW, kernel)

    #prepare data for PCA (stacking and averaging)
    df = df.groupby(['RealUnit', 'StimID', 'MLR']).agg(
        {'SpikeRatesStim': lambda y: list(np.vstack(y)), 'MLRTime': lambda y: list(np.vstack(y))})
    df['SpikeRatesStim'] = df['SpikeRatesStim'].apply(lambda x: np.mean(x, axis=0))
    df.reset_index(inplace=True)
    df.sort_values('RealUnit', inplace=True)
    df = df.groupby(['StimID', 'MLR']).agg(
        {'SpikeRatesStim': lambda y: list(np.vstack(y)), 'MLRTime': lambda y: list(np.vstack(y))}).reset_index()
    df['MLRTime'] = df['MLRTime'].apply(lambda x: np.mean(x, axis=0))
    PCA_Data = np.hstack(df['SpikeRatesStim'].iloc[:]).transpose()

    # estimate PCA
    pca = PCA(n_components=n_components)
    pca.fit(PCA_Data)
    # add PCA to dataframe
    df['PCA']=df['SpikeRatesStim'].apply(lambda x: pca.transform(np.array(x).transpose()))
    # drop unnecessary columns
    df.drop(['SpikeRatesStim', 'MLRTime'], axis=1, inplace=True)
    #add time to each row
    #df['Time']=Time
    #add time and explained var to each row - adds unnecessary column but allows to safe everything in one df
    df['Time'] = [Time for i in range(len(df))]
    df['explained_variance_ratio_']=[pca.explained_variance_ratio_ for i in range(len(df))]
    return df


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #load data
    Path = "/mnt/agmn-srv-1/storage/agmn-srv-1_home/carican1/Ephys_Auswertung/olfactory_visual/DataCorr/"  # Pfad zu den Daten
    file_name = "MLR_070121.xlsx"  # name for excel file
    sheet = "Roh"

    # define time windows for df generation
    TWOdor = [-3., 5]
    TWBaselOdor = [-20, -0.6]
    OdorCodes = ['J', 'F', 'H', 'K', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
    OdorNames = ['1-Pen', '1-Hex', '1-Hep', '1-Oct', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Con']
    TWMLR = [0., 2.0]
    BinSize = np.array([0.05])

    #define time windows for PCA
    TW = [-2.6, 4.6]


    BorderLim = 0.95
    MinSpike = 2.25
    TauFR = 200  # defines sigma for gaussian kernel (half of gaussian width)
    WidthFactor = 5
    dt = 1.
    # kernel = spiketools.gaussian_kernel(sigmaFR, dt, nstd=WidthFactor)          #generates kernel for spiketools.kernel_rate
    kernel = HelperFunctions.alpha_kernel(TauFR, dt=dt, nstd=WidthFactor)



    files= Dataload.find_files(Path, pattern="*.mat")
    df = Dataload.GenDF(files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=0.09)
    df['AnimalID'] = df.AnimalID.apply(
        lambda x: x.replace(Path, ""))  # Remove path from AnimalID
    df_MLR = pd.read_excel(io=Path + file_name, sheet_name=sheet)
    df,_ = Dataload.MergeNeuronal_MLR(df, df_MLR, T0=0.09, TWMLR=TWMLR)
    # run PCA

    Transform = PCA_Neuronal(df, kernel, TW, TWBaselOdor, n_components=2, Stims=['A', 'C', 'G'])
    # plot PCA - 2D
    plt.figure()
    # loop over df rows and plot each row
    for i in range(len(Transform)):
        plt.plot(Transform['PCA'].iloc[i][:, 0], Transform['PCA'].iloc[i][:, 1], label=Transform['StimID'].iloc[i] + " " + str(Transform['MLR'].iloc[i]))
    plt.legend()
    plt.show()

