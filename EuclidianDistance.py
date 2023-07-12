import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys
import Dataload
import HelperFunctions
import itertools

try:
    import pandarallel
    import multiprocessing
except:
    pass




def Euclidian_MLR(df, kernel, TW, TW_BL, normalize=True, Norm='L2', MinResponses=2, Stims=['A', 'C', 'G'], n_processes=None):
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
    ConsideredUnits=Dataload.FindUnits(df, MinResponses, Odorwise=False)

    # limit to found units
    df=Dataload.LimitDFtoUnits(df, ConsideredUnits)
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

    Time=HelperFunctions.ToSpikeRatesTime(df['StimSpikeTimes'].iloc[0], TW, kernel)
    df['MaxStim'] = df['SpikeRatesStim'].apply(np.max)
    FRsNorm = df.groupby(['RealUnit']).agg({'MaxStim': np.max})
    LenghtRates = np.min(df['SpikeRatesStim'].apply(lambda x: len(x)))
    Distance = np.zeros((LenghtRates, 1))



    df = df.groupby(['RealUnit', 'MLR']).agg({'SpikeRatesStim': lambda y: list(np.vstack(y))}).reset_index()
    df['SpikeRatesStim'] = df.apply(lambda x: np.array(x['SpikeRatesStim']).mean(axis=0), axis=1)

    for idUnit, Unit in enumerate(ConsideredUnits):

        if normalize:
            BooleanID = (df['RealUnit'] == Unit) & (df['MLR'] == True)
            MLR = np.array(df[BooleanID]['SpikeRatesStim'].iloc[0]) / FRsNorm.loc[Unit, 'MaxStim']
            BooleanID = (df['RealUnit'] == Unit) & (df['MLR'] == False)
            NoMLR = np.array(df[BooleanID]['SpikeRatesStim'].iloc[0]) / FRsNorm.loc[Unit, 'MaxStim']
        else:
            BooleanID = (df['RealUnit'] == Unit) & (df['MLR'] == True)
            MLR = np.array(df[BooleanID]['SpikeRatesStim'].iloc[0])
            BooleanID = (df['RealUnit'] == Unit) & (df['MLR'] == False)
            NoMLR = np.array(df[BooleanID]['SpikeRatesStim'].iloc[0])

        if Norm == 'L1':
            Distance[:, 0] += np.abs(MLR - NoMLR)
        elif Norm == 'L2':
            Distance[:, 0] += np.square(MLR - NoMLR)
        else:
            Distance[:, 0] += np.square(MLR - NoMLR)
            #print only once
            if idUnit == 0:
                print('No valid norm specified, using L2 norm')
    if ~(Norm == 'L1'):
        Distance[:, 0] = np.sqrt(Distance[:, 0])

    return np.squeeze(Time), np.squeeze(Distance), FRsNorm


def Euclidian_Odors(df, kernel, TW, TW_BL, MLR=False, normalize=True, Norm='L2', FRsNorm=None, MinResponses=2, Stims=['A', 'C', 'G'], n_processes=None):
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
    ConsideredUnits=Dataload.FindUnits(df, MinResponses, Odorwise=False)

    # limit to found units
    df=Dataload.LimitDFtoUnits(df, ConsideredUnits)
    print('Number of units considered: ', len(ConsideredUnits))

    if MLR:
        df = df[df['MLR']].reset_index(drop=True)
    else:
        df = df[~df['MLR']].reset_index(drop=True)


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
    Time=HelperFunctions.ToSpikeRatesTime(df['StimSpikeTimes'].iloc[0], TW, kernel)



    ## Compare Euclidian DIstance between odors with given MLR response

    # normalize firing rates
    if FRsNorm is None:
        df['MaxStim'] = df['SpikeRatesStim'].apply(np.max)
        FRsNorm = df.groupby(['RealUnit']).agg({'MaxStim': np.max})
    LenghtRates = np.min(df['SpikeRatesStim'].apply(lambda x: len(x)))

    # calculate the number of possible comparisons
    comparisons = list(itertools.combinations(Stims, 2))
    n_comparisons = len(comparisons)
    Distance = np.zeros((LenghtRates, n_comparisons))

    DFs=[Dataload.LimitDFtoStimulus(df, Stim).copy(deep=True) for Stim in Stims]
    RemainingUnits=[Dataload.FindUnits2(DF, MinResponses, MLR=MLR) for DF in DFs]

    # loop over all possible comparisons
    for idComp, comp in enumerate(comparisons):
        #get the units that are present in both stims
        UnitOverlap = set.intersection(set(RemainingUnits[Stims.index(comp[0])]), set(RemainingUnits[Stims.index(comp[1])]))
        # limit the dataframe to the units that are present in both stims
        DF_comp = Dataload.LimitDFtoUnits(df, UnitOverlap)
        DF_comp = Dataload.LimitDFtoStimulus(DF_comp, comp)

        DF_comp= DF_comp.groupby(['RealUnit', 'StimID']).agg({'SpikeRatesStim': lambda y: list(np.vstack(y))}).reset_index()
        DF_comp['SpikeRatesStim'] = DF_comp.apply(lambda x: np.array(x['SpikeRatesStim']).mean(axis=0), axis=1)

        for idUnit, Unit in enumerate(UnitOverlap):

            if normalize:
                BooleanID = (DF_comp['RealUnit'] == Unit) & (DF_comp['StimID'] == comp[0])
                O1 = np.array(DF_comp[BooleanID]['SpikeRatesStim'].iloc[0]) / FRsNorm.loc[Unit, 'MaxStim']
                BooleanID = (DF_comp['RealUnit'] == Unit) & (DF_comp['StimID'] == comp[1])
                O2 = np.array(DF_comp[BooleanID]['SpikeRatesStim'].iloc[0]) / FRsNorm.loc[Unit, 'MaxStim']
            else:
                BooleanID = (DF_comp['RealUnit'] == Unit) & (DF_comp['StimID'] == comp[0])
                O1 = np.array(DF_comp[BooleanID]['SpikeRatesStim'].iloc[0])
                BooleanID = (DF_comp['RealUnit'] == Unit) & (DF_comp['StimID'] == comp[1])
                O2 = np.array(DF_comp[BooleanID]['SpikeRatesStim'].iloc[0])

            if Norm == 'L1':
                Distance[:, idComp] += np.abs(O1 - O2)
            elif Norm == 'L2':
                Distance[:, idComp] += np.square(O1 - O2)
            else:
                Distance[:, idComp] += np.square(O1 - O2)
                # print only once
                if idUnit == 0:
                    print('No valid norm specified, using L2 norm')
        if ~(Norm == 'L1'):
            Distance[:, idComp] = np.sqrt(Distance[:, idComp])

    # generate dict to store the results in a dataframe
    Results = [{'Stim1': comp[0], 'Stim2': comp[1], 'Time': Time, 'Distance': np.squeeze(Distance[:, idComp])}
               for idComp, comp in enumerate(comparisons)]
    Results = pd.DataFrame(Results)
    return Results



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
    TauFR = 60  # defines sigma for gaussian kernel (half of gaussian width)
    WidthFactor = 3
    dt = 1.
    # kernel = spiketools.gaussian_kernel(sigmaFR, dt, nstd=WidthFactor)          #generates kernel for spiketools.kernel_rate
    kernel = HelperFunctions.alpha_kernel(TauFR, dt=dt, nstd=WidthFactor)



    files=Dataload.find_files(Path, pattern="*.mat")
    df = Dataload.GenDF(files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=0.09)
    df['AnimalID'] = df.AnimalID.apply(
        lambda x: x.replace(Path, ""))  # Remove path from AnimalID
    df_MLR = pd.read_excel(io=Path + file_name, sheet_name=sheet)
    df,_ = Dataload.MergeNeuronal_MLR(df, df_MLR, T0=0.09, TWMLR=TWMLR)

    # Euclidian distance

    # plot a

    Time, Distance, FRsNorm = Euclidian_MLR(df, kernel, TW, TWBaselOdor, normalize=True, Norm='L2', MinResponses=2, Stims=['A', 'C', 'G'])
    # plot PCA - 2D
    plt.figure()
    # loop over df rows and plot each row
    plt.plot(Time, Distance, label='MLR')
    plt.xlabel('Time [ms]')
    plt.ylabel('Distance')
    plt.xlim([-500, 2000])
    plt.ylim([0, 1.05])
    plt.axhline(np.mean(Distance[Time<-500]), color='k', linestyle='--', label='Mean')

    plt.legend()
    plt.show()

    # figure for odor comparison

    odorCompare=Euclidian_Odors(df, kernel, TW, TWBaselOdor, FRsNorm=FRsNorm, normalize=True, Norm='L2', MinResponses=2, Stims=['A', 'C', 'G'], MLR=False)

    plt.figure()
    # loop over df rows and plot each row and plot distance with label Stim1-Stim2
    for idx,row in odorCompare.iterrows():
        plt.plot(row['Time'], row['Distance'], label=row['Stim1']+'-'+row['Stim2'])
    plt.xlabel('Time [ms]')
    plt.ylabel('Distance')
    plt.xlim([-500, 2000])
    plt.ylim([0, 1.05])
    plt.legend()
    plt.show()