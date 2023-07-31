import numpy as np
import pandas as pd
import sys
from Functions import Dataload, HelperFunctions
import random
import os

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pickle


try:
    import pandarallel
    import multiprocessing
    from pathos.multiprocessing import ProcessPool
except:
    pass

def CreateSamples(df, samples, Units):
    NUnits=len(Units)
    Data=np.zeros((samples,NUnits))
    Result=np.zeros(samples)
    for ii in range(samples):
        Result[ii]= np.random.rand() < .5
        for idx, RealUnit in enumerate(Units):
            DataU=df[(df['MLR']==Result[ii])&(df['RealUnit']==RealUnit)]
            Data[ii, idx]=random.choice(np.array(DataU['Count']))
    return Data, Result

def CreateOdorTestSet(df, Odor, Units, samples=20):
    df=df[df['StimID']==Odor].reset_index(drop=True)
    return CreateSamples(df, samples, Units)

def TrainTestSplitDF_MLR(df, test_size=0.33):
    Classifier_DF_Training = pd.DataFrame()
    Classifier_DF_Test = pd.DataFrame()
    for counter,(name, group) in enumerate(df.groupby(['RealUnit', 'MLR'])):
        DF_train, DF_test = train_test_split(group, test_size=test_size)
        Classifier_DF_Training=pd.concat([Classifier_DF_Training, DF_train.copy(deep=True)], ignore_index=True)
        Classifier_DF_Test=pd.concat([Classifier_DF_Test, DF_test.copy(deep=True)], ignore_index=True)
    return Classifier_DF_Training, Classifier_DF_Test


def TrainTestSplitDF_MLR2(df, test_size=0.33):          #like TrainTestSplitDF_MLR, but with mechanism to cope with single samples
    Classifier_DF_Training = pd.DataFrame()
    Classifier_DF_Test = pd.DataFrame()
    for counter,(name, group) in enumerate(df.groupby(['RealUnit', 'MLR'])):
        if group.shape[0] >1:
            DF_train, DF_test = train_test_split(group, test_size=test_size)
            Classifier_DF_Training=pd.concat([Classifier_DF_Training, DF_train.copy(deep=True)], ignore_index=True)
            Classifier_DF_Test=pd.concat([Classifier_DF_Test, DF_test.copy(deep=True)], ignore_index=True)
        elif group.shape[0] ==0:
            pass
        elif group.shape[0] ==1:
            if np.random.rand() < .5:
                Classifier_DF_Training=pd.concat([Classifier_DF_Test, group.copy(deep=True)], ignore_index=True)
            else:
                Classifier_DF_Test=pd.concat([Classifier_DF_Test, group.copy(deep=True)], ignore_index=True)
    return Classifier_DF_Training, Classifier_DF_Test


def ClassifierTrial(DataFrameCleaned, RemainingUnits, time, LenghtRates, samples, samplestest):
    pid = os.getpid()
    NUnits = len(RemainingUnits)
    Accuracy = np.zeros((LenghtRates))
    Classifier_DF = DataFrameCleaned.copy(deep=True).reset_index(drop=True)
    SubsetUnits = np.random.choice(list(RemainingUnits), NUnits, replace=False)
    Classifier_DF = Dataload.LimitDFtoUnits(Classifier_DF, SubsetUnits).copy(deep=True)
    Classifier_DF_Training, Classifier_DF_Test = TrainTestSplitDF_MLR(Classifier_DF)

    for idT, timepoint in enumerate(time):
        Classifier_DF_Training['Count'] = Classifier_DF_Training['SpikeRatesStim'].apply(lambda y: y[idT])
        Classifier_DF_Test['Count'] = Classifier_DF_Test['SpikeRatesStim'].apply(lambda y: y[idT])

        ## Create Train data and fit classifier
        Data, Result = CreateSamples(Classifier_DF_Training, samples, SubsetUnits)
        scaler = preprocessing.StandardScaler().fit(Data)
        X_scaled = scaler.transform(Data)
        clf = LogisticRegression(random_state=0).fit(X_scaled, Result)

        ## Create Test data
        Data, Result = CreateSamples(Classifier_DF_Test, samplestest, SubsetUnits)
        X_scaled = scaler.transform(Data)
        Accuracy[idT] += clf.score(X_scaled, Result)
        if (idT % 20) == 0:
            print("Worker " + str(pid) + ":" + str(100 * idT / LenghtRates) + "%")
    return Accuracy
def Classifier(df, kernel, TW, TW_BL, shuffle=False, Odorwise=False, MinResponses=10, trials=24, samples=50, samplestest=20, force_restimate=False, Stims=['A', 'C', 'G'],nt=1 , n_processes=None):
    if shuffle == True:
        Filename = os.path.join('Functions', 'ClassifierResultsShuffled.pkl')
    else:
        Filename = os.path.join('Functions', 'ClassifierResults.pkl')
    if force_restimate == False:
        try:
            with open(Filename, 'rb') as f:
                ClassifierResults = pickle.load(f)
                # compare parameters to see if they are the same as in the dictionary
                if ClassifierResults['Parameters']['TW'] == TW and ClassifierResults['Parameters']['TW_BL'] == TW_BL and \
                        ClassifierResults['Parameters']['shuffle'] == shuffle and ClassifierResults['Parameters'][
                    'Odorwise'] == Odorwise and ClassifierResults['Parameters']['MinResponses'] == MinResponses and \
                        ClassifierResults['Parameters']['trials'] == trials and ClassifierResults['Parameters'][
                    'samples'] == samples and ClassifierResults['Parameters']['samplestest'] == samplestest and \
                        ClassifierResults['Parameters']['Stims'] == Stims and ClassifierResults['Parameters']['nt'] == nt:
                    print('Previous results found. Returning them.')
                    return ClassifierResults['Returnvalue']
                else:
                    print('Parameters do not match. Calculating new results.')
        except:
            print('No previous results found. Calculating new ones.')

    # create parameter dictionary
    Parameters = {'TW': TW, 'TW_BL': TW_BL, 'shuffle': shuffle, 'Odorwise': Odorwise, 'MinResponses': MinResponses,
                  'trials': trials, 'samples': samples, 'samplestest': samplestest, 'Stims': Stims, 'nt': nt}
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
    ConsideredUnits= Dataload.FindUnits(df, MinResponses, Odorwise=Odorwise)

    # limit to found units
    df= Dataload.LimitDFtoUnits(df, ConsideredUnits)
    print('Number of units considered: ', len(ConsideredUnits))

    # estimate firing rates
    df.drop(['SpikeRatesBasel', 'SpikeRatesStim'], axis=1, errors='ignore', inplace=True)
    if n_processes > 1:
        df['SpikeRatesStim'] = df.parallel_apply(
            lambda x: HelperFunctions.ToSpikeRates(x.StimSpikeTimes, TW, kernel), axis=1)
    else:
        df['SpikeRatesStim'] = df.apply(
            lambda x: HelperFunctions.ToSpikeRates(x.StimSpikeTimes, TW, kernel), axis=1)

    df['SpikeRatesStim'] = df.apply(
        lambda x: list(np.array(x.SpikeRatesStim) - HelperFunctions.ToFiringRates(x.BaselSpikeTimes, TW_BL)), axis=1)


    Time= HelperFunctions.ToSpikeRatesTime(df['StimSpikeTimes'].iloc[0], TW, kernel)

    if nt > 1:
        # get every nth timepoint and redurce estimated firing rates accordingly
        Time = Time[::nt]
        df['SpikeRatesStim'] = df['SpikeRatesStim'].apply(lambda x: x[::nt])


    if shuffle:
        df = Dataload.ShuffleMLRs(df).copy(deep=True)

    LenghtRates = np.min(df['SpikeRatesStim'].apply(lambda x: len(x)))
    Time = Time[:LenghtRates]
    Accuracy = np.zeros((LenghtRates, trials))

    iterations = [df.copy(deep=True) for iterate in range(trials)]

    if n_processes > 1:
        with ProcessPool(nodes=n_processes) as p:

            Result = p.map(lambda x: ClassifierTrial(x, ConsideredUnits, Time, LenghtRates, samples, samplestest), iterations)
    else:
        Result = map(lambda x: ClassifierTrial(x, ConsideredUnits, Time, LenghtRates, samples, samplestest), iterations)
    for ii, res in enumerate(Result):
        Accuracy[:, ii] = res

    SaveDict = {'Parameters': Parameters, 'Returnvalue': [Accuracy, Time, ConsideredUnits]}
    with open(Filename, 'wb') as f:
        pickle.dump(SaveDict, f)

    return SaveDict['Returnvalue']


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #load data
    Path = ".."+os.path.sep+"Data"+os.path.sep  # Pfad zu den Daten
    file_name = "MLR_data.xlsx"  # name for excel file
    sheet = "Roh"

    # define time windows for df generation
    TWOdor = [-0.9, 2.5]
    TWBaselOdor = [-20, -0.5]
    OdorCodes = ['J', 'F', 'H', 'K', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
    OdorNames = ['1-Pen', '1-Hex', '1-Hep', '1-Oct', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Con']
    TWMLR = [0., 2.0]

    InterestingOdors = ['C', 'G', 'A']

    TauFR = 50  # defines sigma for gaussian kernel (half of gaussian width)
    WidthFactor = 6
    dt = 1.
    # kernel = spiketools.gaussian_kernel(sigmaFR, dt, nstd=WidthFactor)          #generates kernel for spiketools.kernel_rate
    kernel = HelperFunctions.alpha_kernel(TauFR, dt=dt, nstd=WidthFactor, calibrateMass=False)



    files= Dataload.find_files(Path, pattern="*.mat")
    df = Dataload.GenDF(files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=0.09)
    df_MLR = pd.read_excel(io=Path + file_name, sheet_name=sheet)
    df,_ = Dataload.MergeNeuronal_MLR(df, df_MLR, T0=0.09, TWMLR=TWMLR)
    # run classifier without shuffling
    #Accuracy, Time, ConsideredUnits = Classifier(df, kernel, TWOdor, TWBaselOdor, shuffle=False, Odorwise=False,
    #                                             MinResponses=10, trials=24, samples=50, samplestest=20,
    #                                             force_restimate=False,Stims=InterestingOdors)

    # faster version wo shuffling
    Accuracy, Time, ConsideredUnits = Classifier(df, kernel, TWOdor, TWBaselOdor, shuffle=False, Odorwise=False,
                                                 MinResponses=10, trials=24, samples=50, samplestest=20,
                                                 force_restimate=False,Stims=InterestingOdors, nt=1)

    print(ConsideredUnits)

    Accuracy_Shuff, Time_Shuff, ConsideredUnits = Classifier(df, kernel, TWOdor, TWBaselOdor, shuffle=True, Odorwise=False,
                                                 MinResponses=10, trials=24, samples=50, samplestest=20,
                                                 force_restimate=False, Stims=InterestingOdors, nt=1)

    print(ConsideredUnits)

    plt.figure()
    # loop over df rows and plot each row
    plt.plot(np.array(Time)/1000, np.mean(Accuracy, axis=1), label='Mean')
    plt.plot(np.array(Time_Shuff)/1000, np.mean(Accuracy_Shuff, axis=1), label='Shuffled')
    plt.axhline(0.5, color='k', linestyle='--')
    plt.xlim([0, 2.0])
    plt.ylim([0.1, 1.0])
    plt.legend()
    plt.show()

