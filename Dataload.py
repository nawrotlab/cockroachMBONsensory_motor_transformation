import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd                #to use dataframes
from scipy.io import loadmat       #to load .mat files
import os                          #to change path
import re                          #to use regular expressions
from sklearn.utils import shuffle  #to shuffle data

def GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset = 0,LightCodes = 'NaN',LightNames ='NaN' ,LightPulsesN ='NaN',TWLight='NaN', TWBaselLight='NaN'):


    """ takes list of file names and generates one dataframe of all files
         Files : list with all file names; be in the right folder!
         TWOdor : TimeWindow of odor stimulation
         TWBaselOdor : TimeWindow of baseline before odor stimulation
         OdorCodes : Codes of odor stimuli in recording
         OdorNames : Translation of the odor code from the Recording to a given name
         CorrectOdorOnset : shift of spiketimes during olfactory stimulation in seconds
         LightCodes : Codes of light stimuli in recording
         LightNames : Translation of the light code from the Recording to a given name
         LightPulsesN : Number of light pulses per trial
         TWLight : TimeWindow of light stimulation
         TWBaselLight : TimeWindow of baseline before light stimulation
        """

    DataFrame = pd.DataFrame(
        columns=['AnimalID', 'StimID', 'StimName', 'RealUnit', 'UnitID', 'Trial', 'RecOnTime', 'RecOffTime', 'OffTime',
                 'StimSpikeTimes', 'BaselSpikeTimes'])  # create empty pandas dataframe

    Time = np.array([])
    Units = np.array([])
    CurrentUnit = 0
    MaxUnit = 0

    for file in Files:
        data = loadmat(file)
        for key in data.keys():  # loop to find Channel 5, 31 & 32
            if (key.find('Ch5') > -1):
                ch5key = key
            if (key.find('Ch31') > -1):
                ch31key = key
            if (key.find('Ch32') > -1):
                ch32key = key

        Time = np.squeeze(np.array(data[ch5key][0][0][11]))  # merge timespoints from all files in one array
        Units = np.array(data[ch5key][0][0][12])[:, 0]  # merge units from all files in one array
        StimTime = np.array(data[ch31key][0][0][4])  # put all Stimulus Times in one array dependend on Stimulus Code
        StimCode = np.array(data[ch31key][0][0][5])[:, 0]  # put all Stimulus Codes in one array
        StimOnOffTime = np.array(data[ch32key][0][0][4])  # put all On/Off Times for each stimulation in one array
        StimOnOff = np.array(data[ch32key][0][0][5])[:, 0]  # put all On/Off Codes for each stimulation in one array
        OnTimes = StimOnOffTime[StimOnOff == 1]  # extract ontimes in one array
        OffTimes = StimOnOffTime[StimOnOff == 0]  # extract offtimes in one array
        UniqueUnits = np.unique(Units)  # generates array with all Units
        UniqueStim = np.unique(StimCode)  # generates array with all Stimuli
        UniqueStim = np.delete(UniqueStim, UniqueStim == ord('L'))  # deletes "L" from Stimuli array
        UniqueStim = np.delete(UniqueStim, UniqueStim == ord('Q'))  # deletes "Q" from Stimuli array

        for Stimulus in UniqueStim:  # loops over all Stimuli once
            TimeStim = StimTime[StimCode == Stimulus]  # Stimulation Time for each stimuli

            for UnitNumber, Unit in enumerate(UniqueUnits):  # loops through Units
                UnitTime = Time[Units == Unit]

                if LightCodes == 'NaN':
                    if chr(Stimulus) in OdorCodes:
                        OdorName = OdorNames[OdorCodes.index(chr(Stimulus))]

                        for TrialNumber, TrialTimepoint in enumerate(TimeStim):
                            RecOnTime = np.nanmin(OnTimes[TrialTimepoint < OnTimes])
                            StimDur = np.nanmin(OffTimes[TrialTimepoint < OffTimes]) - RecOnTime
                            RecOffTime = np.nanmin(OffTimes[TrialTimepoint < OffTimes])

                            CutStim = UnitTime[np.logical_and(((RecOnTime + TWOdor[0]) < UnitTime),
                                                              ((RecOnTime + TWOdor[1]) > UnitTime))] - (
                                                  RecOnTime + CorrectOdorOnset)
                            CutBasel = UnitTime[np.logical_and(((RecOnTime + TWBaselOdor[0]) < UnitTime),
                                                               ((RecOnTime + TWBaselOdor[1]) > UnitTime))] - RecOnTime

                            DataFrame = DataFrame.append({'AnimalID': os.path.splitext(file)[0], 'UnitID': CurrentUnit,
                                                          'RealUnit': MaxUnit + UnitNumber, 'Trial': TrialNumber,
                                                          'StimID': chr(Stimulus), 'StimName': OdorName, 'OffTime': StimDur,
                                                          'StimSpikeTimes': CutStim, 'BaselSpikeTimes': CutBasel,
                                                          'RecOnTime': RecOnTime, 'RecOffTime': RecOffTime},
                                                         ignore_index=True)

                        try:
                            CurrentUnit = np.nanmax(DataFrame['UnitID']) + 1

                        except:
                            CurrentUnit = CurrentUnit

                    else:
                        pass

                else:

                    if chr(Stimulus) in LightCodes:

                        TimeStimV = TimeStim[::LightPulsesN]  # first Stimulation Time of all pulses in one trial
                        LightName = LightNames[LightCodes.index(chr(Stimulus))]

                        for TrialNumber, TrialTimepoint in enumerate(TimeStimV):
                            RecOnTime = np.nanmin(OnTimes[TrialTimepoint < OnTimes])
                            RecOnTimeI = np.where(OnTimes == RecOnTime)
                            RecOnTimesV = OnTimes[int(RecOnTimeI[0]):int(RecOnTimeI[0]) + LightPulsesN]

                            StimDur = np.nanmin(OffTimes[TrialTimepoint < OffTimes]) - RecOnTime
                            RecOffTime = np.nanmin(OffTimes[TrialTimepoint < OffTimes])
                            RecOffTimeI = np.where(OffTimes == RecOffTime)
                            RecOffTimesV = OffTimes[int(RecOffTimeI[0]):int(RecOffTimeI[0]) + LightPulsesN]

                            CutStim = UnitTime[np.logical_and(((RecOnTime + TWLight[0]) < UnitTime),
                                                              ((RecOnTime + TWLight[1]) > UnitTime))] - RecOnTime
                            CutBasel = UnitTime[np.logical_and(((RecOnTime + TWBaselLight[0]) < UnitTime),
                                                               ((RecOnTime + TWBaselLight[1]) > UnitTime))] - RecOnTime

                            DataFrame = DataFrame.append({'AnimalID': os.path.splitext(file)[0], 'UnitID': CurrentUnit,
                                                          'RealUnit': MaxUnit + UnitNumber, 'Trial': TrialNumber,
                                                          'StimID': chr(Stimulus), 'StimName': LightName,
                                                          'OffTime': StimDur, 'StimSpikeTimes': CutStim,
                                                          'BaselSpikeTimes': CutBasel, 'RecOnTime': RecOnTimesV,
                                                          'RecOffTime': RecOffTimesV}, ignore_index=True)

                        try:
                            CurrentUnit = np.nanmax(DataFrame['UnitID']) + 1

                        except:
                            CurrentUnit = CurrentUnit

                    else:
                        OdorName = OdorNames[OdorCodes.index(chr(Stimulus))]

                        for TrialNumber, TrialTimepoint in enumerate(TimeStim):
                            RecOnTime = np.nanmin(OnTimes[TrialTimepoint < OnTimes])
                            StimDur = np.nanmin(OffTimes[TrialTimepoint < OffTimes]) - RecOnTime
                            RecOffTime = np.nanmin(OffTimes[TrialTimepoint < OffTimes])

                            CutStim = UnitTime[np.logical_and(((RecOnTime + TWOdor[0]) < UnitTime),
                                                              ((RecOnTime + TWOdor[1]) > UnitTime))] - (RecOnTime + CorrectOdorOnset)
                            CutBasel = UnitTime[np.logical_and(((RecOnTime + TWBaselOdor[0]) < UnitTime),
                                                               ((RecOnTime + TWBaselOdor[1]) > UnitTime))] - RecOnTime

                            DataFrame = DataFrame.append({'AnimalID': os.path.splitext(file)[0], 'UnitID': CurrentUnit,
                                                          'RealUnit': MaxUnit + UnitNumber, 'Trial': TrialNumber,
                                                          'StimID': chr(Stimulus), 'StimName': OdorName, 'OffTime': StimDur,
                                                          'StimSpikeTimes': CutStim, 'BaselSpikeTimes': CutBasel,
                                                          'RecOnTime': RecOnTime, 'RecOffTime': RecOffTime},
                                                           ignore_index=True)

                        try:
                            CurrentUnit = np.nanmax(DataFrame['UnitID']) + 1

                        except:
                            CurrentUnit = CurrentUnit


        MaxUnit = MaxUnit + UnitNumber + 1
    if CorrectOdorOnset==False:
        print('false')

    # remove path from AnimalID - only works if all files are in the same folder
    Path = os.path.dirname(file) + '/'
    DataFrame['AnimalID'] = DataFrame.AnimalID.apply(
        lambda x: x.replace(Path, ""))  # Remove path from AnimalID

    return DataFrame


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def MergeNeuronal_MLR(df, DataFrameMLR, TWMLR=[0., 2.], T0=0.):
    '''
    Merge neuronal data with MLR data and returns the dataframe for the intersection of animals of MLR and neuronal data
    :param df: neuronal data frame
    :param DataFrameMLR: MLR data frame
    :return: df: merged MLR and neuronal data frame
    '''
    df=df.copy()
    df["A_ID:short"] = df.AnimalID.str.slice(stop=4)

    RecAnimalID = np.array(df["A_ID:short"])
    MLRAnimalSet = set(np.unique(DataFrameMLR['Animal-ID']))  # gemerate set with MLR Animal
    MLRAnimalSet.intersection_update(set(np.unique(RecAnimalID)))  # keep only animals that are in both sets
    df = df[df['A_ID:short'].isin(MLRAnimalSet)]
    DataFrameMLR = DataFrameMLR[DataFrameMLR['Animal-ID'].isin(MLRAnimalSet)]
    df.reset_index(drop=True, inplace=True)
    DataFrameMLR.reset_index(drop=True, inplace=True)
    DataFrameMLR[["Trial", "MLRTime"]] = DataFrameMLR.apply(lambda x: CalculateTrial(x, df), axis=1,
                                                            result_type="expand")
    DataFrameMLR['MLRTime'] = DataFrameMLR['MLRTime'] - T0
    DataFrameMLR.rename(columns={"Stimulus-ID": "StimID", "Animal-ID": "A_ID:short"}, inplace=True)
    DataFrameMLR = DataFrameMLR[['A_ID:short', "StimID", 'Trial', 'MLRTime']]
    # DataFrameMLR["Trial"] = DataFrameMLR['Trial'].astype(int)
    DataFrameMLR.MLRTime = DataFrameMLR.apply(
        lambda x: x.MLRTime if (x.MLRTime >= TWMLR[0] and x.MLRTime <= TWMLR[1]) else np.nan, axis=1)
    df = df.drop(['MLRTime', 'MLRTime_x', 'MLRTime_y', 'MLRTime_z'], axis=1, errors='ignore')
    df = df.merge(DataFrameMLR, on=["A_ID:short", "StimID", 'Trial'])
    df.drop(["A_ID:short"], inplace=True, axis=1)
    df['MLR'] = ~df['MLRTime'].isnull()
    return df, MLRAnimalSet

def GenDFCorr(DataFrame, DataFrameMLR,OdorCodes, TWMLR):

    """ generates a dataframe by combining DataFrame with spiketimes and Dataframe with behavioral data 
             
            """

    DataFrameCorr = pd.DataFrame(columns=['AnimalID', 'OdorID', 'OdorNames', 'UnitID', 'RealUnit', 'Trial', 'MLR',
                                          'MLRTime'])  # generate empty dataframe for correlation
    RecAnimalID = np.array(DataFrame['AnimalID'].str.slice(stop=4))
    MLRAnimalSet = set(np.unique(DataFrameMLR['Animal-ID']))  # gemerate set with MLR Animal
    RecAnimalSet = set(np.unique(RecAnimalID))  # generate set with recorded Animal
    MLRAnimalSet.intersection_update(RecAnimalSet)  # keep only animals that are in both sets
    MLRAnimalSet = list(MLRAnimalSet)
    MLRAnimalSet.sort(key=natural_keys)

    for Odor in OdorCodes:  # loops through OdorSet
        DfMLROdor = DataFrameMLR[DataFrameMLR['Stimulus-ID'] == Odor]  # generates DF for each odor from MLR DF
        DfOdor = DataFrame[DataFrame['StimID'] == Odor]  # generates DF for each odor drom recording DF
        OdorName = np.unique(DfOdor['StimName'])
        for Animal in MLRAnimalSet:  # loops though AnimalIDs that exist in recordings and MLR data
            DfMLROdorAnimal = DfMLROdor[DfMLROdor[
                                            'Animal-ID'] == Animal]  # generates Dataframe: AnimalID from recordings should match MLR AnimalID
            MLRCol = DfMLROdorAnimal.loc[:, 'MLR']
            MLR = MLRCol.values
            MLRTimeCol = DfMLROdorAnimal.loc[:, 'MLR-Time-[sec]']
            MLRTimes = MLRTimeCol.values
            DfOdorAnimal = DfOdor[DfOdor['AnimalID'].str.slice(stop=4) == Animal]
            UnitSet = np.unique(DfOdorAnimal['UnitID'])
            for Unit in UnitSet:
                DfOdorAnimalUnit = DfOdorAnimal[DfOdorAnimal['UnitID'] == Unit]
                TrialSet = np.unique(DfOdorAnimal['Trial'])
                RUnit = np.unique(DfOdorAnimalUnit['RealUnit'])
                for index, Trial in enumerate(TrialSet):
                    UniqueMLR = MLR[index]
                    MLRTime = MLRTimes[index]
                    UniqueRecOnTime = DfOdorAnimalUnit.iloc[index]['RecOnTime']

                    DataFrameCorr = DataFrameCorr.append(
                        {'AnimalID': (Animal), 'UnitID': (Unit), 'RealUnit': RUnit[0], 'Trial': Trial, 'OdorID': (Odor),
                         'OdorNames': OdorName, 'MLR': (UniqueMLR), 'MLRTime': (MLRTime - (UniqueRecOnTime + TWMLR[0])),
                         'MLRTimeAll': (MLRTime - UniqueRecOnTime)}, ignore_index=True)

    for index, i in enumerate(DataFrameCorr['MLRTime']):
        if i < TWMLR[0] or i > TWMLR[1]:
            DataFrameCorr.MLR[index] = 0
            DataFrameCorr.MLRTime[index] = float("NaN")

    return DataFrameCorr, MLRAnimalSet


def find_files(path, pattern="*.mat"):
    '''
    Find files in a directory
    :param path:
    :param pattern:
    :return:
    '''
    files=glob.glob(path + pattern)
    files.sort()
    return files


def CalculateTrial(row, pDF):
    '''
    Calculate trial number for a given row, considering the time of the row (RecOnTime)
    :param row: row of a dataframe - not index
    :param pDF: Dataframe to look into
    :return: (Trial, MLRTime (relative to Trial start)
    '''
    idxDF=np.where((row['Animal-ID'] == pDF['A_ID:short']) & (pDF['StimID'] == row['Stimulus-ID']) & (pDF['RecOnTime'] >= (row['Time [sec]']-0.5)) & (pDF['RecOnTime'] <= (row["Time [sec]"]+1.5)))
    if len(idxDF[0]):
        idxDF=idxDF[0][0]
        MLRTimeToWrite=row['MLR-Time-[sec]']-pDF.loc[idxDF, 'RecOnTime']
        return (int(pDF.loc[idxDF, 'Trial']), MLRTimeToWrite)
    else:
        return (np.nan, np.nan)

def LimitDFtoStimulus(df, odors):
    '''
    Filters dataframe for the given odors
    :param df: Dataframe to filter
    :param odors: List of odors to keep
    :return: Filtered dataframe
    '''
    DataFrameCleaned=df.copy(deep=True).reset_index(drop=True)
    boolIndex=[x in odors for x in DataFrameCleaned.StimID]
    DataFrameCleaned=DataFrameCleaned[boolIndex].reset_index(drop=True)
    return DataFrameCleaned

def LimitDFtoUnits(df, Units):
    '''
    Filters dataframe for the given units
    :param df: Dataframe to filter
    :param Units: List of units to keep
    :return: Filtered dataframe
    '''
    df=df.copy(deep=True).reset_index(drop=True)
    boolIndex=[x in Units for x in df.RealUnit]
    df=df[boolIndex].reset_index(drop=True)
    return df


def FindUnits(df, MinResponse, Odorwise=False):
    '''
    Finds units that responds at least MinResponse times for min(count(MLR), count(~MLR))
    :param df: Dataframe to look into
    :param MinResponse: Minimal number of responses
    :param Odorwise: If True, looks for units that respond to each odor separately with MinResponses
    :return: List of sets of units that fulfill the criteria
    '''
    Sets=[]
    if not Odorwise:
        DF_Response=df.groupby('RealUnit').agg({'MLR': np.sum}).reset_index()
        Sets.append(set(DF_Response[DF_Response['MLR']>=MinResponse]['RealUnit']))
        DF_Response=df.groupby('RealUnit').agg({'MLR': lambda x: np.sum(~x)}).reset_index()
        Sets.append(set(DF_Response[DF_Response['MLR']>=MinResponse]['RealUnit']))
    else:
        for odor in np.unique(df['StimID']):
            df_loc=LimitDFtoStimulus(df, [odor])
            DF_Response=df_loc.groupby('RealUnit').agg({'MLR': np.sum}).reset_index()
            Sets.append(set(DF_Response[DF_Response['MLR']>=MinResponse]['RealUnit']))
            DF_Response=df_loc.groupby('RealUnit').agg({'MLR': lambda x: np.sum(~x)}).reset_index()
            Sets.append(set(DF_Response[DF_Response['MLR']>=MinResponse]['RealUnit']))
    RemainingUnits=set.intersection(*Sets)
    return RemainingUnits

def FindUnits2(df, MinResponse, MLR=False):
    Sets=[]
    if MLR:
        DF_Response=df.groupby('RealUnit').agg({'MLR': lambda x: np.sum(x)}).reset_index()
    else:
        DF_Response=df.groupby('RealUnit').agg({'MLR': lambda x: np.sum(~x)}).reset_index()
    Sets.append(set(DF_Response[DF_Response['MLR']>=MinResponse]['RealUnit']))
    RemainingUnits=set.intersection(*Sets)
    return RemainingUnits

def ShuffleMLRs(df):
    '''
    Shuffles MLRs for each unit separately (to keep the same number of responses)
    Only the boolean MLR column is shuffled, not the MLRTime
    :param df: Dataframe to shuffle
    :return: Shuffled dataframe
    '''
    for counter,(name, group) in enumerate(df.groupby(['RealUnit'])):
        tmp=group.copy(deep=True)
        tmp['MLR']=shuffle(np.array(tmp['MLR']))
        if counter==0:
            tmpDF= tmp.copy(deep=True)
        else:
            tmpDF=pd.concat([tmpDF, tmp.copy(deep=True)], ignore_index=True)
    return tmpDF


# Only run when this file is run itself
if __name__ == '__main__':
    import glob
    os.chdir("C:/Users/Cansu/Documents/Ephys_Auswertung/olfactory_visual/test") #go to folder with data
    TWOdor = [0.09, 2.09]
    TWBaselOdor = [-20, -0.5]
    TWLight = [0, 0.5]
    TWBaselLight = [-2.5, -0.5]

    OdorCodes = ['F', 'H', 'K', 'J', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
    OdorNames = ['1-Hex', '1-Hep', '1-Oct', '1-Pen', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Ctr']
    LightCodes = ['M', 'N', 'O', 'P']
    LightNames = ['Green100%', 'UV100%', 'Green10%', 'UV10%']
    LightPulsesN = 5

    Files = glob.glob("*.mat")  # open all files in folder
    DataFrame = GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset = 0.09)

    DataFrameMLR = pd.read_excel(io = "MLR_data.xlsx", sheet_name = "Roh")  # create dataframe from Excel file
    TWMLR = [0.09, 2.09]
    DFCorr,MLRAnimalSet = GenDFCorr(DataFrame, DataFrameMLR, OdorCodes, TWMLR)

    print(DFCorr)
    print(MLRAnimalSet)
