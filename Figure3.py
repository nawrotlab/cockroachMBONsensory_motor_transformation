import Dataload
import HelperFunctions
import numpy as np
import matplotlib.pyplot as plt
import os                          # to change path
import glob
import pandas as pd
import spiketools
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pickle

TWOdor = [-1.5, 5]
TWBaselOdor = [-20, -0.5]
BinSize = np.array([0.05])
OdorShift = 0.09
OdorCodes = ['F', 'H', 'K', 'J', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
OdorNames = ['1-Hex', '1-Hep', '1-Oct', '1-Pen', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Ctr']
TWHBar = [0.0, 2.0]

OdorColor = {'C': ((0.765, 0.588, 0.09, 1), (0.914, 0.737, 0.247, 1), (0.95, 0.847, 0.573, 1)),
             'G': ((0.631, 0.114, 0.027, 1), (0.863, 0.522, 0.463, 1), (0.902, 0.659, 0.62, 1)),
             'A': ((0.008, 0.251, 0.455, 1), (0.325, 0.525, 0.694, 1), (0.537, 0.675, 0.788, 1))}


os.chdir("C:/Users/Cansu/Documents/Ephys_Auswertung/olfactory_visual/DataOlfactory")  # go to folder with data
Files = glob.glob("*.mat")     # open all files in folder
DataFrame = Dataload.GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=OdorShift)

DataFrameUnit = DataFrame[DataFrame['RealUnit'] == 15]
TrialSet = np.unique(DataFrameUnit['Trial'])

DataFrame_Control = DataFrameUnit[DataFrameUnit['StimID'] == 'E']

DataFrame_Iso = DataFrameUnit[DataFrameUnit['StimID'] == 'C']
DataFrame_Ben = DataFrameUnit[DataFrameUnit['StimID'] == 'A']
DataFrame_Cin = DataFrameUnit[DataFrameUnit['StimID'] == 'G']

# load PID Data

DataFramePID = []
try:
    with open("PID.pkl", 'rb') as f:
        while True:
            DataFramePID.append(pickle.load(f))
except:
    pass

PID = pd.DataFrame(DataFramePID)
PID34 = PID[PID['PIDID'] == 'PID34']
PIDATime = np.array((PID34.Time[PID34.Stimuli == 'A']-OdorShift))
PIDA = np.array(PID34.PID[PID34.Stimuli == 'A'])
PIDCTime = np.array((PID34.Time[PID34.Stimuli == 'C'])-OdorShift)
PIDC = np.array(PID34.PID[PID34.Stimuli == 'C'])
PIDGTime = np.array((PID34.Time[PID34.Stimuli == 'G'])-OdorShift)
PIDG = np.array(PID34.PID[PID34.Stimuli == 'G'])

#######################################################################################################

# matrix for pop coding & spikecount hbar data

sigmaFR = 60  # defines sigma for kernel
kernel = HelperFunctions.alpha_kernel(sigmaFR, 1.0, calibrateMass=True)

OdorSet = np.unique(DataFrame['StimID'])  # extract all Stimuli
UniqueRealUnits = np.unique(DataFrame['RealUnit'])  # generates Set of Units for every Stimulus

tMin = -1250
tMax = 4000
t = abs(tMin) + tMax
Edge = sigmaFR * 3

MatList = list(range(len(OdorSet)))
AvSpikeCountList = list(range(len(OdorSet)))
y = len(OdorSet)                # number of Stimuli for matrix
x = len(UniqueRealUnits)        # number of Units for matrix
responsematrix = np.zeros((x, y))

for ind, Stimulus in enumerate(OdorNames):      # loops over each Stimulus
    DfStim = DataFrame[DataFrame['StimName'] == Stimulus]  # Df for each Stimulus seperated
    UnitSet = np.unique(DfStim['RealUnit'])     # generates Set of Units for every Stimulus
    matrix = np.zeros((len(UnitSet), t - 2 * Edge))  # generates matrix with zeros with correct dimensions
    AvSpikeCount = np.zeros(len(UnitSet))
    AvSpikeCountBasel = np.zeros(len(UnitSet))

    for idy, Unit in enumerate(UnitSet):        # loops over each Unit
        DfStimUnit = DfStim[DfStim['RealUnit'] == Unit]     # Df for each Unit in specific stimulus
        TrialSet = np.unique(DfStimUnit['Trial'])   # generates Set of Trials for every Unit
        spiketime = np.array([])
        st = np.array([])
        y = np.array([])
        spiketimeBasel = np.array([])
        stBasel = np.array([])
        yBasel = np.array([])
        LenPerTrial = np.array([])
        LenPerTrialBasel = np.array([])

        for idt, Trial in enumerate(TrialSet):  # loops over each trial and gives index
            SpikeTimesPerTrial = DfStimUnit.StimSpikeTimes.iloc[idt]  # identifies cell with respective spiketimes
            CutSpikeTimes = SpikeTimesPerTrial[
                np.logical_and(SpikeTimesPerTrial <= TWHBar[1], SpikeTimesPerTrial >= TWHBar[0])]
            SpikeTimesPerTrialBasel = DfStimUnit.BaselSpikeTimes.iloc[idt]  # identifies cell with respective spiketimes
            AdjLenBasel = len(SpikeTimesPerTrialBasel) / (TWBaselOdor[1] - TWBaselOdor[0]) * (TWHBar[1] - TWHBar[0])
            LenPerTrial = np.append(LenPerTrial, len(CutSpikeTimes))
            LenPerTrialBasel = np.append(LenPerTrialBasel, AdjLenBasel)
            SpikeTimesPerTrialBasel = DfStimUnit.BaselSpikeTimes.iloc[idt]  # identifies cell with respective spiketimes
            st = np.hstack((st,
                            SpikeTimesPerTrial * 1000))  # merges Spiketimes of all trials in one Unit and one Stimulus; s to ms
            stBasel = np.hstack((stBasel,
                                 SpikeTimesPerTrialBasel * 1000))  # merges Spiketimes of all trials in one Unit and one Stimulus; s to ms
            a = np.full(len(SpikeTimesPerTrial),
                        idt)  # generates array with respective trialID depending on number of spikes
            aBasel = np.full(len(SpikeTimesPerTrialBasel),
                             idt)  # generates array with respective trialID depending on number of spikes
            y = np.hstack((y, a))  # merges trialIDs of alle trials in one Unit and one Stimulus
            yBasel = np.hstack((yBasel, aBasel))  # merges trialIDs of alle trials in one Unit and one Stimulus

        AvSpikeCount[idy] = np.average(LenPerTrial) - np.average(LenPerTrialBasel)
        AvSpikeCountBasel[idy] = np.average(LenPerTrialBasel)

        st = np.hstack((st, tMax + 2))  # adds one spiketime after relevant timespan
        stBasel = np.hstack((stBasel, TWBaselOdor[1] + 2))  # adds one spiketime after relevant timespan
        y = np.hstack((y, idt))  # adds one object for the last trial
        yBasel = np.hstack((yBasel, idt))  # adds one object for the last trial
        spiketime = np.vstack([st, y])  # merges spiketimes with respective trialID
        spiketimeBasel = np.vstack([stBasel, yBasel])  # merges spiketimes with respective trialID
        spikerates = spiketools.kernel_rate(spiketime, kernel, tlim=[tMin, tMax],
                                            pool=True)  # calculates spikerates with moving kernel
        spikeratesBasel = spiketools.kernel_rate(spiketimeBasel, kernel,
                                                 tlim=[TWBaselOdor[0] * 1000, TWBaselOdor[1] * 1000],
                                                 pool=True)  # calculates spikerates with moving kernel
        spikeratesMinusBasel = spikerates[0] - np.average(spikeratesBasel[0])
        matrix[idy, :] = spikeratesMinusBasel  # creates matrix with spikerates for all units of one stimulus
        AvSpikeRate = np.average(
            spikeratesMinusBasel[0][(spikerates[1] <= TWHBar[1] * 1000) & (spikerates[1] >= TWHBar[0] * 1000)])
        responsematrix[idy, ind] = AvSpikeRate

    AvSpikeCountList[ind] = AvSpikeCount
    MatList[ind] = matrix
    RangeUnits = list(range(1, len(UnitSet) + 1, 2))

UnitMax = np.zeros([len(UnitSet), len(MatList)])
for mat in range(len(MatList)):
    ActualMat = MatList[mat]
    for Unit in UnitSet:
        UnitMax[Unit, mat] = max(ActualMat[Unit])

for Unit in UnitSet:
    Max = np.amax(UnitMax[Unit])
    RespMax = np.amax(responsematrix[Unit])
    responsematrix[Unit] = responsematrix[Unit] / RespMax

    for mat in range(len(MatList)):
        MatList[mat][Unit] = MatList[mat][Unit] / Max


RateTimes = spikerates[1]-spikerates[1][0]
RateTimes = np.arange(RateTimes[0], RateTimes[-1], 1000)
RateTimeLabels = (RateTimes-1000)/1000
RateTimeLabels = RateTimeLabels.astype(int)

# Figure plotting
cm = 1 / 2.54
xlim = [-0.3, 3.1]
TickLength = 2.5

name = 'my_list'
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
cmap = LinearSegmentedColormap.from_list(name, colors, N=100)
OdorID1 = 10
OdorID2 = 9
OdorID3 = 8
OdorID4 = 7
TLS = 7  # TickLabelSize
LS = 8  # LabelSize
TS = 9  # TitleSize
LetterSize = 12

plt.rcParams["font.family"] = "arial"

fig = plt.figure(figsize=(17 * cm, 9 * cm))
fig.tight_layout()

# Sub1
sub1 = fig.add_subplot(3, 4, 1)
ax1 = sub1.twinx()

ii = 1
diff = 5
Stimulus = np.unique(DataFrame_Control['StimName'])
for index, Trial in DataFrame_Control.iterrows():
    y = np.ones_like(Trial['StimSpikeTimes']) * ii
    ax1.plot(Trial['StimSpikeTimes'], y, '|k', markersize=4, markeredgewidth=0.5)
    ii = ii + diff

ax1.set_title(Stimulus[0], size=TS, y=0.97)
ax1.set_ylabel('Trial', size=LS)
ax1.set_yticks((1, 1 + diff * 9))
ax1.set_yticklabels((1, 10), size=TLS)
ax1.set_ylim(-2, 1 + diff * 9 + 3)
ax1.yaxis.set_ticks_position("left")
ax1.yaxis.set_label_position("left")
ax1.set_xticks((0 - OdorShift, 0, 1, 2, 3))
ax1.axes.xaxis.set_ticklabels([])
ax1.tick_params(axis='both', length=TickLength)
ax1.axvline(x=0, linewidth=0.5, color='grey')
ax1.axvline(x=2, linewidth=0.5, color='grey')
ax1.set_xlim(xlim[0], xlim[1])
sub1.tick_params(axis='both', length=TickLength)
sub1.axes.yaxis.set_ticklabels([])
sub1.axes.yaxis.set_ticks([])
sub1.text(-1.3, 1, 'A', weight='bold', size=LetterSize)

l1, b1, w1, h1 = sub1.get_position().bounds
l1 = l1 - 0.06
w1 = w1 + 0.035
b1 = b1 + 0.1
h1 = h1 / 1.3
sub1.set_position([l1, b1, w1, h1])
sub1.yaxis.set_ticks_position("right")

# Sub2
sub2 = fig.add_subplot(3, 4, 2)
ax2 = sub2.twinx()

ii = 1
diff = 5
Stimulus = np.unique(DataFrame_Cin['StimName'])
for index, Trial in DataFrame_Cin.iterrows():
    y = np.ones_like(Trial['StimSpikeTimes']) * ii
    ax2.plot(Trial['StimSpikeTimes'], y, '|k', markersize=4, markeredgewidth=0.5)
    ii = ii + diff
ax2.yaxis.set_ticks_position("left")
ax2.tick_params(axis='both', length=TickLength)
ax2.set_yticks((1, 1 + diff * 9))
ax2.axes.yaxis.set_ticklabels([])
ax2.set_ylim(-2, 1 + diff * 9 + 3)
sub2.plot(PIDGTime[0], PIDG[0], color=OdorColor['G'][1], linewidth=0.6)
sub2.axes.yaxis.set_ticklabels([])
sub2.tick_params(axis='both', length=TickLength)
sub2.set_title(Stimulus[0], size=TS, y=0.97)
sub2.set_xlim(xlim[0], xlim[1])
sub2.axes.xaxis.set_ticklabels([])
sub2.tick_params(axis='both', length=TickLength)
sub2.set_xticks((0 - OdorShift, 0, 1, 2, 3))
sub2.axvline(x=0, linewidth=0.5, color='grey')
sub2.axvline(x=2, linewidth=0.5, color='grey')

l2, b2, w2, h2 = sub2.get_position().bounds
l2 = l1 + w1 + 0.023
sub2.set_position([l2, b1, w1, h1])
sub2.yaxis.set_ticks_position("right")

# Sub3
sub3 = fig.add_subplot(3, 4, 3)
ax3 = sub3.twinx()
sub3.plot(PIDATime[0], PIDA[0], color=OdorColor['A'][1], linewidth=0.6)
sub3.tick_params(axis='both', length=TickLength)
sub3.yaxis.set_ticks_position("right")
sub3.set_xlim(xlim[0], xlim[1])
sub3.axes.xaxis.set_ticklabels([])
sub3.axes.yaxis.set_ticklabels([])
sub3.set_xticks((0 - OdorShift, 0, 1, 2, 3))
l3, b3, w3, h3 = sub3.get_position().bounds
l3 = l2 + w1 + 0.023
sub3.set_position([l3, b1, w1, h1])
sub3.axvline(x=0, linewidth=0.5, color='grey')
sub3.axvline(x=2, linewidth=0.5, color='grey')

ii = 1
diff = 5
Stimulus = np.unique(DataFrame_Ben['StimName'])
for index, Trial in DataFrame_Ben.iterrows():
    y = np.ones_like(Trial['StimSpikeTimes']) * ii
    ax3.plot(Trial['StimSpikeTimes'], y, '|k', markersize=4, markeredgewidth=0.5)
    ii = ii + diff
ax3.tick_params(axis='both', length=TickLength)
ax3.set_yticks((1, 1 + diff * 9))
ax3.axes.yaxis.set_ticklabels([])
ax3.set_ylim(-2, 1 + diff * 9 + 3)
ax3.yaxis.set_ticks_position("left")
ax3.set_title(Stimulus[0], size=TS, y=0.97)

# Sub3b
sub3b = fig.add_subplot(3, 4, 4)

ax3b = sub3b.twinx()
sub3b.plot(PIDCTime[0], PIDC[0], color=OdorColor['C'][1], linewidth=0.6)
sub3b.tick_params(axis='both', length=TickLength)
sub3b.yaxis.set_ticks_position("right")
sub3b.set_xlim(xlim[0], xlim[1])
sub3b.axes.xaxis.set_ticklabels([])
sub3b.set_xticks((0 - OdorShift, 0, 1, 2, 3))
l3b, b3b, w3b, h3b = sub3b.get_position().bounds
l3b = l3 + w1 + 0.023
sub3b.set_position([l3b, b1, w1, h1])
sub3b.axvline(x=0, linewidth=0.5, color='grey')
sub3b.axvline(x=2, linewidth=0.5, color='grey')
sub3b.set_ylabel('Odor intensity', size=LS)
sub3b.yaxis.set_label_position("right")

ii = 1
diff = 5
Stimulus = np.unique(DataFrame_Iso['StimName'])
for index, Trial in DataFrame_Iso.iterrows():
    y = np.ones_like(Trial['StimSpikeTimes']) * ii
    ax3b.plot(Trial['StimSpikeTimes'], y, '|k', markersize=4, markeredgewidth=0.5)
    ii = ii + diff
ax3b.tick_params(axis='both', length=TickLength)
ax3b.set_yticks((1, 1 + diff * 9))
ax3b.axes.yaxis.set_ticklabels([])
ax3b.set_ylim(-2, 1 + diff * 9 + 3)
ax3b.yaxis.set_ticks_position("left")
ax3b.set_title(Stimulus[0], size=TS, y=0.97)

# Sub4
sub4 = fig.add_subplot(3, 4, 5)
TimeAxis = np.arange(TWOdor[0], TWOdor[1], BinSize)
his = np.zeros(np.shape(TimeAxis)[0] - 1)
for index, Trial in DataFrame_Control.iterrows():
    his = his + np.histogram(Trial['StimSpikeTimes'], bins=TimeAxis)[0]
sub4.bar(TimeAxis[:-1], his, width=np.diff(TimeAxis), color='k', align="edge")
sub4.set_ylim(0, 30)
sub4.set_xlim(xlim[0], xlim[1])
sub4.set_ylabel('Spike count', size=LS)
sub4.tick_params(axis='both', length=TickLength, labelsize=LS)
sub4.set_xticks((0 - OdorShift, 0, 1, 2, 3))
sub4.set_xticklabels(('', 0, 1, 2, 3), size=TLS)
sub4.set_yticks(range(0, 40, 10))
l4, b4, w4, h4 = sub4.get_position().bounds
l4 = l1
b4 = b1 - h1 - 0.05
sub4.set_position([l4, b4, w1, h1])
sub4.axvline(x=0, linewidth=0.5, color='grey')
sub4.axvline(x=2, linewidth=0.5, color='grey')
sub4.plot(-OdorShift, -3.5, '^', markersize=1.5, clip_on=False, color='k')
sub4.text(-1.3, 32, 'B', weight='bold', size=LetterSize)

# Sub5
sub5 = fig.add_subplot(3, 4, 6)
TimeAxis = np.arange(TWOdor[0], TWOdor[1], BinSize)
his = np.zeros(np.shape(TimeAxis)[0] - 1)
for index, Trial in DataFrame_Cin.iterrows():
    his = his + np.histogram(Trial['StimSpikeTimes'], bins=TimeAxis)[0]
sub5.bar(TimeAxis[:-1], his, width=np.diff(TimeAxis), color='k', align="edge")
sub5.set_ylim(0, 30)
sub5.set_xlim(xlim[0], xlim[1])
sub5.tick_params(axis='both', length=TickLength)
sub5.set_xticks((0 - OdorShift, 0, 1, 2, 3))
sub5.set_xticklabels(('', 0, 1, 2, 3), size=TLS)
sub5.plot(-OdorShift, -3.5, '^', markersize=1.5, clip_on=False, color='k')
sub5.set_yticks(range(0, 40, 10))
sub5.axes.yaxis.set_ticklabels([])
l5, b5, w5, h5 = sub5.get_position().bounds
l5 = l4 + w1 + 0.023
sub5.set_position([l5, b4, w1, h1])
sub5.axvline(x=0, linewidth=0.5, color='grey')
sub5.axvline(x=2, linewidth=0.5, color='grey')

# Sub6
sub6 = fig.add_subplot(3, 4, 7)
TimeAxis = np.arange(TWOdor[0], TWOdor[1], BinSize)
his = np.zeros(np.shape(TimeAxis)[0] - 1)
for index, Trial in DataFrame_Ben.iterrows():
    his = his + np.histogram(Trial['StimSpikeTimes'], bins=TimeAxis)[0]
sub6.bar(TimeAxis[:-1], his, width=np.diff(TimeAxis), color='k', align="edge")
sub6.set_ylim(0, 30)
sub6.set_xlim(xlim[0], xlim[1])
sub6.tick_params(axis='both', length=TickLength)
sub6.set_xticks((0-OdorShift, 0, 1, 2, 3))
sub6.set_xticklabels(('', 0, 1, 2, 3), size=TLS)
sub6.plot(-OdorShift, -3.5, '^', markersize=1.5, clip_on=False, color='k')
sub6.set_yticks(range(0, 40, 10))
sub6.axes.yaxis.set_ticklabels([])
l6, b6, w6, h6 = sub6.get_position().bounds
l6 = l5 + w1 + 0.023
sub6.set_position([l6, b4, w1, h1])
sub6.axvline(x=0, linewidth=0.5, color='grey')
sub6.axvline(x=2, linewidth=0.5, color='grey')

# Sub6b
sub6b = fig.add_subplot(3, 4, 8)
TimeAxis = np.arange(TWOdor[0], TWOdor[1], BinSize)
his = np.zeros(np.shape(TimeAxis)[0] - 1)
for index, Trial in DataFrame_Iso.iterrows():
    his = his + np.histogram(Trial['StimSpikeTimes'], bins=TimeAxis)[0]
sub6b.bar(TimeAxis[:-1], his, width=np.diff(TimeAxis), color='k', align="edge")
sub6b.set_ylim(0, 30)
sub6b.set_xlim(xlim[0], xlim[1])
sub6b.tick_params(axis='both', length=TickLength)
sub6b.set_xticks((0-OdorShift, 0, 1, 2, 3))
sub6b.set_xticklabels(('', 0, 1, 2, 3), size=TLS)
sub6b.plot(-OdorShift, -3.5, '^', markersize=1.5, clip_on=False, color='k')
sub6b.set_yticks(range(0, 40, 10))
sub6b.axes.yaxis.set_ticklabels([])
l6b, b6b, w6b, h6b = sub6b.get_position().bounds
l6b = l6 + w1 + 0.023
sub6b.set_position([l6b, b4, w1, h1])
sub6b.axvline(x=0, linewidth=0.5, color='grey')
sub6b.axvline(x=2, linewidth=0.5, color='grey')

# Sub7
sub7 = fig.add_subplot(3, 5, 11)
sns.heatmap(MatList[OdorID1], ax=sub7, cbar=False, vmin=-1.01, vmax=1.01, cmap="coolwarm")
l7, b7, w7, h7 = sub7.get_position().bounds
l7 = l1
w7 = w7 + 0.025
b7 = b7 - 0.01
h7 = h7 + 0.1
sub7.set_position([l7, b7, w7, h7])
sub7.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub7.set_title(OdorNames[OdorID1], size=TS, y=0.96)
sub7.set_ylabel('Unit', size=LS)
sub7.set_yticks((0.5, 30.5))
sub7.set_yticklabels((1, 31), rotation=0)
sub7.invert_yaxis()
sub7.set_xticks(RateTimes)
sub7.set_xticklabels(RateTimeLabels, size=TLS, rotation=0)
sub7.set_xlabel('Time [s]', size=LS)
sub7.set_xlim(500, 4300)
sub7.axvline(x=RateTimes[1], linewidth=0.5, color='grey')
sub7.axvline(x=RateTimes[3], linewidth=0.5, color='grey')
sub7.text(-900, 32, 'C', weight='bold', size=LetterSize)
sub7.plot(390, 15.5, '>', markersize=1.5, clip_on=False, color='k')

# Sub8
sub8 = fig.add_subplot(3, 5, 12)
sns.heatmap(MatList[OdorID2], ax=sub8, cbar=False, vmin=-1.01, vmax=1.01, cmap='coolwarm')
l8, b8, w8, h8 = sub8.get_position().bounds
l8 = l7 + 0.195
w8 = w7
sub8.set_position([l8, b7, w8, h7])
sub8.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub8.axes.yaxis.set_ticklabels([])
sub8.invert_yaxis()
sub8.set_title(OdorNames[OdorID2], size=TS, y=0.96)
sub8.set_xticks(RateTimes)
sub8.set_xticklabels(RateTimeLabels, size=TLS, rotation=0)
sub8.set_yticks((0.5, 30.5))
sub8.set_xlabel('Time [s]', size=LS)
sub8.set_xlim(500, 4300)
sub8.axvline(x=RateTimes[1], linewidth=0.5, color='grey')
sub8.axvline(x=RateTimes[3], linewidth=0.5, color='grey')
sub8.plot(390, 15.5, '>', markersize=1.5, clip_on=False, color='k')

# Sub9
sub9 = fig.add_subplot(3, 5, 13)
sns.heatmap(MatList[OdorID3], ax=sub9, cbar=False, vmin=-1.01, vmax=1.01, cmap="coolwarm")
l9, b9, w9, h9 = sub9.get_position().bounds
l9 = l8 + 0.195
w9 = w8
sub9.set_position([l9, b7, w7, h7])
sub9.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub9.axes.yaxis.set_ticklabels([])
sub9.invert_yaxis()
sub9.set_title(OdorNames[OdorID3], size=TS, y=0.96)
sub9.set_xticks(RateTimes)
sub9.set_yticks((0.5, 30.5))
sub9.set_xticklabels(RateTimeLabels, rotation=0)
sub9.set_xlabel('Time [s]', size=LS)
sub9.set_xlim(500, 4300)
sub9.axvline(x=RateTimes[1], linewidth=0.5, color='grey')
sub9.axvline(x=RateTimes[3], linewidth=0.5, color='grey')
sub9.plot(390, 15.5, '>', markersize=1.5, clip_on=False, color='k')

# Sub10
sub10 = fig.add_subplot(3, 5, 14)
sns.heatmap(MatList[OdorID4], ax=sub10, vmin=-1.01, vmax=1.01, cmap='coolwarm', cbar=False)
sub10.figure.axes[-1].yaxis.label.set_size(LS)
l10, b10, w10, h10 = sub10.get_position().bounds
l10 = l9 + 0.195
w10 = w9
sub10.set_position([l10, b7, w7, h7])
sub10.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub10.axes.yaxis.set_ticklabels([])
sub10.invert_yaxis()
sub10.set_title(OdorNames[OdorID4], size=TS, y=0.96)
sub10.set_xticks(RateTimes)
sub10.set_xticklabels(RateTimeLabels, rotation=0)
sub10.set_yticks((0.5, 30.5))
sub10.set_xlim(500, 4300)
sub10.set_xlabel('Time [s]', size=LS)
sub10.axvline(x=RateTimes[1], linewidth=0.5, color='grey')
sub10.axvline(x=RateTimes[3], linewidth=0.5, color='grey')
sub10.plot(390, 15.5, '>', markersize=1.5, clip_on=False, color='k')

# Sub15
sub15 = fig.add_subplot(3, 5, 15)
fig = sns.heatmap(responsematrix, ax=sub15, square=True, cbar_kws={'label': 'Norm. rate', 'ticks': [-1, 0, 1]},
                  cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.001, linecolor='grey')
sub15.set_yticks((0.5, 30.5))
sub15.axes.yaxis.set_ticklabels([])
sub15.invert_yaxis()
sub15.set_xticks((0.5, 10.5))
sub15.set_xticklabels((1, 11), rotation=0)
sub15.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub15.set_xlabel('Stimulus', size=LS)
sub15.plot(-1.26, 15.5, '>', markersize=1.5, clip_on=False, color='k')
sub15.text(-5, 32, 'D', weight='bold', size=LetterSize)
l15, b15, w15, h15 = sub15.get_position().bounds
l15 = l10 + 0.105
sub15.set_position([l15, b7, w7, h7])
colorbar = plt.gcf().axes[-1]
colorbar.tick_params(labelsize=TLS)
ll15, bb15, ww15, hh15 = colorbar.get_position().bounds
colorbar.set_position([ll15 + 0.05, b7, ww15, h7])
colorbar.figure.axes[-1].yaxis.label.set_size(LS)

plt.savefig('Figure3.png', dpi=300)
plt.show()

