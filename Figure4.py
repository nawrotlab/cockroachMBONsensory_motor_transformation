import pickle
from Functions import Dataload, PCA, Latency, Classifier, HelperFunctions
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats
from Data import DataSelection
import argparse

# command line optional argument for Datapath
parser = argparse.ArgumentParser(description='Path to data folder')
parser.add_argument('--path', type=str, default="Data",
                    help='Path to data folder')
args = parser.parse_args()
Path = args.path

Files = [os.path.join(Path, x) for x in DataSelection.DataCorr]

OdorShift = 0.09
TWOdor = [-3., 5]
TWClassifier = [-0.9, 2.5]
TWPlot = [0.0, 2.2]
TWBaselOdor = [-20.0, -0.5]
TWMLR = [0.0, 2.0]  # TW for MLR DfCorr
TWPCA = [-2.6, 4.6]
TWStimulation = [0.0, 2.0]
BorderLim = 0.97

OdorCodes = ['F', 'H', 'K', 'J', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
OdorNames = ['1-Hex', '1-Hep', '1-Oct', '1-Pen', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Ctr']
FoodOdors = ['C', 'A', 'G']
file_name = "MLR_data.xlsx"     # name for Excel file
sheet = "Roh"

OdorColor = {'C': (0.765, 0.588, 0.09, 1), 'G': (0.753, 0.129, 0.027, 1), 'A': (0.082, 0.31, 0.553, 1)}
OdorColorMLR = {'C': ((0.765, 0.588, 0.09, 1), (0.914, 0.737, 0.247, 1), (0.95, 0.847, 0.573, 1)),
                'G': ((0.631, 0.114, 0.027, 1), (0.863, 0.522, 0.463, 1), (0.902, 0.659, 0.62, 1)),
                'A': ((0.008, 0.251, 0.455, 1), (0.325, 0.525, 0.694, 1), (0.537, 0.675, 0.788, 1))}

# OdorColorMLR 1:Neuro, 2:MLR, 3:Shadow
NeuroColor = {'MLR': (0.102, 0.416, 0.278), 'NoMLR': (0.502, 0.675, 0.6), 'Both': (0.298, 0.5411, 0.439, 1)}
NeuroColorShadow = {'MLR': (0.459, 0.647, 0.565), 'NoMLR': (0.698, 0.804, 0.757)}

#os.chdir("C:/Users/Cansu/Documents/Ephys_Auswertung/olfactory_visual/FinalFigure")  # go to folder with data
#Files = glob.glob("*.mat")     # open all files in folder

DataFrame = Dataload.GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=OdorShift)
Stimuli = np.unique(DataFrame['StimID'])                 # extract all Stimuli
DataFrameMLR = pd.read_excel(io=os.path.join(Path, file_name), sheet_name=sheet)       # create dataframe from Excel file
DfCorr, MLRAnimalSet = Dataload.MergeNeuronal_MLR(DataFrame, DataFrameMLR, TWMLR=TWMLR, T0=OdorShift)


# calculate median spike count for subplot 1

DFCorr3Odors = Dataload.LimitDFtoStimulus(DfCorr, FoodOdors)
UnitOfInterest = 14
StimDur = TWStimulation[1] - TWStimulation[0]
UniqueRealUnits = np.unique(DFCorr3Odors['RealUnit'])
DFCorr3Odors['CountStim'] = DFCorr3Odors.StimSpikeTimes.apply(lambda x: len(x[(x >= TWStimulation[0]) & (x <= TWStimulation[1])]))
BaselDur = TWBaselOdor[1] - TWBaselOdor[0]
DFCorr3Odors['CountBaselAdj'] = DFCorr3Odors.BaselSpikeTimes.apply(lambda x: len(x) / BaselDur * StimDur)
DFCorr3Odors['AdjustedCount'] = DFCorr3Odors['CountStim'] - DFCorr3Odors['CountBaselAdj']
SPCountMatrixMLR0 = np.zeros(len(UniqueRealUnits))
SPCountMatrixMLR1 = np.zeros(len(UniqueRealUnits))

for ind, Unit in enumerate(UniqueRealUnits):
    DfUnit = DFCorr3Odors[DFCorr3Odors.RealUnit == Unit]
    DfMLR0 = DfUnit[DfUnit.MLR == 0]
    DfMLR1 = DfUnit[DfUnit.MLR == 1]
    SpikeCountMLR0 = np.median(DfMLR0.AdjustedCount)
    SpikeCountMLR1 = np.median(DfMLR1.AdjustedCount)
    SPCountMatrixMLR0[ind] = SpikeCountMLR0
    SPCountMatrixMLR1[ind] = SpikeCountMLR1


# calculate spike rate for subplot 2 & 3

sigmaFR = 60  # defines sigma for kernel
kernel = HelperFunctions.alpha_kernel(sigmaFR, 1.0, calibrateMass=True)
KernelLimit = np.max(kernel) * 1000 * 2.25 / 10
UniqueRealUnits = np.unique(DFCorr3Odors['RealUnit'])
Edge = sigmaFR * 3
tMin = (TWPlot[0] * 1000) - Edge
tMax = (TWPlot[1] * 1000) + Edge
t = tMax - tMin
Dur = int((TWPlot[1] - TWPlot[0]) * 1000)
SpRaMatrix0 = np.zeros((len(UniqueRealUnits), Dur))
SpRaMatrix1 = np.zeros((len(UniqueRealUnits), Dur))
TotalMLR0Trials = 0
TotalMLR1Trials = 0
Maxima = {}
Bound = {}

for idy, Unit in enumerate(UniqueRealUnits):  # loops over each Unit
    DfCorrUnit = DFCorr3Odors[DFCorr3Odors['RealUnit'] == Unit]
    DfCorrUnitMLR0 = DfCorrUnit[DfCorrUnit.MLR == 0]
    DfCorrUnitMLR1 = DfCorrUnit[DfCorrUnit.MLR == 1]
    LenStimMLR0 = np.array([])
    LenStimMLR1 = np.array([])
    LenBaselMLR0 = np.array([])
    LenBaselMLR1 = np.array([])
    LenBasel = np.array([])
    SpTStimMLR0 = np.array([])
    SpTStimMLR1 = np.array([])
    SpTBaselMLR0 = np.array([])
    SpTBaselMLR1 = np.array([])
    SpTBasel = np.array([])
    TrialsStimMLR0ST = np.array([])
    TrialsStimMLR1ST = np.array([])
    TrialsBaselMLR0ST = np.array([])
    TrialsBaselMLR1ST = np.array([])
    TrialsBaselST = np.array([])
    SpTTrStimMLR0 = np.array([])
    SpTTrStimMLR1 = np.array([])
    SpTTrBaselMLR0 = np.array([])
    SpTTrBaselMLR1 = np.array([])
    SpTTrBasel = np.array([])
    m = 0
    n = 0
    o = 0

    for idt, row in DfCorrUnitMLR0.iterrows():  # loops over each trial and gives index
        SpikeTimesPerTrialtMLR0 = row.StimSpikeTimes
        SpikeTimesPerTrialBaseltMLR0 = row.BaselSpikeTimes
        LenStimMLR0 = np.append(LenStimMLR0, len(SpikeTimesPerTrialtMLR0))
        LenBaselMLR0 = np.append(LenBaselMLR0, len(SpikeTimesPerTrialBaseltMLR0))
        SpTStimMLR0 = np.hstack((SpTStimMLR0, SpikeTimesPerTrialtMLR0 * 1000))
        SpTBaselMLR0 = np.hstack((SpTBaselMLR0, SpikeTimesPerTrialBaseltMLR0 * 1000))
        TrialsStimMLR0 = np.full(len(SpikeTimesPerTrialtMLR0), m)
        TrialsBaselMLR0 = np.full(len(SpikeTimesPerTrialBaseltMLR0), m)
        TrialsStimMLR0ST = np.hstack((TrialsStimMLR0ST, TrialsStimMLR0))
        TrialsBaselMLR0ST = np.hstack((TrialsBaselMLR0ST, TrialsBaselMLR0))
        m = m + 1

    for idt, row in DfCorrUnitMLR1.iterrows():  # loops over each trial and gives index
        SpikeTimesPerTrialtMLR1 = row.StimSpikeTimes
        SpikeTimesPerTrialBaseltMLR1 = row.BaselSpikeTimes
        LenStimMLR1 = np.append(LenStimMLR1, len(SpikeTimesPerTrialtMLR1))
        LenBaselMLR1 = np.append(LenBaselMLR1, len(SpikeTimesPerTrialBaseltMLR1))
        SpTStimMLR1 = np.hstack((SpTStimMLR1, SpikeTimesPerTrialtMLR1 * 1000))
        SpTBaselMLR1 = np.hstack((SpTBaselMLR1, SpikeTimesPerTrialBaseltMLR1 * 1000))
        TrialsStimMLR1 = np.full(len(SpikeTimesPerTrialtMLR1), n)
        TrialsBaselMLR1 = np.full(len(SpikeTimesPerTrialBaseltMLR1), n)
        TrialsStimMLR1ST = np.hstack((TrialsStimMLR1ST, TrialsStimMLR1))
        TrialsBaselMLR1ST = np.hstack((TrialsBaselMLR1ST, TrialsBaselMLR1))
        n = n + 1

    for idt, row in DfCorrUnit.iterrows():  # loops over each trial and gives index
        SpikeTimesPerTrialBasel = row.BaselSpikeTimes
        LenBasel = np.append(LenBasel, len(SpikeTimesPerTrialBasel))
        SpTBasel = np.hstack((SpTBasel, SpikeTimesPerTrialBasel * 1000))
        TrialsBasel = np.full(len(SpikeTimesPerTrialBasel), o)
        TrialsBaselST = np.hstack((TrialsBaselST, TrialsBasel))

    SpTStimMLR0 = np.hstack((SpTStimMLR0, tMax + 2))
    SpTStimMLR1 = np.hstack((SpTStimMLR1, tMax + 2))
    SpTBaselMLR0 = np.hstack((SpTBaselMLR0, TWBaselOdor[1] + 2))
    SpTBaselMLR1 = np.hstack((SpTBaselMLR1, TWBaselOdor[1] + 2))
    SpTBasel = np.hstack((SpTBasel, TWBaselOdor[1] + 2))
    TrialsStimMLR0ST = np.hstack((TrialsStimMLR0ST, m))
    TrialsStimMLR1ST = np.hstack((TrialsStimMLR1ST, n))
    TrialsBaselMLR0ST = np.hstack((TrialsBaselMLR0ST, m))
    TrialsBaselMLR1ST = np.hstack((TrialsBaselMLR1ST, n))
    TrialsBaselST = np.hstack((TrialsBaselST, n))
    SpTTrStimMLR0 = np.vstack([SpTStimMLR0, TrialsStimMLR0ST])
    SpTTrStimMLR1 = np.vstack([SpTStimMLR1, TrialsStimMLR1ST])
    SpTTrBaselMLR0 = np.vstack([SpTBaselMLR0, TrialsBaselMLR0ST])
    SpTTrBaselMLR1 = np.vstack([SpTBaselMLR1, TrialsBaselMLR1ST])
    SpTTrBasel = np.vstack([SpTBasel, TrialsBaselST])
    SpRaBasel = HelperFunctions.kernel_rate(SpTTrBasel, kernel, tlim=[TWBaselOdor[0] * 1000, TWBaselOdor[1] * 1000], pool=True)
    SpRaStimMLR0 = HelperFunctions.kernel_rate(SpTTrStimMLR0, kernel, tlim=[tMin, tMax], pool=True)
    SpRaBaselMLR0 = HelperFunctions.kernel_rate(SpTTrBaselMLR0, kernel, tlim=[TWBaselOdor[0] * 1000, TWBaselOdor[1] * 1000], pool=True)
    SpRaMLR0 = SpRaStimMLR0[0][0] - np.average(SpRaBaselMLR0[0])
    SpRaStimMLR1 = HelperFunctions.kernel_rate(SpTTrStimMLR1, kernel, tlim=[tMin, tMax], pool=True)
    SpRaBaselMLR1 = HelperFunctions.kernel_rate(SpTTrBaselMLR1, kernel, tlim=[TWBaselOdor[0] * 1000, TWBaselOdor[1] * 1000], pool=True)
    SpRaMLR1 = SpRaStimMLR1[0][0] - np.average(SpRaBaselMLR1[0])
    SpRa = SpRaBasel[0][0] - np.average(SpRaBasel[0])
    SRMax = np.max((SpRaMLR1, SpRaMLR0))
    NormSpRaMLR0 = SpRaMLR0 / SRMax
    NormSpRaMLR1 = SpRaMLR1 / SRMax
    Maxima.update({Unit:(SRMax)})

    if Unit == UnitOfInterest:
        SpRaMLR0UnitOfInt = SpRaMLR0
        SpRaMLR1UnitOfInt = SpRaMLR1
        TotalMLR0Trials = TotalMLR0Trials + m
        TotalMLR1Trials = TotalMLR1Trials + n
        SortedBasel = np.sort(SpRa)
        LowerInd = len(SortedBasel) * (1 - BorderLim)
        UpperInd = len(SortedBasel) * (BorderLim)
        LowerLimit = SortedBasel[int(np.floor(LowerInd)) - 1]
        UpperLimit = SortedBasel[int(np.ceil(UpperInd)) - 1]

        if UpperLimit < KernelLimit:
            UpperLimit = KernelLimit

        Bound.update({UnitOfInterest: (LowerLimit, UpperLimit)})

    SpRaMatrix0[idy] = NormSpRaMLR0
    SpRaMatrix1[idy] = NormSpRaMLR1

MeanSpRa0 = np.mean(SpRaMatrix0, axis=0)
MeanSpRa1 = np.mean(SpRaMatrix1, axis=0)
StErSpRa0 = np.std(SpRaMatrix0, axis=0, ddof=1) / np.sqrt(np.size(SpRaMatrix0, axis=0))
StErSpRa1 = np.std(SpRaMatrix1, axis=0, ddof=1) / np.sqrt(np.size(SpRaMatrix1, axis=0))
RateTimes = SpRaStimMLR1[1] - SpRaStimMLR1[1][0]
RateTimes = np.arange(RateTimes[0], RateTimes[-1], 500)
RateTimeLabels = np.arange(SpRaStimMLR1[1][0] + 0.5, SpRaStimMLR1[1][-1] + 0.5, 500) / 1000
RateTimesOG = SpRaStimMLR1[1]

# Statistical Test
T = np.arange(0, SpRaMatrix1.shape[1])
significance_level = 0.05
significance = np.ones_like(T)*0.99999
for ii in range(SpRaMatrix1.shape[1]):
    jj = stats.wilcoxon(SpRaMatrix1[:, ii], SpRaMatrix0[:, ii], alternative='greater', zero_method='wilcox')[1]
    significance[ii] = jj
bool_sig = significance < significance_level
SigTime = (np.min(T[bool_sig]))

# PCA for subplot 4
TauFR = 200
WidthFactor = 5
dt = 1.
KernelPCA = HelperFunctions.alpha_kernel(TauFR, dt=dt, nstd=WidthFactor)
Transform = PCA.PCA_Neuronal(DfCorr, KernelPCA, TWPCA, TWBaselOdor, n_components=2, Stims=['A', 'C', 'G'])
BaseInd = np.where(np.array(Transform.Time[0]) < 90)
StimInd = np.where(np.array(Transform.Time[0]) >= 90)

# calculating Latencies for subplots 5-7
# we generate a new dataframe with new cut windows of spiketimes, due to kernel onset correction

UseSingeTrialBL = True
MinSpike = 2.25
TauFR = 250
WidthFactor = 5
dt = 1.
kernel = HelperFunctions.exponential_kernel(TauFR, dt=dt, nstd=WidthFactor)
AnimalsExclude = ["CA70", "CA69", "CA71"]  # exclude because of too low frame rate
DFLatency = Dataload.GenDF(Files, TWStimulation, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=OdorShift)
DFLatency.drop(DFLatency[[x[0:4] in AnimalsExclude for x in DFLatency['AnimalID']]].index,
        inplace=True)
DFLatency.reset_index(drop=True, inplace=True)
DFLatency, _ = Dataload.MergeNeuronal_MLR(DFLatency, DataFrameMLR, T0=OdorShift, TWMLR=TWMLR)
LatencyDF = Latency.Latency(DFLatency, kernel, TWStimulation, TWBaselOdor, MinSpike=MinSpike, Border=BorderLim,
                            Single_Trial_BL=UseSingeTrialBL, Stims=['A', 'C', 'G'])
LatencyDF = LatencyDF[LatencyDF['MLR'] == True]

# classifier for subplot 8
TauFR = 50  # defines sigma for gaussian kernel (half of gaussian width)
WidthFactor = 6
dt = 1.
KernelClassifier = HelperFunctions.alpha_kernel(TauFR, dt=dt, nstd=WidthFactor, calibrateMass=False)
InterestingOdors = ['C', 'G', 'A']
ClassifierAccuracy, ClassifierTime, ConsideredUnits = Classifier.Classifier(DfCorr, KernelClassifier, TWClassifier, TWBaselOdor, shuffle=False, Odorwise=False,
                                                                            MinResponses=10, trials=24, samples=50, samplestest=20,
                                                                            force_restimate=False, Stims=InterestingOdors, nt=1)

print(ConsideredUnits)

Accuracy_Shuff, Time_Shuff, ConsideredUnits = Classifier.Classifier(DfCorr, KernelClassifier, TWClassifier, TWBaselOdor, shuffle=True, Odorwise=False,
                                                                    MinResponses=10, trials=24, samples=50, samplestest=20,
                                                                    force_restimate=False, Stims=InterestingOdors, nt=1)

print(ConsideredUnits)



# figure params

plt.rcParams["font.family"] = "arial"
cm = 1/2.54
TickLength = 2.5
TLS = 6    # TickLabelSize
LS = 7       # LabelSize
TS = 8       # TitleSize
xlimClose = [0, 1.15]
LetterSize = 12


fig = plt.figure(figsize=(11.5*cm, 14.5*cm))

# Sub1
sub1 = fig.add_subplot(4, 2, 1)
sub1.plot([-50, 80], [-50, 80], color='k', linestyle='-', linewidth=0.5)
sub1.plot(SPCountMatrixMLR0, SPCountMatrixMLR1, '.', color=NeuroColor['Both'], markersize=2.5, label='unit')
sub1.plot(SPCountMatrixMLR0[UnitOfInterest], SPCountMatrixMLR1[UnitOfInterest], marker='s', color=NeuroColor['Both'],
          markersize=3, label='unit')
sub1.set_xlabel('# no MLR', size=LS, labelpad=-6)
sub1.set_ylabel('# MLR', size=LS, labelpad=-6)
sub1.set_xlim(-6.5, 50)
sub1.set_ylim(-6.5, 50)
sub1.set_xticks((0, 10, 20, 30, 40, 50))
sub1.set_xticklabels(('0', '', '', '', '', '50'))
sub1.set_yticks((0, 10, 20, 30, 40, 50))
sub1.set_yticklabels(('0', '', '', '', '', '50'))
sub1.tick_params(axis='both', labelsize=TLS, length=TickLength)
sub1.spines["right"].set_visible(False)
sub1.spines["top"].set_visible(False)
sub1.set_aspect('equal')
l1, b1, w1, h1 = sub1.get_position().bounds
sub1.set_position([l1-0.0125, b1, w1, h1])
sub1.text(-30, 52.75, 'A', weight='bold', size=LetterSize)

# Sub2
sub2 = fig.add_subplot(4, 2, 3)
sub2.plot(SpRaMLR1UnitOfInt, linewidth=0.9, color=NeuroColor['MLR'])
sub2.plot(SpRaMLR0UnitOfInt, linewidth=0.9, color=NeuroColor['NoMLR'])
sub2.axhline(Bound[UnitOfInterest][1], color='grey', linestyle='--',linewidth=0.9)
sub2.set_ylabel('Rate [Hz]',size=LS,labelpad=-3.5)
sub2.spines["right"].set_visible(False)
sub2.spines["top"].set_visible(False)
sub2.tick_params(axis='both', labelsize=TLS,length=TickLength)
sub2.set_xticks(RateTimes)
sub2.set_xticklabels(RateTimeLabels,rotation=0)
sub2.set_yticks((0,10,20,30))
sub2.set_yticklabels(('0','','','30'))
sub2.legend(['MLR','no MLR'], fontsize=TLS, loc="upper right", frameon=False, handlelength=1, labelspacing=0.2)
l2, b2, w2, h2 = sub2.get_position().bounds
l2 = l2 - 0.025
w2 = w2 + 0.02
sub2.set_position([l2, b2, w2, h2])
sub2.set_xlim(0, 2000)
sub2.set_ylim(-2, 30)
sub2.text(-370, 31.5, 'B', weight='bold', size=LetterSize)

# Sub3
sub3 = fig.add_subplot(4, 2, 5)
sub3.fill_between(RateTimesOG, MeanSpRa0-StErSpRa0, MeanSpRa0+StErSpRa0, facecolor=NeuroColorShadow['NoMLR'])
sub3.plot(RateTimesOG, MeanSpRa0, linewidth=0.9,color=NeuroColor['NoMLR'])
sub3.fill_between(RateTimesOG, MeanSpRa1-StErSpRa1, MeanSpRa1+StErSpRa1, facecolor=NeuroColorShadow['MLR'])
sub3.plot(RateTimesOG, MeanSpRa1, linewidth=0.9, color=NeuroColor['MLR'])
sub3.plot(SigTime, 0.85, marker='v', clip_on=False, color='k', markersize=2)
sub3.plot((SigTime, SigTime), (-1, 0.8), '--k', linewidth=0.6)
sub3.text(SigTime-45, 0.9, str(SigTime)+'ms', size=6)
sub3.set_ylim(-0.5, 1)
sub3.set_xlabel('Time [s]', size=LS)
sub3.set_ylabel('Norm. rate', size=LS, labelpad=0)
sub3.spines["right"].set_visible(False)
sub3.spines["top"].set_visible(False)
sub3.set_yticks((0, 0.25, 0.5, 0.75, 1))
sub3.set_yticklabels(('0', '', '', '', '1'))
sub3.tick_params(axis='both', labelsize=TLS, length=TickLength)
sub3.set_xticks(RateTimes)
sub3.set_xticklabels(RateTimeLabels, rotation=0)
l3, b3, w3, h3 = sub3.get_position().bounds
l3 = l2
w3 = w2
sub3.set_position([l3, b3, w3, h3])
sub3.set_xlim(0, 2000)
sub3.set_ylim(-0.03, 1.02)
sub3.text(-370, 1.07, 'C', weight='bold', size=LetterSize)

# sub4
sub4 = fig.add_subplot(4,2,7)
sub4.plot(Transform.PCA[5][BaseInd, 0][0], Transform.PCA[5][BaseInd, 1][0], color='darkgrey', linewidth=0.9)    # StimG
sub4.plot(Transform.PCA[1][BaseInd, 0][0], Transform.PCA[1][BaseInd, 1][0], color='darkgrey', linewidth=0.9)    # StimA
sub4.plot(Transform.PCA[3][BaseInd, 0][0], Transform.PCA[3][BaseInd, 1][0], color='darkgrey', linewidth=0.9)    # StimC

sub4.plot(Transform.PCA[4][BaseInd, 0][0], Transform.PCA[4][BaseInd, 1][0], color='darkgrey', linewidth=0.9)    # StimG
sub4.plot(Transform.PCA[0][BaseInd, 0][0], Transform.PCA[0][BaseInd, 1][0], color='darkgrey', linewidth=0.9)    # StimA
sub4.plot(Transform.PCA[2][BaseInd, 0][0], Transform.PCA[2][BaseInd, 1][0], color='darkgrey', linewidth=0.9)    # StimC

sub4.plot(Transform.PCA[5][StimInd, 0][0], Transform.PCA[5][StimInd, 1][0],
          color=OdorColorMLR['G'][0], linewidth=0.9, label='Cin')
sub4.plot(Transform.PCA[1][StimInd, 0][0], Transform.PCA[1][StimInd, 1][0],
          color=OdorColorMLR['A'][0], linewidth=0.9, label='Ben')
sub4.plot(Transform.PCA[3][StimInd, 0][0], Transform.PCA[3][StimInd, 1][0],
          color=OdorColorMLR['C'][0], linewidth=0.9, label='Iso')

sub4.plot(Transform.PCA[4][StimInd, 0][0], Transform.PCA[4][StimInd, 1][0],
          color=OdorColorMLR['G'][0], linewidth=0.9, label='Cin', linestyle=(0, (1, 0.8)))
sub4.plot(Transform.PCA[0][StimInd, 0][0], Transform.PCA[0][StimInd, 1][0],
          color=OdorColorMLR['A'][0], linewidth=0.9, label='Ben', linestyle=(0, (1, 0.8)))
sub4.plot(Transform.PCA[2][StimInd, 0][0], Transform.PCA[2][StimInd, 1][0],
          color=OdorColorMLR['C'][0], linewidth=0.9, label='Iso', linestyle=(0, (1, 0.8)))


sub4.set_xlabel('PC1 ('+str(Transform['explained_variance_ratio_'][0][0]*100)[:4]+' %)', size=LS)
sub4.set_ylabel('PC2 ('+str(Transform['explained_variance_ratio_'][0][1]*100)[:4]+' %)', size=LS)
sub4.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub4.legend(fontsize=TLS, bbox_to_anchor=(0.37, 1.03), loc='upper left', borderaxespad=0, frameon=False, ncol=2,
            handlelength=1, title='MLR        no MLR', title_fontsize=TLS+0.5, labelspacing=0.2)
sub4.spines["right"].set_visible(False)
sub4.spines["top"].set_visible(False)
sub4.set_xticklabels([])
sub4.set_yticklabels([])
sub4.set_xticks([])
sub4.set_yticks([])
sub4.text(-69, 55, 'D', weight='bold', size=LetterSize)
l4, b4, w4, h4 = sub4.get_position().bounds
l4 = l2
w4 = w2
b4 = b4-0.05
sub4.set_position([l4, b4, w4, h4])

# sub5
sub5 = fig.add_subplot(4, 2, 2)
Latency.PlotLatencyCDF(LatencyDF[LatencyDF.StimID == 'G'], title='', color=OdorColorMLR['G'], TWOdor=[0, 2], ax=sub5)
sub5.set_xlabel('', size=LS)
sub5.set_ylabel('CDF', size=LS, labelpad=0)
sub5.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub5.set_yticks((0, 0.25, 0.5, 0.75, 1))
sub5.set_yticklabels(('0', '', '', '', '1'))
sub5.legend(fontsize=TLS, loc="lower right", frameon=False, title='Cin', title_fontsize=TLS+0.5, handlelength=1,
            labelspacing=0.2)
sub5.spines["right"].set_visible(False)
sub5.spines["top"].set_visible(False)
sub5.set_xlim(0, 2)
sub5.set_ylim(-0.02, 1.02)
l5, b5, w5, h5 = sub5.get_position().bounds
l5 = l5+0.06
w5 = w2
sub5.set_position([l5, b5, w5, h5])
sub5.text(-0.42, 1.07, 'E', weight='bold', size=LetterSize)

# sub6
sub6 = fig.add_subplot(4, 2, 4)
Latency.PlotLatencyCDF(LatencyDF[LatencyDF.StimID == 'A'], title='', color=OdorColorMLR['A'], TWOdor=[0, 2], ax=sub6)
sub6.set_xlabel('', size=LS)
sub6.set_ylabel('CDF', size=LS, labelpad=0)
sub6.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub6.set_yticks((0, 0.25, 0.5, 0.75, 1))
sub6.set_yticklabels(('0', '', '', '', '1'))
sub6.legend(fontsize=TLS, loc="lower right", frameon=False, title='Ben',title_fontsize=TLS+0.5, handlelength=1,
            labelspacing=0.2)
sub6.spines["right"].set_visible(False)
sub6.spines["top"].set_visible(False)
sub6.set_xlim(0, 2)
sub6.set_ylim(-0.02, 1.02)

l6, b6, w6, h6 = sub6.get_position().bounds
l6 = l5
w6 = w5
sub6.set_position([l6, b2, w6, h6])
sub6.text(-0.42, 1.07, 'F', weight='bold', size=LetterSize)

# sub7
sub7 = fig.add_subplot(4, 2, 6)
Latency.PlotLatencyCDF(LatencyDF[LatencyDF.StimID == 'C'], title='', color=OdorColorMLR['C'], TWOdor=[0, 2], ax=sub7)
sub7.set_xlabel('', size=LS)
sub7.set_ylabel('CDF', size=LS, labelpad=0)
sub7.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub7.set_yticks((0, 0.25, 0.5, 0.75, 1))
sub7.set_yticklabels(('0', '', '', '', '1'))
sub7.legend(fontsize=TLS, loc="lower right", frameon=False, title='Iso', title_fontsize=TLS+0.5, handlelength=1,
            labelspacing=0.2)
sub7.spines["right"].set_visible(False)
sub7.spines["top"].set_visible(False)
sub7.set_xlim(0, 2)
sub7.set_ylim(-0.02, 1.02)
l7, b7, w7, h7 = sub7.get_position().bounds
l7 = l5
w7 = w5
sub7.set_position([l7, b3, w7, h7])
sub7.text(-0.42, 1.07, 'G', weight='bold', size=LetterSize)

# sub8
sub8 = fig.add_subplot(4, 2, 8)
sub8.plot(np.array(Time_Shuff)/1000, np.mean(Accuracy_Shuff, axis=1), color=(0.8, 0.77, 0.79, 1), linewidth=0.9, label="Shuffled")
sub8.plot(np.array(ClassifierTime)/1000, np.mean(ClassifierAccuracy, axis=1), color='k', linewidth=0.9, label="Classifier")
sub8.axhline(y=0.5, color='grey', linestyle='--', linewidth=0.9)
sub8.set_xlabel('Time [s]', size=LS)
sub8.set_ylabel('Accuracy [%]', size=LS, labelpad=-4)
sub8.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub8.set_yticks((0.2, 0.4, 0.6, 0.8, 1))
sub8.set_yticklabels(('20', '', '', '', '100'))
sub8.legend(fontsize=TLS, loc="lower center", bbox_to_anchor=(0.5,-0.05),frameon=False, handlelength=1,
            labelspacing=0.2, ncol=2)
l8, b8, w8, h8 = sub8.get_position().bounds
l8 = l7
w8 = w5
sub8.set_position([l8, b4, w8, h8])
sub8.spines["right"].set_visible(False)
sub8.spines["top"].set_visible(False)
sub8.set_xlim(xlimClose[0], 2)
sub8.set_ylim(0.1, 1.01)
sub8.text(-0.42, 1.045, 'H', weight='bold', size=LetterSize)
plt.savefig(os.path.join('Figures', 'Figure4.png'), dpi=300)

plt.show()


# save maxima to pickle data

with open("UnitMaxima.pkl", 'wb') as f:
    pickle.dump(Maxima, f)

