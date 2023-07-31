from Functions import Dataload, HelperFunctions, EuclidianDistance
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
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
TWPlot = [0.0, 2.2]
TWBaselOdor = [-20.0, -0.5]
TWMLR = [0.0, 2.0]  # TW for MLR DfCorr
TWStimulation = [0.0, 2.0]
sigmaFR = 60

OdorCodes = ['F', 'H', 'K', 'J', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
OdorNames = ['1-Hex', '1-Hep', '1-Oct', '1-Pen', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Ctr']
FoodOdors = ['A', 'C', 'G']
file_name = "MLR_data.xlsx"     # name for Excel file
sheet = "Roh"
OdorColorMLR = {'C': ((0.765, 0.588, 0.09, 1), (0.914, 0.737, 0.247, 1),(0.95, 0.847, 0.573, 1)),
                'G': ((0.631, 0.114, 0.027, 1), (0.863, 0.522, 0.463, 1), (0.902, 0.659, 0.62, 1)),
                'A': ((0.008, 0.251, 0.455, 1), (0.325, 0.525, 0.694, 1), (0.537, 0.675, 0.788, 1))}
NeuroColor = {'MLR': (0.102, 0.416, 0.278), 'NoMLR': (0.502, 0.675, 0.6), 'Both': (0.298, 0.5411, 0.439, 1)}
OdorMix = {'IsBe': (0.1, 0.1, 0.1, 1), 'IsCi': (0.5, 0.5, 0.5, 1), 'BeCi': (0.7, 0.7, 0.7, 1)}

DataFrame = Dataload.GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=OdorShift)
DataFrameMLR = pd.read_excel(io=os.path.join(Path, file_name), sheet_name=sheet)       # create dataframe from Excel file
DfCorr, _ = Dataload.MergeNeuronal_MLR(DataFrame, DataFrameMLR, TWMLR=TWMLR, T0=OdorShift)
DfCorr = Dataload.LimitDFtoStimulus(DfCorr, FoodOdors)

# Firing rates of single odors for subplot 1-3

# Load unit maxima from Figure4
try:
    with open("UnitMaxima.pkl", 'rb') as f:
        Maxima = pickle.load(f)
except FileNotFoundError:
    print("File not found, please run Figure4.py first")
    raise

#Maxima = {4: 34.563429239543325, 5: 3.504108503812676, 6: 7.4513089046702765, 7: 13.582322125948838, 8: 83.18571139243342,
# 9: 11.712228497724277, 10: 28.796190913727195, 11: 23.299364831206475, 12: 26.57745467630763, 13: 5.7757642496715205,
# 14: 18.58274246517002, 15: 23.172658555730006, 16: 40.539778106957584, 17: 17.396946249329922, 18: 13.181685652193018,
# 19: 15.198913655127601, 20: 8.456442793284545, 21: 10.201569929549226, 22: 22.27826535008664, 23: 12.430281081252522,
# 26: 4.9805199614862765, 27: 37.35269474048714, 28: 11.832082965549327}  # maximal firing rates per unit calculated in
# Figure4.py, needed for normalizing firing rate for each unit
kernel = HelperFunctions.alpha_kernel(sigmaFR, 1.0, calibrateMass=True)
Edge = sigmaFR * 3
tMin = (TWPlot[0] * 1000) - Edge
tMax = (TWPlot[1] * 1000) + Edge
t = tMax - tMin
Dur = int((TWPlot[1] - TWPlot[0]) * 1000)
UniqueRealUnits = np.unique(DfCorr['RealUnit'])
SpRaMatrix0 = np.zeros((len(UniqueRealUnits), Dur))
SpRaMatrix1 = np.zeros((len(UniqueRealUnits), Dur))
MeanSpRa0SeperateOdors = np.zeros((len(FoodOdors), Dur))
MeanSpRa1SeperateOdors = np.zeros((len(FoodOdors), Dur))
StErSpRa0 = np.zeros((len(FoodOdors), Dur))
StErSpRa1 = np.zeros((len(FoodOdors), Dur))
SingleUnitSpRa0SeperateOdors = np.zeros((len(FoodOdors), Dur))
SingleUnitSpRa1SeperateOdors = np.zeros((len(FoodOdors), Dur))

for OdorNum, Odor in enumerate(FoodOdors):
    DfStim = DfCorr[DfCorr['StimID'] == Odor]  # Df for each Odor

    for idy, Unit in enumerate(UniqueRealUnits):  # loops over each Unit
        DfCorrUnit = DfStim[DfStim['RealUnit'] == Unit]  # Df for each Unit
        DfCorrUnitMLR0 = DfCorrUnit[DfCorrUnit.MLR == 0]
        DfCorrUnitMLR1 = DfCorrUnit[DfCorrUnit.MLR == 1]
        LenStimMLR0 = np.array([])
        LenStimMLR1 = np.array([])
        LenBaselMLR0 = np.array([])
        LenBaselMLR1 = np.array([])
        SpTStimMLR0 = np.array([])
        SpTStimMLR1 = np.array([])
        SpTBaselMLR0 = np.array([])
        SpTBaselMLR1 = np.array([])
        TrialsStimMLR0ST = np.array([])
        TrialsStimMLR1ST = np.array([])
        TrialsBaselMLR0ST = np.array([])
        TrialsBaselMLR1ST = np.array([])
        SpTTrStimMLR0 = np.array([])
        SpTTrStimMLR1 = np.array([])
        SpTTrBaselMLR0 = np.array([])
        SpTTrBaselMLR1 = np.array([])
        n = 0
        m = 0
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
        SpTStimMLR0 = np.hstack((SpTStimMLR0, tMax + 2))
        SpTStimMLR1 = np.hstack((SpTStimMLR1, tMax + 2))
        SpTBaselMLR0 = np.hstack((SpTBaselMLR0, TWBaselOdor[1] + 2))
        SpTBaselMLR1 = np.hstack((SpTBaselMLR1, TWBaselOdor[1] + 2))
        TrialsStimMLR0ST = np.hstack((TrialsStimMLR0ST, m))
        TrialsStimMLR1ST = np.hstack((TrialsStimMLR1ST, n))
        TrialsBaselMLR0ST = np.hstack((TrialsBaselMLR0ST, m))
        TrialsBaselMLR1ST = np.hstack((TrialsBaselMLR1ST, n))
        SpTTrStimMLR0 = np.vstack([SpTStimMLR0, TrialsStimMLR0ST])
        SpTTrStimMLR1 = np.vstack([SpTStimMLR1, TrialsStimMLR1ST])
        SpTTrBaselMLR0 = np.vstack([SpTBaselMLR0, TrialsBaselMLR0ST])
        SpTTrBaselMLR1 = np.vstack([SpTBaselMLR1, TrialsBaselMLR1ST])
        SpRaStimMLR0 = HelperFunctions.kernel_rate(SpTTrStimMLR0, kernel, tlim=[tMin, tMax], pool=True)
        SpRaBaselMLR0 = HelperFunctions.kernel_rate(SpTTrBaselMLR0, kernel, tlim=[TWBaselOdor[0] * 1000,
                                                                                  TWBaselOdor[1] * 1000], pool=True)
        SpRaMLR0 = SpRaStimMLR0[0][0] - np.average(SpRaBaselMLR0[0])
        SpRaStimMLR1 = HelperFunctions.kernel_rate(SpTTrStimMLR1, kernel, tlim=[tMin, tMax], pool=True)
        SpRaBaselMLR1 = HelperFunctions.kernel_rate(SpTTrBaselMLR1, kernel, tlim=[TWBaselOdor[0] * 1000,
                                                                                  TWBaselOdor[1] * 1000], pool=True)
        SpRaMLR1 = SpRaStimMLR1[0][0] - np.average(SpRaBaselMLR1[0])
        NormSpRaMLR0 = SpRaMLR0 / Maxima[Unit]
        NormSpRaMLR1 = SpRaMLR1 / Maxima[Unit]
        if sum(NormSpRaMLR1) == 0:
            NormSpRaMLR1[NormSpRaMLR1 == 0] = 'nan'

        if sum(NormSpRaMLR0) == 0:
            NormSpRaMLR0[NormSpRaMLR0 == 0] = 'nan'

        SpRaMatrix0[idy] = NormSpRaMLR0
        SpRaMatrix1[idy] = NormSpRaMLR1

    MeanSpRa0perOdor = np.nanmean(SpRaMatrix0, axis=0)
    MeanSpRa1perOdor = np.nanmean(SpRaMatrix1, axis=0)
    MeanSpRa0SeperateOdors[OdorNum] = MeanSpRa0perOdor
    NUnits = np.count_nonzero(~np.isnan(SpRaMatrix0[:, 0]))
    MeanSpRa1SeperateOdors[OdorNum] = MeanSpRa1perOdor
    StErSpRa0[OdorNum] = np.nanstd(SpRaMatrix0, axis=0, ddof=1) / np.sqrt(NUnits)
    StErSpRa1[OdorNum] = np.nanstd(SpRaMatrix1, axis=0, ddof=1) / np.sqrt(NUnits)

RateTimesOG = SpRaStimMLR1[1]
RateTimes = SpRaStimMLR1[1]-SpRaStimMLR1[1][0]
RateTimes = np.arange(RateTimes[0], RateTimes[-1], 500)
RateTimeLabels = np.arange(SpRaStimMLR1[1][0] + 0.5, SpRaStimMLR1[1][-1] + 0.5, 500) / 1000

# Euclidean distance for subplot 4&5
TauFR = 60  # defines sigma for gaussian kernel (half of gaussian width)
WidthFactor = 3
dt = 1.
KernelEuclidean = HelperFunctions.alpha_kernel(TauFR, dt=dt, nstd=WidthFactor)
TWEuc = [-2.6, 4.6]
EucTime, EucDistance, EucFRsNorm = EuclidianDistance.Euclidian_MLR(DfCorr, KernelEuclidean, TWEuc, TWBaselOdor,
                                                                   normalize=True, Norm='L2', MinResponses=2,
                                                                   Stims=['A', 'C', 'G'])
OdorCompare = EuclidianDistance.Euclidian_Odors(DfCorr, KernelEuclidean, TWEuc, TWBaselOdor, FRsNorm=EucFRsNorm,
                                                normalize=True, Norm='L2', MinResponses=2, Stims=['A', 'C', 'G'],
                                                MLR=False)
EucMeanBasel = np.mean([np.mean(OdorCompare.Distance[0][np.array(OdorCompare.Time[0]) < -500]),
                        np.mean(OdorCompare.Distance[1][np.array(OdorCompare.Time[1]) < -500]),
                        np.mean(OdorCompare.Distance[2][np.array(OdorCompare.Time[2]) < -500])])
# figure params

plt.rcParams["font.family"] = "arial"
cm = 1/2.54
TickLength = 2.5
TLS = 6    # TickLabelSize
LS = 7       # LabelSize
TS = 8       # TitleSize
xlimClose = [0, 1.15]
LetterSize = 12

fig = plt.figure(figsize=(16*cm, 10*cm))

# sub1
sub1 = fig.add_subplot(2, 3, 1)
sub1.fill_between(RateTimesOG, MeanSpRa1SeperateOdors[2]-StErSpRa1[2], MeanSpRa1SeperateOdors[2]+StErSpRa1[2],
                  facecolor=OdorColorMLR['G'][2])
sub1.plot(RateTimesOG, MeanSpRa1SeperateOdors[2], linewidth=0.9, color=OdorColorMLR['G'][0], label='MLR')
sub1.fill_between(RateTimesOG, MeanSpRa0SeperateOdors[2]-StErSpRa0[2], MeanSpRa0SeperateOdors[2]+StErSpRa0[2],
                  facecolor=OdorColorMLR['G'][2])
sub1.plot(RateTimesOG, MeanSpRa0SeperateOdors[2], linewidth=0.9, color=OdorColorMLR['G'][0], label='no MLR',
          linestyle=(0, (1, 2)))
sub1.legend(fontsize=TLS, bbox_to_anchor=(0.64, 0.95), loc='upper left', borderaxespad=0, frameon=False, handlelength=1,
            title='Cin', title_fontsize=TLS+0.5, labelspacing=0.2)
sub1.set_ylabel('Normalized rate', size=LS, labelpad=0)
sub1.spines["right"].set_visible(False)
sub1.spines["top"].set_visible(False)
sub1.set_yticks((0, 0.25, 0.5, 0.75, 1))
sub1.set_yticklabels(('0', '', '', '', '1'))
sub1.tick_params(axis='both', labelsize=TLS, length=TickLength)
sub1.set_xticks(RateTimes)
sub1.set_xticklabels(RateTimeLabels, rotation=0)
l1, b1, w1, h1 = sub1.get_position().bounds
b1 = b1+0.05
w1 = w1+0.05
h1 = h1-0.015
sub1.set_position([l1-0.07, b1, w1, h1])
sub1.set_xlim(0, 2000)
sub1.set_ylim(-0.05, 1.1)
sub1.text(-360, 1.15, 'A', size=LS, weight='bold')

# sub2
sub2 = fig.add_subplot(2, 3, 2)
sub2.fill_between(RateTimesOG, MeanSpRa1SeperateOdors[0] - StErSpRa1[0], MeanSpRa1SeperateOdors[0] + StErSpRa1[0],
                  facecolor=OdorColorMLR['A'][2])
sub2.plot(MeanSpRa1SeperateOdors[0], linewidth=0.9, color=OdorColorMLR['A'][0], label='MLR')
sub2.fill_between(RateTimesOG, MeanSpRa0SeperateOdors[0] - StErSpRa0[0], MeanSpRa0SeperateOdors[0] + StErSpRa0[0],
                  facecolor=OdorColorMLR['A'][2])
sub2.plot(MeanSpRa0SeperateOdors[0], linewidth=0.9, color=OdorColorMLR['A'][0], label='no MLR', linestyle=(0, (1, 2)))
sub2.legend(fontsize=TLS, bbox_to_anchor=(0.64, 0.95), loc='upper left', borderaxespad=0, frameon=False, handlelength=1,
            title='Ben', title_fontsize=TLS+0.5, labelspacing=0.2)
sub2.spines["right"].set_visible(False)
sub2.spines["top"].set_visible(False)
sub2.set_yticks((0, 0.25, 0.5, 0.75, 1))
sub2.set_yticklabels((''))
sub2.tick_params(axis='both', labelsize=TLS,length=TickLength)
sub2.set_xticks(RateTimes)
sub2.set_xticklabels(RateTimeLabels,rotation=0)
l2, b2, w2, h2 = sub2.get_position().bounds
sub2.set_position([l2-0.02, b1, w1, h1])
sub2.set_xlim(0, 2000)
sub2.set_ylim(-0.05, 1.1)
sub2.text(-220, 1.15, 'B', size=LS, weight='bold')


# sub3
sub3 = fig.add_subplot(2, 3, 3)
sub3.fill_between(RateTimesOG, MeanSpRa1SeperateOdors[1] - StErSpRa1[1], MeanSpRa1SeperateOdors[1] + StErSpRa1[1],
                  facecolor=OdorColorMLR['C'][2])
sub3.plot(MeanSpRa1SeperateOdors[1], linewidth=0.9, color=OdorColorMLR['C'][0], label='MLR')
sub3.fill_between(RateTimesOG, MeanSpRa0SeperateOdors[1] - StErSpRa0[1], MeanSpRa0SeperateOdors[1] + StErSpRa0[1],
                  facecolor=OdorColorMLR['C'][2])
sub3.plot(MeanSpRa0SeperateOdors[1], linewidth=0.9, color=OdorColorMLR['C'][0], label='no MLR', linestyle=(0, (1, 2)))
sub3.legend(fontsize=TLS, bbox_to_anchor=(0.64, 0.95), loc='upper left', borderaxespad=0, frameon=False, handlelength=1,
            title='Iso', title_fontsize=TLS+0.5, labelspacing=0.2)
sub3.spines["right"].set_visible(False)
sub3.spines["top"].set_visible(False)
sub3.set_yticks((0, 0.25, 0.5, 0.75, 1))
sub3.set_yticklabels((''))
sub3.tick_params(axis='both', labelsize=TLS, length=TickLength)
sub3.set_xticks(RateTimes)
sub3.set_xticklabels(RateTimeLabels, rotation=0)
l3, b3, w3, h3 = sub3.get_position().bounds
sub3.set_position([l3+0.03, b1, w1, h1])
sub3.set_xlim(0, 2000)
sub3.set_ylim(-0.05, 1.10)
sub3.text(-220, 1.15, 'C', size=LS, weight='bold')

# sub4
sub4 = fig.add_subplot(2, 2, 3)
sub4.plot(np.array(EucTime)/1000, EucDistance, color=NeuroColor['Both'], linewidth=0.9, label='MLR vs no MLR')
sub4.axhline(np.mean(EucDistance[EucTime < -500]), color='grey', linestyle='--', linewidth=0.9)
sub4.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub4.set_xlabel('Time [s]', size=LS)
sub4.set_ylabel('Euclidean distance', size=LS)
sub4.set_ylim(0.0, 1.02)
sub4.set_xlim(-0.5, 2)
sub4.set_xticks((-0.5, 0, 0.5, 1, 1.5, 2))
sub4.spines["right"].set_visible(False)
sub4.spines["top"].set_visible(False)
sub4.legend(fontsize=TLS, loc="lower right", frameon=False)
l4, b4, w4, h4 = sub4.get_position().bounds
l4 = l4
sub4.set_position([l4, b4, w4, h1])
sub4.text(-1.02, 1.07, 'D', size=LS, weight='bold')


# sub5
sub5 = fig.add_subplot(2, 2, 4)

sub5.plot(np.array(OdorCompare.Time[0])/1000, OdorCompare.Distance[0], color=OdorMix['IsBe'], linewidth=0.9,
          label="Iso vs Ben")
sub5.plot(np.array(OdorCompare.Time[2])/1000, OdorCompare.Distance[2], color=OdorMix['IsCi'], linewidth=0.9,
          label="Iso vs Cin")
sub5.plot(np.array(OdorCompare.Time[1])/1000, OdorCompare.Distance[1], color=OdorMix['BeCi'], linewidth=0.9,
          label="Ben vs Cin")
sub5.axhline(EucMeanBasel, color='grey', linestyle='--', linewidth=0.9)
sub5.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub5.set_xlabel('Time [s]', size=LS)
sub5.set_ylim(0.0, 1.02)
sub5.set_xlim(-0.5, 2)
sub5.set_xticks((-0.5, 0, 0.5, 1, 1.5, 2))
sub5.set_yticklabels([''])
sub5.spines["right"].set_visible(False)
sub5.spines["top"].set_visible(False)
sub5.legend(fontsize=TLS, loc="upper right", frameon=False)
l5, b5, w5, h5 = sub5.get_position().bounds
l5 = l5+0.015
sub5.set_position([l5, b4, w4, h1])
sub5.text(-0.78, 1.07, 'E', size=LS, weight='bold')


plt.savefig(os.path.join('Figures', 'FigS3.svg'), dpi=300)
plt.show()
