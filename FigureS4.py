from Functions import Dataload, Latency, HelperFunctions
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
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
TWOdor = [0, 2]
TWPlot = [0.0, 2.2]
TWBaselOdor = [-20.0, -0.5]
TWMLR = [0.0, 2.0]  # TW for MLR DfCorr
TWStimulation = [0.0, 2.0]
NewUnitCodes = {4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19,
                18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 26: 29, 27: 30, 28: 31}
OdorCodes = ['J', 'F', 'H', 'K', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
OdorNames = ['1-Pen', '1-Hex', '1-Hep', '1-Oct', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Con']
file_name = "MLR_data.xlsx"     # name for Excel file
sheet = "Roh"
AnimalsExclude = ["CA70", "CA69", "CA71"]  # exclude because of too low frame rate
DataFrame = Dataload.GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=OdorShift)
DataFrameMLR = pd.read_excel(io=os.path.join(Path, file_name), sheet_name=sheet)       # create dataframe from Excel file
DataFrame.drop(DataFrame[[x[0:4] in AnimalsExclude for x in DataFrame['AnimalID']]].index,
        inplace=True)
DataFrame.reset_index(drop=True, inplace=True)
DfCorr, _ = Dataload.MergeNeuronal_MLR(DataFrame, DataFrameMLR, TWMLR=TWMLR, T0=OdorShift)
DfCorr['RealUnit'] = DfCorr['RealUnit'].apply(lambda x: NewUnitCodes[x])

# calculate single trial latencies for subplot 1
UseSingeTrialBL = True
BorderLim = 0.97
MinSpike = 2.25
TauFR = 250
WidthFactor = 5
dt = 1.
kernel = HelperFunctions.exponential_kernel(TauFR, dt=dt, nstd=WidthFactor)

Latencydf = Latency.Latency(DfCorr, kernel, TWStimulation, TWBaselOdor, MinSpike=MinSpike, Border=BorderLim,
                            Single_Trial_BL=UseSingeTrialBL, Stims=['A', 'C', 'G'])
Latencydf = Latencydf[Latencydf['MLR'] == True]
Latencydf = Dataload.LimitDFtoStimulus(Latencydf, ['C'])
Sort = Latencydf.groupby('RealUnit').agg({'NeuronalOnset': np.median}).sort_values('NeuronalOnset').reset_index()
SortedUnit = Sort['RealUnit'].tolist()
print(Latencydf.groupby(['StimID']))
print(Latencydf.groupby('StimID').agg({'NeuronalOnset': [np.mean, np.min, np.max]}))

# calculate pooled latencies for subplot 2

PooledLatencydf = Latency.Latency_pooled(DfCorr, kernel, TWStimulation, TWBaselOdor, MinSpike=MinSpike,
                                         Border=BorderLim, Stims=['A', 'C', 'G'])
PooledLatencydf = Dataload.LimitDFtoStimulus(PooledLatencydf, ['C'])

df1 = pd.DataFrame(np.array(PooledLatencydf['MLRTime'])).assign(Type='MLR')
df2 = pd.DataFrame(np.array(PooledLatencydf['NeuronalOnset'])).assign(Type='Neuronal')
cdf = pd.concat([df2, df1])
LatencyCon = pd.melt(cdf, id_vars=['Type'], var_name=['Number'])
print(LatencyCon)
LatencyCon = LatencyCon.drop(labels=None, axis=0, index=[15, 16])
print(LatencyCon.groupby('Type').agg(np.median))

plt.rcParams["font.family"] = "arial"
cm = 1/2.54
TickLength = 2.5
TLS = 6    # TickLabelSize
LS = 7       # LabelSize
TS = 8       # TitleSize
xlimClose = [0, 1.15]
LetterSize = 12
lw = 0.8

# Figure
fig = plt.figure(figsize=(10*cm, 5*cm))

# sub1
sub1 = fig.add_subplot(1, 2, 1)
sub1 = sns.boxplot(data=Latencydf, ax=sub1, x="RealUnit", y="NeuronalOnset", color="white", order=SortedUnit,
                   medianprops=dict(color=(0.753, 0.086, 0.086), alpha=1, linewidth=lw),
                   boxprops=dict(facecolor='lightgrey', edgecolor='k', linewidth=lw),
                   whiskerprops=dict(color='k', linewidth=lw), capprops=dict(color='k', linewidth=lw),
                   flierprops=dict(markerfacecolor="grey", marker="o", markersize=1, markeredgecolor='grey'))
sub1.spines["right"].set_visible(False)
sub1.spines["top"].set_visible(False)
sub1.set_xlabel("Unit", fontsize=LS)
sub1.set_ylabel("Latency [s]", fontsize=LS)
l1, b1, w1, h1 = sub1.get_position().bounds
l1 = l1+0.01
b1 = b1+0.03
w1 = w1+0.22
sub1.set_position([l1, b1, w1, h1])
sub1.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub1.set_ylim([-0.05, 1.8])
sub1.text(-4.5, 1.85, 'A', size=LS, weight='bold')

# fig2
sub2 = fig.add_subplot(1, 2, 2)
sub2 = sns.boxplot(data=LatencyCon, ax=sub2, x='Type', y='value', color="white",
                   medianprops=dict(color=(0.753, 0.086, 0.086), alpha=1, linewidth=lw),
                   boxprops=dict(facecolor='lightgrey', edgecolor='k', linewidth=lw),
                   whiskerprops=dict(color='k', linewidth=lw), capprops=dict(color='k', linewidth=lw),
                   flierprops=dict(markerfacecolor="grey", marker="o", markersize=1, markeredgecolor='grey'))
sub2.spines["right"].set_visible(False)
sub2.spines["top"].set_visible(False)
sub2.set_xlabel("", fontsize=LS)
sub2.set_ylim([-0.05, 1.8])
sub2.set_yticklabels("", fontsize=TLS)
sub2.text(0.24, 1.1, '***', size=12)
sub2.tick_params(axis='both', length=TickLength, labelsize=TLS)
sub2.set_ylabel("", fontsize=LS)
l2, b2, w2, h2 = sub2.get_position().bounds
w2 = w2-0.2
l2 = l2+0.22
sub2.set_position([l2, b1, w2, h2])
sub2.text(-1, 1.85, 'B', size=LS, weight='bold')


fig.savefig(os.path.join('Figures', 'FigS4.png'), dpi=300)
plt.show()
