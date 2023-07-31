from Functions import Dataload
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
TWOdor = [-3., 5]
TWPlot = [0.0, 2.2]
TWBaselOdor = [-20.0, -0.5]
TWMLR = [0.0, 2.0]  # TW for MLR DfCorr
TWStimulation = [0.0, 2.0]
OdorCodes = ['F', 'H', 'K', 'J', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
OdorNames = ['1-Hex', '1-Hep', '1-Oct', '1-Pen', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Ctr']
FoodOdors = ['A', 'C', 'G']
NonFoodOdors = ['F', 'H', 'K', 'J', 'I', 'D', 'B']

file_name = "MLR_data.xlsx"     # name for Excel file
sheet = "Roh"

DataFrame = Dataload.GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=OdorShift)
DataFrameMLR = pd.read_excel(io=os.path.join(Path, file_name), sheet_name=sheet)       # create dataframe from Excel file
DfCorr, _ = Dataload.MergeNeuronal_MLR(DataFrame, DataFrameMLR, TWMLR=TWMLR, T0=OdorShift)
StimDur = TWStimulation[1]-TWStimulation[0]
DfCorr['CountStim'] = DfCorr.StimSpikeTimes.apply(lambda x: len(x[(x>=TWMLR[0])&(x<=TWMLR[1])]))
BaselDur = TWBaselOdor[1]-TWBaselOdor[0]
DfCorr['CountBaselAdj'] = DfCorr.BaselSpikeTimes.apply(lambda x: len(x)/BaselDur*StimDur)
DfCorr['AdjustedCount'] = DfCorr['CountStim'] - DfCorr['CountBaselAdj']
DfCorrNonFood = Dataload.LimitDFtoStimulus(DfCorr, NonFoodOdors)
DfCorrFood = Dataload.LimitDFtoStimulus(DfCorr, FoodOdors)
DfMLR1 = DfCorrFood[DfCorrFood.MLR == 1]
DfMLR0 = DfCorrFood[DfCorrFood.MLR == 0]


plt.rcParams["font.family"] = "arial"
cm = 1/2.54

TickLength = 2.5
xShift = 0.09
TLS = 6    # TickLabelSize
LS = 7       # LabelSize
TS = 8       # TitleSize
lw = 0.8

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16*cm, 5*cm), sharey=True)

# ax1
sns.boxplot(data=DfCorrNonFood, x="Trial", y="AdjustedCount", color="white", ax=ax1,
            medianprops=dict(color=(0.753, 0.086, 0.086), alpha=1, linewidth=lw),
            boxprops=dict(facecolor='lightgrey', edgecolor='k', linewidth=lw),
            whiskerprops=dict(color='k', linewidth=lw), capprops=dict(color='k', linewidth=lw),
            flierprops=dict(markerfacecolor="grey", marker="o", markersize=0.5, markeredgecolor='grey'))

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.set_xlabel("Trial", fontsize=LS)
ax1.set_ylabel("Spike count", fontsize=LS)
ax1.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
ax1.tick_params(axis='both', length=TickLength, labelsize=TLS)
ax1.set_ylim(-62, 210)

l1, b1, w1, h1 = ax1.get_position().bounds
l1 = l1-0.04
b1 = b1+0.05
w1 = w1+0.03
ax1.set_position([l1, b1, w1, h1])
ax1.text(-3.3, 215, 'A', size=LS, weight='bold')

# ax2

sns.boxplot(data=DfMLR0, x="Trial", y="AdjustedCount", color="white", ax=ax2,
            medianprops=dict(color=(0.753, 0.086, 0.086), alpha=1, linewidth=lw),
            boxprops=dict(facecolor='lightgrey', edgecolor='k', linewidth=lw),
            whiskerprops=dict(color='k', linewidth=lw), capprops=dict(color='k',linewidth=lw),
            flierprops=dict(markerfacecolor="grey", marker="o", markersize=0.5, markeredgecolor='grey'))

ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_xlabel("Trial", fontsize=LS)
ax2.set_ylabel("", fontsize=LS)
ax2.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
ax2.tick_params(axis='both', length=TickLength, labelsize=TLS)
ax2.set_ylim(-62,210)
l2, b2, w2, h2 = ax2.get_position().bounds
l2 = l1+w1+0.04
ax2.set_position([l2, b1, w1, h2])
ax2.text(-1.7, 215, 'B', size=LS, weight='bold')


# ax3

sns.boxplot(data=DfMLR1, x="Trial",y="AdjustedCount", color="white", ax=ax3,
            medianprops=dict(color=(0.753, 0.086, 0.086), alpha=1, linewidth=lw),
            boxprops=dict(facecolor='lightgrey',edgecolor='k',linewidth=lw), whiskerprops=dict(color='k', linewidth=lw),
            capprops=dict(color='k', linewidth=lw), flierprops=dict(markerfacecolor="grey", marker="o", markersize=0.5,
                                                                    markeredgecolor='grey'))
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.set_xlabel("Trial", fontsize=LS)
ax3.set_ylabel("", fontsize=LS)
ax3.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
ax3.tick_params(axis='both', length=TickLength, labelsize=TLS)
ax3.set_ylim(-62, 210)
l3, b3, w3, h3 = ax3.get_position().bounds
l3 = l2+w1+0.04
ax3.set_position([l3, b1, w1, h3])
ax3.text(-1.7, 215, 'C', size=LS, weight='bold')

fig.savefig(os.path.join('Figures', 'FigS2.jpg'), dpi=300)
plt.show()
