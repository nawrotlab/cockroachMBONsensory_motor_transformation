import Dataload
import numpy as np
import matplotlib.pyplot as plt
import os                          # to change path
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import glob

TWOdor = [0.09, 2.09]
TWBaselOdor = [-20, -0.5]
TWMLR = [0.09, 2.09]
MLRAllMin = 0
BinSize = np.array([0.05])
OdorCodes = ['F', 'H', 'K', 'J', 'I', 'D', 'B', 'C', 'A', 'G', 'E']
OdorNames = ['1-Hex', '1-Hep', '1-Oct', '1-Pen', 'Hep', 'Oct', '2-Hep', 'Iso', 'Ben', 'Cin', 'Ctr']
file_name = "MLR_data.xlsx"     # name for Excel file
sheet = "Roh"
os.chdir("C:/Users/Cansu/Documents/Ephys_Auswertung/olfactory_visual/DataMLR")  # go to folder with data
DataFrameMLR = pd.read_excel(io=file_name, sheet_name=sheet)       # create dataframe from Excel file
Files = glob.glob("*.mat")     # opens all files in folder

DataFrame = Dataload.GenDF(Files, TWOdor, TWBaselOdor, OdorCodes, OdorNames, CorrectOdorOnset=0.09)
DFCorr, MLRAnimalSet = Dataload.GenDFBehavior(DataFrame, DataFrameMLR, OdorCodes, TWMLR, CorrectOdorOnset=0.09)

NAni = len(MLRAnimalSet)
NOdor = len(OdorCodes)
matrix = np.zeros((NOdor, NAni))

for ind, Odor in enumerate(OdorCodes):                      # loops through OdorCodes
    DfMLROdor = DFCorr[DFCorr['OdorID'] == Odor]            # generates DF for each odor from MLR DF

    for y, Animal in enumerate(MLRAnimalSet):                           # loops though AnimalIDs that exist in recordings and MLR data
        DfMLROdorAnimal = DfMLROdor[DfMLROdor['AnimalID'] == Animal]    # generates Dataframe: AnimalID from recordings should match MLR AnimalID
        MLRCol = DfMLROdorAnimal.loc[:, 'MLR']
        MLR = MLRCol.values
        Times = np.unique(DfMLROdorAnimal['MLRTime'])
        matrix[ind, y] = DfMLROdorAnimal.MLR.mean()*100

DropIndex = np.where(DFCorr['OdorID'] == 'E')                             # generate DF without control stimulus
DataFrameOdors = DFCorr.drop(labels=None, axis=0, index=DropIndex[0])

DropIndex = np.where(DataFrameOdors['AnimalID'] == 'CA69')                      # CA69 was recorded with 5 fps
DataFrameDrop = DataFrameOdors.drop(labels=None, axis=0, index=DropIndex[0])
DropIndex = np.where(DataFrameOdors['AnimalID'] == 'CA70')                      # CA70 was recorded with 5 fps
DataFrameDrop = DataFrameDrop.drop(labels=None, axis=0, index=DropIndex[0])
DropIndex = np.where(DataFrameOdors['AnimalID'] == 'CA71')                      # CA71 was recorded with 8 fps
DataFrameDrop = DataFrameDrop.drop(labels=None, axis=0, index=DropIndex[0])


cm = 1 / 2.54
TLS = 6     # TickLabelSize
LS = 7      # LabelSize
TS = 8      # TitleSize
LetterSize = 12     # Subplot letter fontsize
TickLength = 2.5    # Ticksize

plt.rcParams["font.family"] = "arial"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.5 * cm, 5 * cm),
                                    gridspec_kw={'width_ratios': [3, 1, 2.5]})  # generates one subplot in one figure
fig.subplots_adjust(wspace=0.02)
myColors = ((0.0, 0.35, 0.2, 0.0), (0.0, 0.35, 0.2, 0.2), (0.0, 0.35, 0.2, 0.3), (0.0, 0.35, 0.2, 0.4),
            (0.0, 0.35, 0.2, 0.5), (0.0, 0.35, 0.2, 0.6), (0.0, 0.35, 0.2, 0.7), (0.0, 0.35, 0.2, 0.8),
            (0.0, 0.35, 0.2, 0.9), (0.0, 0.35, 0.2, 1))

cmap = LinearSegmentedColormap.from_list('Custom', myColors, 11)

fig1 = sns.heatmap(matrix, ax=ax1, cmap=cmap, linewidths=.5, linecolor='black', square=False,
                   cbar_kws=dict(use_gridspec=False, location="left"))

colorbar = ax1.collections[0].colorbar
colorbar.set_ticks([5, 14, 23, 32, 41, 50, 59, 68, 77, 86, 95])
colorbar.set_ticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
colorbar.ax.tick_params(labelsize=TLS, length=TickLength)
colorbar.ax.set_frame_on(True)
for spine in colorbar.ax.spines.values():
    spine.set(visible=True, lw=0.5, edgecolor="black")
for spine in ax1.spines.values():
    spine.set(visible=True, lw=1, edgecolor="black")

colorbar.set_label('# MLR', size=LS, labelpad=-1)
xticklabel = range(1, len(MLRAnimalSet) + 1)
ax1.set_xticks((0.5, 3.5, 6.5, 9.5, 12.5, 15.5))
ax1.set_xticklabels((1, 4, 7, 10, 13, 16), size=TLS)
ax1.set_yticks((0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5))
ax1.set_yticklabels(OdorNames, size=TLS, rotation=30)
fig1.set_xlabel('Individual animal', size=LS)
ax1.text(-14.5, -0.1, 'A', weight="bold", size=LetterSize)
ax1.tick_params(axis='both', length=TickLength)

l1, b1, w1, h1 = ax1.get_position().bounds
b1 = b1 + 0.06
h1 = h1 - 0.06
ax1.set_position([l1, b1, w1, h1])
ll, bb, ww, hh = colorbar.ax.get_position().bounds
colorbar.ax.set_position([ll - 0.07, bb, ww, hh])

AverageResponse = np.sum(matrix, axis=1) / NAni
ax2.barh(OdorNames, AverageResponse, color=myColors[9], edgecolor='k', height=1)

for spine in ax2.spines.values():
    spine.set(visible=True, lw=1, edgecolor="black")

ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax2.axes.yaxis.set_visible(False)
ax2.set_xlim(0, 100)
ax2.set_ylim(-0.5, 10.5)
ax2.tick_params(axis='x', labelsize=TLS, length=TickLength)
ax2.set_xlabel('MLR [%]', size=LS)
ax2.invert_yaxis()
l2, b2, w2, h2 = ax2.get_position().bounds
ax2.set_position([l2 + 0.005, b1, w2, h1])


ax3.add_patch(Rectangle((0, 0), 2, 85, facecolor='lightgrey', fill=True, lw=0))
N, bins, patches = ax3.hist(DataFrameDrop['MLRTimeAll'], bins=50, range=[0, 5.0], color=myColors[9])
ax3.set_xlim([0, 5.0])
ax3.set_ylim([0, 62])
ax3.set_xlabel('Response latency [s]', size=LS)
ax3.set_xticks([0, 1, 2, 3, 4, 5])
ax3.tick_params(axis='both', labelsize=TLS, length=TickLength)
ax3.set_ylabel('# MLR', size=LS, rotation=90)
ax3.yaxis.labelpad = 0.5

for spine in ax3.spines.values():
    spine.set(visible=True, lw=1, edgecolor="black")

ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.text(-1.7, 63, 'B', weight="bold", size=LetterSize)

l3, b3, w3, h3 = ax3.get_position().bounds
ax3.set_position([l3 + 0.09, b1, w3, h1])


plt.savefig('Figure2'+'.jpg', dpi=1200)
plt.show()

