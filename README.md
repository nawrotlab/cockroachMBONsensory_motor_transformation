# cockroachMBONsensory_motor_transformation

### Description
We provide the code to reproduce all figures and supplemental figures of the paper "The mushroom body output 
encodes behavioral decision during sensory-motor transformation" [^1]. The data is publicly available at...

### Resporsitory structure
#### Directory root
The main directory contains the code to reproduce all figures and supplemental figures of the paper.
Furthermore, it contains the requirements.txt file. This file contains all packages and their versions
that are needed to run the code. The requirements.txt file can be used to create a virtual environment with the
required packages.
#### Directory Data
The directory "Data" contains files which are not part of the data published on GIN and a lookup table to map the files
to the figures.
#### Directory Figures
The directory "Figures" contains the Figures contained in the paper and the supplemental figures. These files can be
overwritten by running the FigureX.py files in the root directory.
#### Directory Functions
The directory "Functions" contains custom functions used in the analysis.

### Downloading the data
The data can be either downloaded via the command line or via the browser. 
#### Using the command line
To download the data via the command line,
run the following command in the terminal:
```bash
git clone git@name_of_the_repository
```
#### Using the browser
Download the latest release as a zip file by clicking on Releases on the main page at 
https://web.gin.g-node.org/INM6/multielectrode_grasp. This zip file contains all mat files with the spikesorted data
and a csv file containing the behavioral data. The zip file has to be extracted to the "Data" folder.


### Running the code
To run the code, the data has to be downloaded as described above. The code can be run in the terminal by running the
following command after the virtual environment has been created and activated. 
To create a virtual environment with the required packages, run the following command in the terminal:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
To run the code, run the following command in the terminal:
```bash
python3 Figure2.py
```
This command will run the code to reproduce Figure 2 of the paper. To reproduce another figure, the corresponding file
has to be run.


[^1] DOI and link to the paper will be added once the paper is published.

