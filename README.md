# Abaqus Scripting
Hi there! These are my Python scripts to:
    a) Create Abaqus input files (.inp)
    b) Extract information from Abaqus output databases (.odb)
    c) Analyze and visualize data parsed from Abaqus output databases

> [!IMPORTANT]
> Although I am pleased to share my codes with the broader community for education, research and development purposes, I do not take any responsibility for the results obtained. You are responsible for your results.
> **Attribution:** If you use this code for academic of research purposes, proper attribution to the original author (myself) is appreciated.

> [!TIP]
> Go through this ReadMe file in detail to understand the repository structure and the usage of the scripts.

## Requirements
> [!NOTE]
> Abaqus 2021 or later version. Older output database files need to be upgraded first. 
> Python 3 or later version. You might need to install additional packages to run the scripts.

## Repository Structure
As of January 2023, the repository is organized as follows:

### Data Extraction
Contains the following scripts:

#### **Sets_ODB.py:**
To run this script '1.Sets_ODB`, you need to previously process:
- An .odb file. 
Upon definition of a 'core' region of interest (ROI), the script will output:
- A .txt file containing all the elements contained in the ROI, and their connectivities.
- A .txt file containing all the nodes contained in the ROI, and their coordinates.

#### **Extract_Responses.py:**
To run this script, you need to previously extract:
- A .txt file containing all the elements contained in a region of interest, and their connectivities.
- A .txt file containing all the nodes contained in a region of interest, and their coordinates.
The script will output:
- As many .txt files as time steps are contained in the .odb, including nodal information for the strain/stress/displacement field.

### Scientific Visualization



==
> [!IMPORTANT]
> Codes might contain bugs, you are responsible for your results.

> [!WARNING]
> Settings might need to be adjusted.


> [!CAUTION]