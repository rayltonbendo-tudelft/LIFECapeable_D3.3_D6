# LIFECapeable_D3.3_D6
This repository contains the codes for the reproduction of the uni- and multivariate extreme value analysis of significant wave height and water level for the following task of the LIFECapeable project:

> " Data set with simulated wave heights and water levels at the toe of the dike that will be used for the preparatory wave impact test and field experiment. The uni- and multivariate extreme value analysis of Hs and SWL will be carried out using open access programming languages, e.g., Python and/or R, and the script will be made publicly available on GitHub for reproducibility."

## Background and Methodology
This repository provides an overview of the steps taken to define the wave and water level conditions at the toe of the dike. All the coordinates used in this work are expressed according to the Swedish national reference systems: EPSG:3006 - SWEREF99 TM for the coordinates and the RH2000 for the vertical datum.

### Data Collection
Hourly wave data were obtained from the hindcast model managed by Lund University (Adell et al., 2023). Specifically:

- Data from model extraction point 10195 (Lat: 353285m- Long: 6141052m), located on the west side of the Falsterbo peninsula, were used to characterize northern storms.
- Data from model extraction point 10958 (Lat: 367538m- Long: 6129815m), located south of the Falsterbo peninsula, were used to describe south-eastern and south-western storms.
  
Hourly water level data were retrieved from the open-access Swedish Meteorological and Hydrological Institute ([SMHI](https://www.smhi.se/nyhetsarkiv)) database, using records from the station located in Ystad (Lat: 425094m - Long: 6141800m) for this study.

Reference: Adell, A., Almström, B., Kroon, A., Larson, M., Bertacchi Uvo, C., Hallin, C. (2023). Spatial and temporal wave climate variability along the south coast of Sweden during 1959–2021. 
Regional Studies in Marine Science, 63. https://doi.org/10.1016/j.rsma.2023.103011.

### Extreme Value Analysis and Wave Simulations
For each storm direction, both univariate and multivariate extreme value analyses were performed to determine the boundary conditions (wave parameters and water level) required for the wave simulations. 
These simulations were performed using SWAN, an open-access numerical wave model.

In the monovariate analysis, the wave and water level conditions were considered completely uncorrelated events; this means that for a given return period (e.g. 100 years), the 100-year wave conditions and the 100-year water level were assumed to occur simultaneously, although their actual correlation may be different.

## Results and Dataset Structure
The final dataset provides wave climate parameters—significant wave height (Hm0), peak period (Tp), and wave direction—along with water depth along the dike perimeter. The data is categorized into three scenarios:

- Monovariate analysis 
- Multivariate "OR" scenario
- Multivariate "AND" scenario

See the file "LIFECapeable-DleiverableD3.3-Significant wave height and water levels at the toe of the dike.csv" for more details on the result.

### Scenario Definitions
- "AND" scenario: Represents the joint exceedance probability where both wave height and water level exceed their respective univariate thresholds.
- "OR" scenario: Represents the joint exceedance probability where either wave height, water level, or both exceed their respective univariate thresholds. This scenario is less conservative than the "AND" scenario, accounting for cases where a single variable dominates extreme conditions.

The results cover seven return periods and 236 points along the coastal protection structures, spaced approximately 100 meters apart; At each point, three extreme wave climate and water level values are computed, each corresponding to one of the three simulated storm directions. The final dataset is constructed by selecting, for each point, the extreme condition that produced the highest wave height among the three storm directions. Once the dominant storm direction is identified, the associated wave period, wave direction, and water level are also selected to characterize the conditions at that location along the seawall.

### Repository structure
The code consists of the following files and directories:
- **Data.rar**: a folder containing the wave and water level data 
  - Data_Falsterbo10195_YSTAD.csv: data from model extraction point 10195, located on the west side of the Falsterbo peninsula and Ystad water level data 
  - Data_Falsterbo10958_YSTAD.csv: data from model extraction point 10958, located on the south side of the Falsterbo peninsula and Ystad water level data 
    
- **py_scripts**: a folder containing the main scripts to execute the extreme value analysis
  - Monovariate extreme value analysis:
    - Water Levels:
      - py_WL_scr011_POT_Def_Om.py
      - py_WL_scr021_EVA_Fitting_ConfInerval_Om.py             
    - Significant Wave Height:
      - py_UN_Hm0_scr011_POT_Threshold_DeclustetingTime_SE.py
      - py_UN_Hm0_scr012_POT_Threshold_DeclustetingTime_SW.py
      - py_UN_Hm0_scr013_POT_Threshold_DeclustetingTime_NW.py
      - py_UN_Hm0_scr021_POT_EVA_ConfInterval_SE.py
      - py_UN_Hm0_scr022_POT_EVA_ConfInterval_SW.py
      - py_UN_Hm0_scr022_POT_EVA_ConfInterval_NW.py
      - py_UN_Hm0_scr031_POT_Fitting_WindSpeed_PeakPeriod_SE.py
      - py_UN_Hm0_scr031_POT_Fitting_WindSpeed_PeakPeriod_SW.py

  - Bivariate extreme value analysis: 
    - py_WD_scr011_POT_Correlation_ExtInd_SE.py               
    - py_WD_scr012_POT_Correlation_ExtInd_SW.py               
    - py_WD_scr013_POT_Correlation_ExtInd_N.py                
    - py_WD_scr021_FittingMargins_UniEVA_SE.py                
    - py_WD_scr022_FittingMargins_UniEVA_SW.py                
    - py_WD_scr023_FittingMargins_UniEVA_N.py                 
    - py_WD_scr031_CopulaSelection_BivEVA_SE.py               
    - py_WD_scr032_CopulaSelection_BivEVA_SW.py               
    - py_WD_scr033_CopulaSelection_BivEVA_N.py                
    - py_WD_scr041_Fitting_WindSpeed_PeakPeriod_SE.py         
    - py_WD_scr042_Fitting_WindSpeed_PeakPeriod_SW.py         
    - py_WD_scr043_Fitting_WindSpeed_PeakPeriod_N.py  
