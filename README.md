# Regional Variations of Context-based Association Rules in OpenStreetMap

This repository contains the code and data for the analyses described in the paper _Regional Variations of Context-based Association Rules in OpenStreetMap_ by C.Ludwig, S. Fendrich and A. Zipf at the GIScience Research Group, Institute of Geography, Heidelberg University. 

The code is licensed under BSD 3-Clause "New" or "Revised" License.


## Installation 

Running the analysis requires Python > 3.7 and all packages listed in ```./requirements.txt```. To set up the environment using anaconda/miniconda run:

```
conda create -n osm_rules python=3.7
conda activate osm_rules
conda install --file requirements.txt
```

## Content

The results of the analysis may be reproduced using the two jupyter notebooks and the extracted data provided in ```./data```. 

#### 1. Exploratory Data Analysis

The notebook  ```./src/01-Exploratory_Data_Analysis.ipynb``` contains the exploratory data analysis. 

#### 2. Association Rule Mining

The notebook ```./src/02-Association_Rule_Mining.ipynb``` contains the association rule analysis. 


#### 3. Data Extraction 

If you would like to perform this analysis for another city, you may use the Python script `./src/00-Data_Extraction.py` to extract the data using the [ohsome API](https://api.ohsome.org/v0.9/swagger-ui.html). For this to run add the city to the configuration file ```./config/parks.yaml``` and adapt the change the city name in the Python file. To run the data extraction for multiple cities, run ```./src/00-Data_Extraction_Batch.sh```.
