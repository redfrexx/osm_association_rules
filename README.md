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

The analysis is split into three parts:

#### 1. Data Extraction

The extraction of OSM data for a single city is performed within ```./src/00-data_extraction.py``` using the [ohsome API](https://api.ohsome.org/v0.9/swagger-ui.html).  To run the data extraction for multiple cities, run ```./src/batch_data_extraction.sh```. The data extraction for all cities requires quite some time to run. Therefore, all extracted data is stored within the repository in ```./data```. 

#### 2. Exploratory Data Analysis

An exploratory data analysis can be found in the Jupyter Notebook ```./src/01-Exploratory_Data_Analysis.ipynb```

#### 3. Association Rule Mining

The context-based association rule mining is performed within the second Jupyter Notebook ```./src/02-Association_Rule_Mining.ipynb```.





