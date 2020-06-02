#!/bin/sh

python 00-data_extraction.py dresden ./config/parks.yaml
python 00-data_extraction.py berlin ./config/parks.yaml
python 00-data_extraction.py london ./config/parks.yaml
python 00-data_extraction.py newyork ./config/parks.yaml
python 00-data_extraction.py telaviv ./config/parks.yaml
python 00-data_extraction.py tokyo ./config/parks.yaml
python 00-data_extraction.py vancouver ./config/parks.yaml
python 00-data_extraction.py osaka ./config/parks.yaml




