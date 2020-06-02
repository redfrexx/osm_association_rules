#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions used for data handling
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import os
import yaml
from shapely.geometry import box
import numpy as np
import pandas as pd
import geopandas as gpd
import json

from nb_utils.utils import create_bbox, reproject_to_utm


CONTEXT_NAMES = {"area": "Area", "building_density": "Building density", "age": "Days since creation",
                "n_tags": "Number of tags", "changes": "Number of changes", "max_version": "Version number",
                "user_count_inner": "Inner user count", "user_density_inner": "Inner user density",
                "user_count_outer": "Outer user count",  "user_density_outer": "Outer user density",
                "feature_count": "Feature count",  "random": "Random"}

rules_colnames = ['antecedents', 'consequents', 'antecedent support',
       'consequent support', 'support', 'confidence', 'lift', 'leverage',
       'conviction', "context", "context_min", "context_max", "context_p_min", "context_p_max", "nfeatures", "rule"]

pretty_names_units = {"area": "Area [ha]", "building_density": "Building density", "feature_count": "Feature count", "age": "Days since creation", "n_tags": "Number of tags", "changes": "Number of changes", "max_version": "Version number", "user_count_inner": "Inner user count", "user_density_inner": "Inner user density", "user_count_outer": "Outer user count",
               "user_density_outer": "Outer user density",  "random": "Random"}



def load_config(config_file, cities):
    """
    Load config parameters from file
    :param config_file:
    :param cities:
    :return:
    """
    if not os.path.exists(config_file):
        print("ERROR: Config file {} does not exist.".format(config_file))
    else:
        with open(config_file, 'r') as src:
            config = yaml.load(src, Loader=yaml.FullLoader)
        config_cities = config["locations"]
        config_cities = {city: config_cities[city] for city in cities}
    return config_cities


def load_data(cities, data_dir):
    """
    Load data into notebook from file
    :return:
    """
    loaded_tags_dfs = []
    loaded_context_dfs = []
    for city in cities:
        print("Loading {}...".format(city))

        # Check paths
        tags_file = os.path.join(data_dir, city, "{}_tags.json".format(city))
        context_file = os.path.join(data_dir, city, "{}_context.geojson".format(city))
        if (not os.path.exists(tags_file)) or (not os.path.exists(context_file)):
            print("{}: Input files not found.".format(city))
            return None, None, None

        # Read data and set index
        tags_df = pd.read_json(tags_file).set_index("@osmId")
        context_df = gpd.read_file(context_file).set_index("@osmId")

        # Calculate area (should be moved to data_extraction)
        context_df["area"] = reproject_to_utm(context_df).area #/ 10000. # conversion to ha
        # Add column holding the city name
        context_df["city"] = city

        loaded_tags_dfs.append(tags_df)
        loaded_context_dfs.append(context_df)

    # Convert list of dataframes to dataframe
    all_tags_df = pd.concat(loaded_tags_dfs, axis=0)
    all_tags_df = all_tags_df.fillna(False)
    all_context_df = pd.concat(loaded_context_dfs, axis=0)
    all_features = all_context_df.join(all_tags_df, sort=False)

    # Add dummy columns for "no antecedent" and random context variable
    all_features["none"] = True
    all_features["random"] = np.random.rand(len(all_features))
    # The park iteself is always counted as an objects inside of it. Therefore, subtract 1.
    all_features["feature_count"] = all_features["feature_count"] - 1
    # Delete unnecessary columns
    unnecessary_cols = list(filter(lambda x: x.startswith("gt:"), all_features.columns)) + ["leisure=park"]
    all_features.drop(unnecessary_cols, axis=1, inplace=True)

    return all_features


def create_city_bboxes(config_cities):
    """
    Creat bboxes of cities
    :return:
    """
    bboxes = {c: box(*create_bbox(config_cities[c]["center"], config_cities[c]["width"])) for c in config_cities.keys()}
    bbox_df = pd.DataFrame().from_dict(bboxes, orient="index", columns=["geometry"])
    return gpd.GeoDataFrame(bbox_df)


def dump_city_rules(city_rules, interim_dir):
    """
    Write results from context based association rule analysis to file
    :param city_rules:
    :param interim_dir:
    :return:
    """
    city_rules_dir = os.path.join(interim_dir, "city_rules")
    if not os.path.exists(city_rules_dir):
        os.mkdir(city_rules_dir)
    for k, v in city_rules.items():
        print(k)
        v["heatmap"].to_json(os.path.join(city_rules_dir, "{}_heatmap.json".format(k)))
        v["valid_rules"].reset_index().to_json(os.path.join(city_rules_dir, "{}_valid_rules.json".format(k)))
        with open(os.path.join(city_rules_dir, "{}_sel_features.json".format(k)), "w") as dst:
            json.dump(list(v["sel_features"].index), dst)


def load_city_rules(cities, interim_dir, all_features):
    """
    Load results from context based association rule analysis to file
    :param cities:
    :param interim_dir:
    :param all_features:
    :return:
    """
    city_rules = {}
    for city in cities:
        with open(os.path.join(interim_dir, "city_rules", "{}_sel_features.json".format(city))) as dst:
            selected_ids = json.load(dst)
        sel_features = all_features.loc[selected_ids]
        selected_osmids = json
        city_rules[city] = {
            "heatmap": pd.read_json(os.path.join(interim_dir, "city_rules", "{}_heatmap.json".format(city))),
            "valid_rules": pd.read_json(
                os.path.join(interim_dir, "city_rules", "{}_valid_rules.json".format(city))).set_index("index"),
            "sel_features": sel_features}
    return city_rules
