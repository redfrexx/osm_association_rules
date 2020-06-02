#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""__description__
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import os
import pandas as pd
from nb_utils import load_data, load_city_rules, dump_city_rules


def test_load_city_rules():
    """
    Test if context-based association rules are correctly written and loaded to disk
    :return:
    """

    interim_dir = "../../interim_results"
    cities = ["dresden"]
    data_dir = "../../data"

    all_features = load_data(cities, data_dir)
    city_rules = load_city_rules(cities, interim_dir, all_features)

    tmp_dir = os.path.join(interim_dir, "test")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    dump_city_rules(city_rules, tmp_dir)
    city_rules2 = load_city_rules(cities, tmp_dir, all_features)

    pd.testing.assert_frame_equal(city_rules["dresden"]["heatmap"], city_rules2["dresden"]["heatmap"])
    pd.testing.assert_frame_equal(city_rules["dresden"]["valid_rules"], city_rules2["dresden"]["valid_rules"])
    pd.testing.assert_frame_equal(city_rules["dresden"]["sel_features"], city_rules2["dresden"]["sel_features"])
