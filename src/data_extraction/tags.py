#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Counts the tags which are located inside each green space
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"


import os
import pandas as pd
import logging
import glob
from ohsome import OhsomeClient, OhsomeException
import geopandas as gpd


def tags_inside_greenspaces(outdir, region_name, timestamp, context_keys):
    """
    Counts the tags which are located inside each green space and converts it to a format suitable for association
    rule mining.
    :param outdir:
    :param region_name:
    :param timestamp:
    :param context_keys:
    :return:
    """

    logger = logging.getLogger("osm_association")

    feature_files = glob.glob(os.path.join(outdir, region_name, "*.geojson"))
    if len(feature_files) == 0:
        logger.critical("No feature files found.")
        return 1

    # Send ohsome request
    client = OhsomeClient()

    feature_content_dfs = {}
    all_feature_tag_dfs = []
    for f in feature_files:

        logger.info("Processing %s" % f)

        # Features incl. age, ntags, nversions
        features_df = gpd.read_file(f)

        # Get OSM tag of green space type
        greentag = "=".join(os.path.splitext(os.path.basename(f))[0].split("_")[1:])

        # Convert tags of features to binary columns
        feature_tag_df = features_df.drop(["age", "changes", "max_version", "n_tags", "geometry", "id"], axis=1)
        feature_tag_df = feature_tag_df.set_index("@osmId")
        feature_tag_df["geometry"] = features_df.set_index("@osmId")["geometry"]
        feature_tag_df["green_tag"] = greentag
        all_feature_tag_dfs.append(feature_tag_df)

        # Get count of contained objects
        feature_content_df = features_df.loc[:, ["id", "@osmId", "geometry"]].copy()
        feature_content_df.set_index("id", inplace=True)

        for key in context_keys:
            logger.info("Querying {}=* ...".format(key))

            try:
                res = client.elements.count.groupBy.boundary.groupBy.tag.post(bpolys=feature_content_df, time=timestamp, keys=[key],
                                                                      groupByKey=key, format="json", types=["NODE", "WAY"]) #
            except OhsomeException as e:
                logger.critical("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))
                continue

            count_df = res.as_dataframe()
            if count_df.empty:
                logging.info("No features found")
                continue
            count_df = count_df.loc[~count_df.index.isin(["remainder"], level=1)]
            count_df = count_df.droplevel("timestamp")
            count_df["id"] = count_df.index.get_level_values(0)
            count_df["tag"] = count_df.index.get_level_values(1)
            count_df = count_df.droplevel("boundary")
            count_df = count_df.set_index("id")
            count_df = count_df.pivot_table('value', ['id'], 'tag')

            # Append to features dataframe
            feature_content_df = feature_content_df.join(count_df)

        # Set index
        feature_content_df = feature_content_df.set_index("@osmId")
        feature_content_df["gt:" + greentag] = 1

        # Append to dict of dataframes
        feature_content_dfs[greentag] = feature_content_df

    del client

    # Contents of features
    all_feature_content = pd.concat(feature_content_dfs.values(), sort=True)
    all_feature_content = all_feature_content.drop(["geometry"], axis=1)
    all_feature_content_bin = all_feature_content > 0
    all_feature_content_bin.reset_index(inplace=True)
    pd.DataFrame(all_feature_content_bin).to_json(os.path.join(outdir, os.path.splitext(os.path.basename(f))[0] + "_tags.json"), orient="records")

    return 0

if __name__ == "__main__":
    pass