#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Calculates the context variables for each green space feature
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"


from ohsome import OhsomeClient, OhsomeException
import os
import glob
import geopandas as gpd
import logging
from nb_utils.utils import buffer_in_meters, reproject_to_utm, convert_meter_to_degree_longitude
import datetime

import warnings
warnings.filterwarnings('ignore')


def context_variables(outdir, region, config):
    """
    Calculates the context variables for each green space feature
    :param feature_dir: 
    :param outdir:
    :param timestamp: 
    :return: 
    """

    logger = logging.getLogger("osm_association")

    # Directory containing features
    feature_dir = os.path.join(outdir, region)
    if not os.path.exists(feature_dir):
        return 1, "Directory containing features not found"

    # timestamp
    timestamp = config["project"]["timeperiod"].split(",")[1]

    # Get all files with OSM features
    feature_files = glob.glob(os.path.join(feature_dir, "*.geojson"))

    # Send ohsome request
    client = OhsomeClient()

    for f in feature_files:
        logger.info("Processing %s" % os.path.basename(f))

        features_df = gpd.read_file(f)
        if len(features_df) == 0:
            logger.warning("%s is empty.")
            continue

        features_notags_df = features_df.loc[:, ["id", "@osmId", "geometry", "age", "max_version", "changes", "n_tags"]].copy()
        features_notags_df.set_index("id", inplace=True)

        # Create a json object which holds geometry, id and osmid for ohsome query
        features_geom_df = features_notags_df.loc[:, ["geometry", "@osmId"]].copy()
        features_geom_df["geometry"] = features_geom_df["geometry"].map(lambda x: x.buffer(convert_meter_to_degree_longitude(-1, x.centroid.y)))

        # Buffered features
        features_buffered_df = features_notags_df.copy()
        features_buffered_df["geometry"] = features_notags_df["geometry"].map(lambda x: buffer_in_meters(x, 200))

        # Surrounding area without green space
        features_surrounding_df = gpd.overlay(features_buffered_df, features_geom_df, how="difference")

        # Number of objects within green space
        try:
            logger.info("Querying number of features within green space ...")
            res = client.elements.count.groupBy.boundary.post(bpolys=features_geom_df.iloc[:2], time=timestamp, types=["NODE", "WAY"])
            feature_count = res.as_dataframe()
            feature_count = feature_count.droplevel("timestamp")
            features_notags_df["feature_count"] = feature_count["value"]
        except OhsomeException as e:
            error_file = os.path.join(outdir, "logs", "OhsomeException_%s.txt" % (datetime.datetime.now().strftime("%Y%m%d-%H%M")))
            with open(error_file, "w") as dst:
                dst.write("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))
            logger.critical("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))

        # Density of buildings within 200m bounding box around green space
        try:
            logger.info("Querying building density outside of green space ...")
            res = client.elements.area.density.groupBy.boundary.post(bpolys=features_surrounding_df, time=timestamp, keys=["building"], types=["WAY"])
            count_buildings = res.as_dataframe()
            count_buildings = count_buildings.droplevel("timestamp")
            features_notags_df["building_density"] = count_buildings["value"] #/ features_buffered_bbox_df["area"]
        except OhsomeException as e:
            error_file = os.path.join(outdir, "logs", "OhsomeException_%s.txt" % (datetime.datetime.now().strftime("%Y%m%d-%H%M")))
            with open(error_file, "w") as dst:
                dst.write("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))
            logger.critical("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))

        # Density of users within green space
        try:
            logger.info("Querying user density within green space ...")
            res = client.users.count.density.groupBy.boundary.post(bpolys=features_geom_df, time=config["project"]["timeperiod"], types=["NODE"])
            count_user = res.as_dataframe().droplevel(2).droplevel(1)
            features_notags_df["user_density_inner"] = count_user["value"]
        except OhsomeException as e:
            error_file = os.path.join(outdir, "logs", "OhsomeException_%s.txt" % (datetime.datetime.now().strftime("%Y%m%d-%H%M")))
            with open(error_file, "w") as dst:
                dst.write("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))
            logger.critical("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))

        # Count of users within of green space
        try:
            logger.info("Querying user count within green space ...")
            res = client.users.count.groupBy.boundary.post(bpolys=features_geom_df, time=config["project"]["timeperiod"], types=["NODE"])
            count_user = res.as_dataframe().droplevel(2).droplevel(1)
            features_notags_df["user_count_inner"] = count_user["value"]
        except OhsomeException as e:
            error_file = os.path.join(outdir, "logs", "OhsomeException_%s.txt" % (datetime.datetime.now().strftime("%Y%m%d-%H%M")))
            with open(error_file, "w") as dst:
                dst.write("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))
            logger.critical("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))

        # Density of users within 200m bounding box around green space
        try:
            logger.info("Querying user density within buffered green space ...")
            res = client.users.count.density.groupBy.boundary.post(bpolys=features_surrounding_df, time=config["project"]["timeperiod"], types=["NODE"])
            count_user = res.as_dataframe().droplevel(2).droplevel(1)
            features_notags_df["user_density_outer"] = count_user["value"]
        except OhsomeException as e:
            error_file = os.path.join(outdir, "logs", "OhsomeException_%s.txt" % (datetime.datetime.now().strftime("%Y%m%d-%H%M")))
            with open(error_file, "w") as dst:
                dst.write("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))
            logger.critical("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))

        # Count of users within 200m bounding box around green space
        try:
            logger.info("Querying user count within buffered green space ...")
            res = client.users.count.groupBy.boundary.post(bpolys=features_surrounding_df, time=config["project"]["timeperiod"], types=["NODE"])
            count_user = res.as_dataframe().droplevel(2).droplevel(1)
            features_notags_df["user_count_outer"] = count_user["value"]
        except OhsomeException as e:
            error_file = os.path.join(outdir, "logs", "OhsomeException_%s.txt" % (datetime.datetime.now().strftime("%Y%m%d-%H%M")))
            with open(error_file, "w") as dst:
                dst.write("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))
            logger.critical("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))

        # Write to file
        outfile = os.path.join(feature_dir, os.path.basename(f).split(".")[0] + "_context.geojson")
        features_notags_df.reset_index().to_file(outfile, driver="GeoJSON", na="drop")

    del client

    return 0

if __name__ == "__main__":

    root_dir = "/Users/chludwig/Development/meinGruen/data/osm_associations/run2"
    timestamp = "2019-06-30"
    region_name = "athens"

    features_dir = os.path.join(root_dir, region_name, "features")

    outdir_region = os.path.join(root_dir, region_name, "local_stats")
    if not os.path.exists(outdir_region):
        os.mkdir(outdir_region)

    context_variables(features_dir, outdir_region, timestamp)