#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""__description__
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"


from ohsome import OhsomeClient, OhsomeException
import os
import pandas as pd
import logging
import nb_utils.utils as utils


def age_version_changes(dataframe, now):
    """
    Calculate age, number of versions and the number of changes and for each feature
    :param dataframe:
    :param now:
    :return:
    """
    dataframe_copy = dataframe.copy()
    dataframe_copy["validFrom_temp"] = dataframe_copy.index.get_level_values("@validFrom")
    temporal_attributes = pd.DataFrame(dataframe_copy.loc[:, "validFrom_temp"].groupby("@osmId").min())
    temporal_attributes.columns = ["created_on"]
    temporal_attributes["age"] = now - temporal_attributes["created_on"]
    temporal_attributes["age"] = temporal_attributes["age"].map(lambda x: x.days)
    temporal_attributes["changes"] = dataframe_copy["validFrom_temp"].groupby("@osmId").count()
    temporal_attributes["max_version"] = dataframe_copy["@version"].groupby("@osmId").max()
    temporal_attributes = temporal_attributes.drop(["created_on"], axis=1)

    return temporal_attributes


def download_greenspace_features(tags:list, timeperiod:str, outdir:str, region:str, bbox:[str, list]=None):
    """
    Get features from ohsome
    :param osm_key:
    :param osm_value:
    :param bbox:
    :param timestamp:
    :param types:
    :param outdir:
    :return:
    """

    logger = logging.getLogger("osm_association")

    # Create output folder for features
    outdir_features = os.path.join(outdir, region)
    if not os.path.exists(outdir_features):
        os.mkdir(outdir_features)

    for tag in tags:

        osm_key, osm_value = tag
        logger.info("Downloading %s=%s" % (osm_key, osm_value))

        try:
            logger.info("Waiting for ohsome response...")
            client = OhsomeClient()
            ohsome_response = client.elementsFullHistory.geometry.post(bboxes=bbox, time=timeperiod,
                                                                       keys=osm_key, values=osm_value,
                                                                       types=['POLYGON'], properties=["tags", "metadata"])
        except OhsomeException as e:
            logger.critical("{}\nURL:{}\nParameters:{}".format(e, e.url, e.params))
            return 1
        finally:
            del client

        logger.info("Processing ohsome response ...")
        feature_history_df = ohsome_response.as_geodataframe()
        if len(feature_history_df) == 0:
            logger.info("No features found for tag %s-%s" % (osm_key, osm_value))
            continue

        # Get the latest version of the features which exist at the end of the study period
        logger.info("Extracting latest feature version ...")
        last_timestamp = pd.to_datetime(timeperiod.split(",")[1], format="%Y-%m-%d")
        feature_history_df_noidx = feature_history_df.reset_index()
        idx = feature_history_df_noidx.loc[:,["@osmId", "@validTo"]].groupby("@osmId").idxmax()["@validTo"].values
        latest_features = feature_history_df_noidx.iloc[idx, :]
        features_now = latest_features.loc[latest_features["@validTo"] == last_timestamp]
        features_now.set_index("@osmId", inplace=True)

        # Compute age, max version and number of changes
        logger.info("Calculating age, max version and number of changes ...")
        temporal_attrs_df = age_version_changes(feature_history_df, last_timestamp)
        remove_columns = list(filter(lambda x: x.startswith("@"), features_now.columns))
        features_now = features_now.drop(remove_columns, axis=1)
        # Calculate number of tags
        features_now["n_tags"] = features_now.shape[1] - 1 - features_now.isna().sum(axis=1)
        # Calculate area
        features_now["area"] = utils.reproject_to_utm(features_now).area

        # Join data frames
        features_now = features_now.join(temporal_attrs_df, how="left")
        if len(features_now) == 0:
            logger.info("No features found for tag %s-%s" % (osm_key, osm_value))
            continue

        # Write features to file
        logger.info("Export to file ...")
        outfile_name = os.path.splitext(os.path.basename(outdir))[0]
        outputfile_now = os.path.join(outdir_features, "%s_%s_%s.geojson" % (outfile_name, osm_key, osm_value))
        with open(outputfile_now, "w") as dst:
            features_now_json = features_now.reset_index().to_json(na="drop", indent=4) #.drop("id", axis=1)
            dst.write(features_now_json)

        # Write features to file
        logger.info("Export to file ...")
        outfile_name = os.path.splitext(os.path.basename(outdir))[0]
        outputfile_now = os.path.join(outdir_features, "%s_%s_%s.geojson" % (outfile_name, osm_key, osm_value))
        with open(outputfile_now, "w") as dst:
            features_now_json = features_now.reset_index().to_json(na="drop", indent=4) #.drop("id", axis=1)
            dst.write(features_now_json)

    return 0

if __name__ == "__main__":

    # input
    region_name = "london"
    bbox = "136.84541981197225,35.11419158500382,136.91728503470182,35.1729987140306"
    timeperiod = "2007-11-01,2019-11-30"
    types = ["polygon"]
    outdir = "/Users/chludwig/Development/meinGruen/code/osm_associations/temp"

    outdir_region = os.path.join(outdir, region_name)

    if not os.path.exists(outdir_region):
        os.mkdir(outdir_region)

    tags = [("leisure", "park"), ("leisure", "garden"), ("landuse", "cemetery"),
            ("landuse", "grass"), ("landuse", "allotments"), ("landuse", "village_green")]

    for tag in tags:
        outfile = download_greenspace_features(tag[0], tag[1], bbox, timeperiod, types, outdir_region)
        print(outfile)
