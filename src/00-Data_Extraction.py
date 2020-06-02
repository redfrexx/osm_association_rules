#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performs data extraction for association rule analysis using the ohsome API
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"


import os, sys
import argparse
import geopy

from data_extraction.download import download_greenspace_features
from data_extraction.context import context_variables
from data_extraction.tags import tags_inside_greenspaces
from utils import create_bbox, center_from_name, load_config_yaml, init_logger, update_config_yaml
from geopy.exc import GeocoderTimedOut


def main():

    DEBUG = True

    if not DEBUG:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Download and preprocess data for OSM associtation rule analysis.')
        parser.add_argument('region', help='Name of region')
        parser.add_argument('config', help='Path to configuration file (.yaml)')
        args = parser.parse_args()
        region_name = args.region
        config_file = args.config
    else:
        # For testing: Command line parameters
        region_name = "dresden"
        config_file = "../config/parks.yaml"

    # Load config parameters
    if not os.path.exists(os.path.abspath(config_file)):
        print("Config file %s does not exist" % os.path.abspath(config_file))
        exit(1)
    config = load_config_yaml(os.path.abspath(config_file))
    location_config = config["locations"][region_name]
    greenspace_tags = [t.strip().split("=") for t in config['greenspace_tags']]
    timestamp = config["project"]["timeperiod"].split(",")[1].replace("-", "")
    keys = [key['key'] for key in config["processing"]["greenspace_content_keys"]]

    # Set output directory
    out_dir = os.path.abspath(config["project"]["data_dir"])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Set up logger
    log_dir = os.path.join(out_dir, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = init_logger("osm_association", log_dir)

    logger.info("Start processing %s" % location_config["name"])

    # Parse parameters
    try:
        if "center" in location_config.keys():
            center = location_config["center"]
        else:
            try:
                center = center_from_name(location_config["name"], location_config["country_code"])
            except geopy.exc.GeocoderServiceError as e:
                logger.critical(e)
                exit(1)
            # Update config file
            location_config["center"] = list(center)
            config["locations"][region_name] = location_config
            update_config_yaml(config, config_file)
        bbox = create_bbox(center, location_config['width'])
    except ValueError as e:
        logger.critical(e)
        return 1
    except KeyError as e:
        logger.critical(e)
        return 1
    except GeocoderTimedOut as e:
        logger.critical(e)
        return 1

    # 1. Download green space features from OSM
    # -----------------------------------------
    logger.info("Step 1: Downloading green space features...")
    res = download_greenspace_features(tags=greenspace_tags, timeperiod=config["project"]["timeperiod"], outdir=out_dir,
                                       region=region_name, bbox=bbox)
    if res != 0:
        sys.exit(1)
    logger.info("Step 1: Done.")

    # 2. Calculate context variables
    # ----------------------------------------------------------
    logger.info("Step 2: Calculating context variables...")
    res = context_variables(out_dir, region_name, config)
    if res != 0:
        sys.exit(1)
    logger.info("Step 2: Done.")

    # 3. Get tags inside of green spaces
    # --------------------------------------
    logger.info("Step 3: Counting tags inside of green spaces...")
    res = tags_inside_greenspaces(out_dir, region_name, timestamp, keys)
    if res != 0:
        sys.exit(1)
    logger.info("Step 3: Done.")

    return 0


if __name__ == "__main__":
    main()