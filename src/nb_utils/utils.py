#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Additional utility functions
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"



import utm
import math
import pyproj
from shapely.ops import transform
from shapely.geometry import box, Point
import yaml
import logging
import geopy
import os
import datetime
from functools import partial


earthRadius = 6371000.
ONE_DEGREE_IN_METERS_AT_EQUATOR = earthRadius * math.pi / 180.


def reproject_to_utm(dataframe):
    """ Reproject a dataframe with epsg:4326 to UTM in respective zone
    Reprojects a dataframe to UTM
    :param dataframe:
    :return:
    """
    center = box(*dataframe.total_bounds).centroid
    utm_zone = utm.from_latlon(center.y, center.x)
    proj4_string = '+proj=utm +zone=%s +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0' % utm_zone[2]

    return dataframe.to_crs(proj4_string)


def convert_meter_to_degree_longitude(distanceInMeters, latitude):
    """
    Converts meters to degree longitude
    :param distanceInMeters:
    :param latitude:
    :return:
    """
    return distanceInMeters / (math.cos(math.radians(latitude)) * ONE_DEGREE_IN_METERS_AT_EQUATOR);


def buffer_in_meters(geom, distance):
    """
    Buffers a shapely geometry given in geographic coordinates (epsg:4326) by a distance given in meters
    :param geom:
    :param distance:
    :return:
    """
    utm_zone = utm.from_latlon(geom.centroid.y, geom.centroid.x)
    proj4_string = '+proj=utm +zone=%s +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0' % utm_zone[2]

    transform_degree_meter = partial(pyproj.transform, pyproj.Proj(init='EPSG:4326'),
                                     pyproj.Proj(proj4_string))
    # convert the local projection back the the WGS84 and write to the output shp
    transform_meter_degree = partial(pyproj.transform, pyproj.Proj(proj4_string), pyproj.Proj(init='EPSG:4326'))

    return transform(transform_meter_degree, transform(transform_degree_meter, geom).buffer(distance))


def create_bbox(center, width):
    """
    Create a bounding box based on center coordinates and width
    :param center:
    :param width:
    :return:
    """
    utm_zone = utm.from_latlon(center[1], center[0])

    # Define transformation functions
    proj4_string = '+proj=utm +zone=%s +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0' % utm_zone[2]
    transform_degree2meter = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                                     pyproj.Proj(proj4_string))
    transform_meter2degree = partial(pyproj.transform, pyproj.Proj(proj4_string), pyproj.Proj(init='epsg:4326'))

    # Create geometry for center point
    center_geom = Point(center[0], center[1])
    # Transform center point to utm
    center_geom_utm = transform(transform_degree2meter, center_geom)

    minx = center_geom_utm.x - width / 2
    maxx = center_geom_utm.x + width / 2
    miny = center_geom_utm.y - width / 2
    maxy = center_geom_utm.y + width / 2

    bbox_utm = box(minx, miny, maxx, maxy)
    bbox = transform(transform_meter2degree, bbox_utm)

    return bbox.bounds


def load_config_yaml(config_file):
    """
    Load parameters from config file
    :param config_file: path to config file
    :return:
    """
    with open(config_file, 'r') as src:
        config = yaml.load(src, Loader=yaml.FullLoader)
    return config


def update_config_yaml(config, config_file):
    """
    Load parameters from config file
    :param config_file: path to config file
    :return:
    """
    with open(config_file, 'w') as dst:
        config = yaml.dump(config, dst, default_flow_style=False, indent=4)
    return config


def init_logger(name, log_dir):
    """
    Set up logger
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Add handlers
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    filehandler = logging.FileHandler(filename=os.path.join(log_dir, "associate_%s.log" % (datetime.datetime.now().strftime("%Y%m%d"))))
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger


def center_from_name(region_name, country_code=None):
    """
    Get the bounding box of a region using Nominatim
    :param region_name:
    :param country_code:
    :return:
    """

    geocoder = geopy.Nominatim(user_agent="chl2", scheme="http")

    res = geocoder.geocode(region_name, geometry="geojson", country_codes=country_code)
    if isinstance(res, geopy.location.Location):
        raw_lat = res.latitude
        raw_lon = res.longitude
        return (raw_lon, raw_lat)
    else:
        raise ValueError("Location not found.")
