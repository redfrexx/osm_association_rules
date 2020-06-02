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

