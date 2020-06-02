#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions used for sending requests to the ohsome API
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"


from shapely.geometry import box
import pandas as pd
import geopandas as gpd

from nb_utils import create_bbox
import ohsome


def ohsome_parks_over_time(config_cities, keys, values, timeperiod, types):
    """ Calculates the number of parks in OSM for each city over time
    """

    # Create bounding box geometry of each city
    bboxes = {c: box(*create_bbox(config_cities[c]["center"], config_cities[c]["width"])) for c in config_cities.keys()}
    bbox_df = pd.DataFrame().from_dict(bboxes, orient="index", columns=["geometry"])
    bbox_df = gpd.GeoDataFrame(bbox_df)
    pretty_city_names = {city: config_cities[city]["name"] for city in config_cities.keys()}

    client = ohsome.OhsomeClient()
    res = client.elements.count.groupBy.boundary.post(bpolys=bbox_df, keys=keys, values=values,
                                                      time=timeperiod, types=types)
    del client

    parks_unstacked = res.as_dataframe().unstack(level=0)
    parks_unstacked.columns = parks_unstacked.columns.droplevel(level=0)
    parks_unstacked.rename(columns=pretty_city_names, inplace=True)
    parks_unstacked.columns.name = "Cities"

    return parks_unstacked


def ohsome_users_tokyo(bpolys, timeperiod):
    """

    :param bpolys:
    :return:
    """

    client = ohsome.OhsomeClient()

    # All users
    res = client.users.count.post(bpolys=bpolys, time=timeperiod, types=["NODE", "WAY"])
    users_all_unstacked = res.as_dataframe().drop("fromTimestamp", axis=1).set_index("toTimestamp")
    users_all_unstacked.rename(columns={"value": "All active users"}, inplace=True)

    # Users mapping parks
    res = client.users.count.post(bpolys=bpolys, keys=["leisure"], values=["park"],
                                  time=timeperiod, types=["WAY", "RELATION"])
    users_parks_unstacked = res.as_dataframe().drop("fromTimestamp", axis=1).set_index("toTimestamp")
    users_parks_unstacked.rename(columns={"value": "Users mapping parks"}, inplace=True)

    del client
    return users_all_unstacked.join(users_parks_unstacked)


def ohsome_source_tag(bpolys, start_date, end_date):
    """
    Query the source tag
    :param bpolys:
    :param start_date:
    :param end_date:
    :return:
    """

    client = ohsome.OhsomeClient()
    res = client.elements.count.groupBy.tag.post(bpolys=bpolys, keys=["leisure"], values=["park"],
                                                 groupByKey="source", time="{0},{1}".format(start_date, end_date),
                                                 types=["WAY", "RELATIONS"])

    source_tags = res.as_dataframe().reset_index(1)
    source_tags = source_tags.pivot(columns="timestamp", values="value")
    source_tags["difference"] = source_tags[end_date] - source_tags[start_date]
    source_tags.sort_values("difference", ascending=False, inplace=True)

    return source_tags

