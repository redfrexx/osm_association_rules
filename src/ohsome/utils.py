"""
Ohsome API client for Python

Utility functions

"""

import math
import numpy as np


def format_coordinates(feature_collection):
    """
    Format coordinates of feature to match ohsome requirements
    :param feature:
    :return:
    """
    features = feature_collection["features"]

    geometry_strings = []
    for feature in features:

        if feature["geometry"]["type"] == "Polygon":
            outer_ring = feature["geometry"]['coordinates'][0]
            geometry_strings.append(",".join([str(c[0]) + "," + str(c[1]) for c in outer_ring]))
        elif feature["geometry"]["type"] == "MultiPolygon":
            outer_rings = feature["geometry"]['coordinates'][0]
            for ring in outer_rings:
                geometry_strings.append(",".join([str(c[0]) + "," + str(c[1]) for c in ring]))
        else:
            print("Geometry type is not implemented")

    return "|".join(geometry_strings)


def find_groupby_names(url):
    """
    Get the groupBy names
    :return:
    """
    return [name.strip("/") for name in url.split("groupBy")[1:]]


def format_geodataframe(geodataframe):
    """
    Converts a geodataframe to a json object to be passed to an ohsome request
    :param geodataframe:
    :return:
    """
    if not "id" in geodataframe.columns:
        UserWarning("Dataframe does not contain an 'id' column. Joining the ohsome query results and the geodataframe will not be possible.")

    # Create a json object which holds geometry, id and osmid for ohsome query
    return geodataframe.to_json(na="drop") #.loc[:, ["id", "geometry"]]


def split_bbox(bbox, tile_size):
    """
    Split bounding box in tiles

    :param bbox: bounding box in format [xmin, ymin, xmax, ymax]
    :param tile_size: the size of the tiles in x and y direction in crs coordinates in format (x_tilesize, y_tilesize)
    :return:
    """
    x_min, y_min, x_max, y_max = bbox
    dx, dy = tile_size

    # Number of full tiles in x and y direction
    x_tiles = math.floor(round(x_max - x_min, 6) / dx)
    y_tiles = math.floor(round(y_max - y_min, 6) / dy)

    # Remainder of bbox in x and y direction
    x_rest = round(x_max - x_min, 6) % dx
    y_rest = round(y_max - y_min, 6) % dy

    if x_tiles > 0 and y_tiles > 0:
        for y in range(0, y_tiles):
            for x in range(0, x_tiles):
                yield tuple(np.array([x_min + dx * x, y_min + dy * y, x_min + dx * (x+1), y_min + dy * (y+1)]).round(6))
            if x_rest != 0:
                yield tuple(np.array([x_min + dx * (x+1), y_min + dy * y, x_max, y_min + dy * (y + 1)]).round(6))

    # Last row
    if y_rest != 0:
        for x in range(0, x_tiles):
            yield tuple(np.array([x_min + dx * x, y_min + dy * y_tiles, x_min + dx * (x + 1), y_max]).round(6))
        yield tuple(np.array([x_min + dx * x_tiles, y_min + dy * y_tiles, x_max, y_max]).round(6))

