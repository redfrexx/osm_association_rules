#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Functions used to create tables of the paper
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import os
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori

from nb_utils import CONTEXT_NAMES

def tab_1(all_features, city_labels, config_cities, tables_dir=None):
    """
    Creates a latex table containing the study sites and the number of parks used within this study
    :param all_features: GeoDataframe containing park features
    :param tables_dir: Directory for tables
    :param city_labels: Dictionary containing proper city names used within the table
    :param config_cities: Dictionary containing configuration parameters of the cities
    :return:
    """
    city_stats = pd.DataFrame({"Number of parks in OSM": all_features.groupby("city").count().loc[:, "age"]})
    city_stats["Country"] = city_stats.index.map(lambda x: config_cities[x]["country"])
    city_stats = city_stats.reindex(city_labels.keys()).rename(index=city_labels)
    city_stats = city_stats.reindex(city_labels.values())
    city_stats = city_stats.rename(index=city_labels)
    city_stats.index.name = "City"
    city_stats = city_stats[["Country", "Number of parks in OSM"]]
    city_stats = city_stats.reset_index()
    # Export to file
    if tables_dir is not None:
        with open(os.path.join(tables_dir, "tab_cities.tex"), "w") as dst:
            dst.write(city_stats.to_latex(index=False))

    return city_stats


def tab_2(all_features, tables_dir=None):
    """
    Calculates the Spearman correlation coefficient between context variables and plots it as a latex table
    :param all_features:
    :param tables_dir:
    :return:
    """
    index_dict = {v:k for k,v in enumerate(list(CONTEXT_NAMES.keys())[:-1])}

    corr_df = pd.DataFrame(all_features).select_dtypes(include=np.number).drop("random", axis=1).corr("spearman")
    corr_df = corr_df.round(2)
    # Change order of columns
    corr_df = corr_df[list(CONTEXT_NAMES.keys())[:-1]]
    # Change order of rows
    corr_df["row_names"] = corr_df.index.values
    corr_df = corr_df.reset_index(drop=False)
    corr_df["idx"] = corr_df["index"].map(lambda x: index_dict[x])
    corr_df = corr_df.sort_values("idx")
    corr_df.set_index("row_names", inplace=True)
    corr_df.drop(["index", "idx"], axis=1, inplace=True)
    corr_df.index.name = ""
    # Rename columns
    corr_df = corr_df.rename(columns=CONTEXT_NAMES)
    corr_df = corr_df.rename(index=CONTEXT_NAMES)
    corr_df = corr_df.apply(lambda col: ["\textbf{%s}" % x if (x > 0.5)|(x < -0.5) else x for x in col ], axis=0)
    # export to file
    if tables_dir is not None:
        with open(os.path.join(tables_dir, "tab_correlation.tex"), "w") as dst:
            latex_string = corr_df.to_latex(escape=False)
            latex_string = latex_string.replace("\\begin{tabular}{llllllllllll}", "\\begin{tabularx}{\\textwidth}{lCCCCCCCCCCC}")
            latex_string = latex_string.replace("\\end{tabular}", "\\end{tabularx}")
            dst.write(latex_string)
    return corr_df


def tab_3(cities, all_features, city_labels, tables_dir=None):
    """
    Creates a latex table of frequent tags within parks of each city.
    :param cities:
    :param all_features:
    :param city_labels:
    :param tables_dir:
    :return:
    """
    itemsets = []
    mostfrequent = []
    for r in cities:
        frequent_itemsets = apriori(all_features.drop("none", axis=1).loc[(all_features["city"] == r)].select_dtypes(include="bool"), min_support=0.001, use_colnames=True, max_len=1)
        items = frequent_itemsets.sort_values("support", ascending=False)
        items["itemsets"] = items["itemsets"].map(lambda x: list(x)[0])
        items.set_index("itemsets", inplace=True)
        mostfrequent.extend(items[:5].index.values.tolist())
        items.columns = [r]
        itemsets.append(items)

    itemsets_df = pd.concat(itemsets, axis=1)
    itemsets_df = itemsets_df.loc[set(mostfrequent),:]
    itemsets_df["mean"] = itemsets_df.mean(axis=1)
    itemsets_df.sort_values("mean", ascending=False, inplace=True)
    itemsets_df = itemsets_df.round(2)

    itemsets_df = itemsets_df.apply(lambda col: ["\textbf{%s}" % x if x in list(col.nlargest(5)) else x for x in col ], axis=0)
    itemsets_df.index = itemsets_df.index.map(lambda x: x.replace("_", "\_"))
    itemsets_df.drop("mean", axis=1, inplace=True)

    # Export to file
    if tables_dir is not None:
        with open(os.path.join(tables_dir, "tab_frequent_tags.tex"), "w") as dst:
            dst.write(itemsets_df.rename(columns=city_labels).to_latex(escape=False))
    return itemsets_df


def tab_4(counts, tables_dir=None):
    """
    Creates a latex table showing the influence of the context variables on the association rules of the cities
    :param counts: pd.DataFrame containing the result of the context-based association rule analysis
    :param tables_dir: (str) Target directory
    :return:
    """
    counts_pos = counts.xs("pos", level=1, axis=1)
    counts_pos["Median"] = counts_pos.median(axis=1)
    counts_neg = counts.xs("neg", level=1, axis=1)
    counts_neg["Median"] = counts_neg.median(axis=1)
    counts_rank = counts.xs("pos", level=1, axis=1).rank(axis=0, ascending=False, method="min")
    counts_rank["Median"] = counts_rank.median(axis=1)
    counts_joined = pd.concat(dict(counts_rank = counts_rank, counts_neg = counts_neg, counts_pos = counts_pos),axis=1)
    counts_joined = counts_joined.sort_values(("counts_rank", "Median"))
    counts_joined_label = counts_joined.apply(lambda x: ["{:1.0f}/{:1.0f} ({:1.0f})".format(p,n,r) for p,n,r in zip(x[("counts_pos")], x[("counts_neg")], x[("counts_rank")])], axis=1, result_type="expand")
    counts_joined_label.columns = counts_joined[("counts_rank")].columns
    counts_joined_label.drop("Median", axis=1, inplace=True)
    counts_joined_label = counts_joined_label.apply(lambda col: ["\textbf{%s}" % x if int(x.split("(")[1].split(")")[0]) <= 2 else x for x in col], axis=0)
    # export to file
    if tables_dir is not None:
        with open(os.path.join(tables_dir, "tab_context_rule.tex"), "w") as dst:
            dst.write(counts_joined_label.reset_index().to_latex(index=False, escape=False))
    return counts_joined_label