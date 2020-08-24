#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions used for association rule mining
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.stats import percentileofscore, spearmanr

from nb_utils import rules_colnames, CONTEXT_NAMES, pretty_names_units


def calculate_rules(park_features, max_len=2, min_support=0.05):
    """
    Calculates association rules. To reduce computation time a minimum support threshold is set.
    :param park_features: GeoDataFrame containing parks
    :param max_len: Maximum length of association rules.
    :param min_support: Minimum support threshold
    :return:
    """
    frequent_itemsets = apriori(park_features.select_dtypes(include="bool"), min_support=min_support, use_colnames=True,
                                max_len=max_len)
    if len(frequent_itemsets) == 0:
        return None
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_support)
    # Calculate interest as additional metric
    rules["interest"] = rules["confidence"] - rules["consequent support"]
    # Reformat tags and rule description
    rules["antecedents"] = rules["antecedents"].map(lambda x: list(x))
    rules["consequents"] = rules["consequents"].map(lambda x: list(x))
    if len(rules) > 0:
        rules.loc[:, "rule"] = rules.apply(lambda x: "%s->%s" % (x["antecedents"], x["consequents"]), axis=1)
    return rules


def generate_association_rules(park_features, max_rule_size, min_support, city_labels):
    """
    Generate association rules for all cities
    :param park_features: GeoDataFrame containing park features
    :param max_rule_size: Maximum size of association rules
    :param min_support: Minimum support threshold
    :param city_labels: Dictionary containing proper cities names for plotting
    :return:
    """

    all_rules = []
    labels = {}
    for city in park_features.city.unique():
        # Select park features of the current city
        features = park_features.loc[park_features["city"] == city]
        print("%s: %s features" % (city, len(features)))
        # Create label for the plot with the number of features in the current city
        labels[city] = "{} ({})".format(city_labels[city], len(features))
        rules = calculate_rules(features, max_len=max_rule_size, min_support=min_support)
        # Filter out rules without meaningful consequents
        interesting_rules = rules[rules["consequents"].map(lambda x: "none" not in x)]
        interesting_rules = interesting_rules[interesting_rules["antecedents"].map(lambda x: "none" not in x)]

        if len(interesting_rules) == 0:
            interesting_rules = pd.DataFrame(columns=list(rules.columns) + ["city"])
            interesting_rules = interesting_rules.append({"city": city}, ignore_index=True)
        else:
            interesting_rules.loc[:, "city"] = city
        all_rules.append(interesting_rules)
    return pd.concat(all_rules), labels


def select_interesting_rules(all_rules, min_confidence, min_lift):
    """
    Select intersting rules based on confidence and lift and converts rules to heatmap dataframe fit for
    plotting using seaborn.
    :param all_rules: DataFrame containing association rules
    :param min_confidence: Minimum confidence threshold
    :param min_lift: Minimum lift threshold
    :return: DataFrames containing interesting rrules in heatmap format and original format
    """
    # Reformat rules as strings
    all_rules["rule"] = all_rules.apply(lambda x: "%s → %s" % (", ".join(x["antecedents"]), ", ".join(x["consequents"])), axis=1)

    # Create list of cities
    unique_cities = all_rules["city"].unique()

    # Select relevant rules for heatmap
    interesting_rules = all_rules.loc[(all_rules["lift"] >= min_lift) &
                                   (all_rules["confidence"] >= min_confidence)]
    if len(interesting_rules) == 0:
        empty_df = pd.DataFrame({"city": ["None"], "rule": ["None"], "confidence": [1], "support": [1]})
        return empty_df, interesting_rules

    # Convert rules to heatmap of rules
    unique_rules = interesting_rules["rule"].unique().tolist()
    regions_series = np.array([[x] * len(unique_rules) for x in unique_cities]).flatten()
    rules_series = unique_rules * len(unique_cities)
    heatmap_df = pd.DataFrame({"city": regions_series, "rule": rules_series})
    all_rules = all_rules.set_index(["city", "rule"])
    heatmap_df = heatmap_df.join(all_rules.loc[:, ["confidence", "support", "lift"]], on=["city", "rule"], how="left")
    heatmap_df = heatmap_df.pivot("rule", "city", ["confidence", "lift"])

    return heatmap_df, interesting_rules


def plot_rule_heatmap(rules_heatmap_df, metric="confidence", figsize=(10, 12), labels=None, vmin=None, vmax=None,
                      left=0.25, right=1, top=0.8, bottom=0.05, fontsize=8):
    """
    Plots a heat map of association rules for all cities.
    :param rules_heatmap_df:
    :param metric:
    :param figsize:
    :param labels:
    :param vmin:
    :param vmax:
    :param left:
    :param right:
    :param top:
    :param bottom:
    :param fontsize:
    :return:
    """
    rules_heatmap_df_sns_labels = rules_heatmap_df.apply(
        lambda x: ["{:0.2f}/{:0.2f}".format(c, l) if ~np.isnan(c) else "" for c, l in
                   zip(x[("confidence")], x[("lift")])], axis=1, result_type="expand")
    rules_heatmap_df_sns_labels.columns = rules_heatmap_df["confidence"].columns
    # rules_heatmap_df_sns_labels
    rules_heatmap_df = rules_heatmap_df.rename(columns=labels)

    fig, ax = plt.subplots(figsize=figsize)
    map = sns.heatmap(rules_heatmap_df[metric], cmap="Blues", vmin=vmin, vmax=vmax, ax=ax, linecolor="white",
                      linewidths=-0.5, rasterized=True, cbar_kws={'label': metric.capitalize(), "aspect": 40},
                      annot_kws={"size": fontsize}, fmt="s", annot=rules_heatmap_df_sns_labels)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    map.set_xticklabels(map.get_xticklabels(), rotation=45, horizontalalignment='left', fontsize=fontsize)
    map.set_yticklabels(map.get_yticklabels(), fontsize=fontsize)
    plt.gcf().subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    plt.xlabel('Cities', fontsize=fontsize)
    plt.ylabel('Association rules', fontsize=fontsize)


def plot_graph(all_feat, current_rules, col, bins=50, ax=None, metric="confidence", cond_max=None):
    """
    Plot graph showing confidence or lift values of an association rules derived from different subsets of parks
    :param all_feat:
    :param current_rules:
    :param col:
    :param bins:
    :param ax:
    :param metric:
    :param cond_max:
    :return:
    """

    if cond_max:
        all_feat = all_feat.loc[all_feat[col] <= cond_max]
        bins=cond_max

    if ax is None:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)

    # Print histogram
    sns.distplot(all_feat[col], ax=ax, bins=bins, color="grey", kde=False, norm_hist=True)
    # Adapt layout of graph
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Frequency', labelpad=2)
    ax.set_xlabel(pretty_names_units[col], fontdict={"fontsize": 10}, labelpad=2)
    ax.margins(0)

    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 10
    }
    labels = ["A", "B", "C", "D", "E"]

    from matplotlib.cm import get_cmap

    name = "Set1"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list

    current_rules = current_rules.sort_values("confidence", ascending=True)

    max_rule_idx = current_rules.apply(lambda x: x["context_p_max"] - x["context_p_min"], axis=1).idxmax()
    for i, row in enumerate(current_rules.iterrows()):
        if i >= 20:
            continue
        ax3 = ax.twinx()
        ax3.plot([row[1]["context_min"], min(cond_max, row[1]["context_max"])],[row[1][metric], row[1][metric]], '-', color=colors[i], solid_capstyle="round")
        ax3.annotate("|", xy=(row[1]["context_min"], row[1][metric]), color=colors[i], **anno_args) #color="r",
        ax3.annotate("|", xy=(min(cond_max, row[1]["context_max"]), row[1][metric]), color=colors[i], **anno_args) # color="r",

        if metric == "confidence":
            ax3.annotate(labels[i], xy=(min(cond_max, row[1]["context_max"]), row[1][metric] + 0.05), color=colors[i], **anno_args) # color="r",
            ax3.set_ylim([0.25,1])
        elif metric == "lift":
            ax3.annotate(labels[i], xy=(min(cond_max, row[1]["context_max"]), row[1][metric] + 0.1), color=colors[i], **anno_args) # color="r",
            min_lift = current_rules["lift"].min() * 0.8
            max_lift = current_rules["lift"].max() * 1.2
            ax3.set_ylim([min_lift,max_lift])
            ax3.set_yticks(np.arange(1, 3, 0.5))
        if i > 0:
            ax3.axis("off")
        else:
            ax3.set_ylabel(metric.capitalize(), {"fontsize": 10})
            ax3.spines['top'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['right'].set_visible(False)

            ax.set_title("", fontdict={"fontsize": 10}, pad=1)
    plt.gcf().tight_layout(pad=1.0)


def calculate_context_dependent_rules(all_features, city, col, perc=False, min_nfeatures=100, min_support=0.05, max_len=2):
    """
    Derive association rules for subsets of features

    :param all_features:
    :param region:
    :param col:
    :param perc:
    :param min_nfeatures:
    :param min_support:
    :return:
    """

    valid_rules = pd.DataFrame(columns=rules_colnames)

    # Select data by region
    selected_features = all_features.loc[(all_features["city"] == city)]

    if len(selected_features) == 0:
        return valid_rules, selected_features

    if perc:
        selected_features = selected_features.loc[selected_features[col] < selected_features[col].quantile(0.99), :]

    chunks = [selected_features]
    nfeatures = len(selected_features)
    baseline = True

    if nfeatures < min_nfeatures:
        return valid_rules, selected_features

    # Recursively split dataset in two subsets until the minimum size is reached
    while len(chunks) > 0:
        # For each subset, calculate association rules.
        for chunk in chunks:

            rules_sub = calculate_rules(chunk, max_len=max_len, min_support=min_support)
            selected_rules_sub = rules_sub.loc[
                (rules_sub["support"] > min_support) & (rules_sub["consequents"] != "none")]

            # (rules_sub["antecedent support"] > rules_sub["consequent support"]) &
            if len(selected_rules_sub) > 0:
                selected_rules_sub.loc[:, "context"] = col
                selected_rules_sub.loc[:, "context_min"] = chunk.loc[:, col].min()
                selected_rules_sub.loc[:, "context_max"] = chunk.loc[:, col].max()
                selected_rules_sub.loc[:, "context_p_min"] = percentileofscore(all_features[col].to_numpy(),
                                                                               chunk.loc[:, col].min())
                selected_rules_sub.loc[:, "context_p_max"] = percentileofscore(all_features[col].to_numpy(),
                                                                               chunk.loc[:, col].max())
                selected_rules_sub.loc[:, "nfeatures"] = len(chunk)
                selected_rules_sub.loc[:, "context_mean"] = chunk.loc[:, col].mean()
                selected_rules_sub.loc[:, "baseline"] = baseline
                # First run on all data is the baseline values which all other rules are compared to
                if baseline:
                    baseline = False
                for rule in selected_rules_sub.iterrows():
                    valid_rules = pd.concat([valid_rules, pd.DataFrame(rule[1]).T], axis=0, sort=False)
        new_chunks = split_df(chunks, col)
        new_chunks = list(filter(lambda x: len(x) > min_nfeatures, new_chunks))
        if [len(x) for x in chunks] == [len(x) for x in new_chunks]:
            break
        chunks = new_chunks

    # if len(valid_rules) > 0:
    #     valid_rules.loc[:,"rule"] = valid_rules.apply(lambda x: "%s → %s" % (x["antecedents"], list["consequents"])[0]), axis=1)
    return valid_rules, selected_features


def context_association_rules_all_cities(all_features, min_nfeatures, min_support, city_labels, max_len=2):
    """
    Performs a context-based association rule analysis for all cities.
    :param all_features: GeoDataFrame containing parks of all cities
    :param min_nfeatures: Minimum number of features in a subset of parks
    :param min_support: Minimum support
    :param city_labels: Dictionary containing proper city names for the table
    :return:
    """
    city_rules = {}
    context_names = list(all_features.select_dtypes(np.number).columns)
    counts = pd.DataFrame(index=CONTEXT_NAMES.values(), columns=pd.MultiIndex.from_product([city_labels.values(), ["pos", "neg", "net"]]))
    collect_counts = []
    for city in list(city_labels.keys()):
        print(city)
        all_rule_rank = []
        all_valid_rules = []
        all_sel_features = []
        for con in context_names:

            valid_rules, sel_features = calculate_context_dependent_rules(all_features, city, col=con, min_nfeatures=min_nfeatures, min_support=min_support, max_len=max_len)
            if len(valid_rules) == 0:
                continue
            #valid_rules = valid_rules[valid_rules["antecedent support"] < valid_rules["consequent support"]]
            all_valid_rules.append(valid_rules)
            all_sel_features.append(sel_features)
            rule_rank = pd.DataFrame({"max_conf": valid_rules.groupby("rule").apply(lambda x: max(x["confidence"])),
                                     "valid": valid_rules.groupby("rule").apply(lambda x: sum([(c >= 0.7) & (l > 1.1) for c, l in zip(x["confidence"], x["lift"])]) > 0 ), # | ((c >= 0.7) & (l >= 2.)
                                      #"valid": valid_rules.groupby("rule").apply(lambda x: sum([((c >= 0.8) & (l > 1.0)) | ((c >= 0.7) & (l >= 2.)) for c, l in zip(x["confidence"], x["lift"])]) > 0 ),
                                     "max_lift": valid_rules.groupby("rule").apply(lambda x: max(x["lift"])),
                                     "corr": valid_rules.groupby("rule").apply(lambda x: get_corr(x)),
                                     "cond_range": valid_rules.groupby("rule").apply(lambda x: max(x["context_p_max"]) - min(x["context_p_min"]))})

            # Calculate difference between baseline confidence and max confidence
            baseline_conf = valid_rules.loc[valid_rules["baseline"],:].groupby("rule").apply(lambda x: max(x["confidence"]))
            baseline_conf.name = "baseline_conf"
            rule_rank["baseline_conf"] = rule_rank.join(baseline_conf, how="outer")["baseline_conf"]
            rule_rank.loc[rule_rank["baseline_conf"].isna(), "baseline_conf"] = 0
            rule_rank["diff"] = rule_rank["max_conf"] - rule_rank["baseline_conf"]

            # increase or decrease in confidence
            rule_rank["diff_sign"] = rule_rank["diff"] * rule_rank["corr"].map(lambda x: -1 if x < 0 else 1)
            rule_rank.replace(-99., np.nan, inplace=True)
            rule_rank = rule_rank.sort_values("corr", ascending=False)
            sel_rule_rank = rule_rank.loc[(rule_rank["valid"]) & (rule_rank["diff"] >= 0.2) & (rule_rank["cond_range"] > 0.75) &
                                          (~rule_rank["corr"].isnull()) & ((rule_rank["corr"] > 0.9) | (rule_rank["corr"] < -0.9))] #
            sel_rule_rank.loc[:, CONTEXT_NAMES[con]] = sel_rule_rank["diff_sign"]
            all_rule_rank.append(sel_rule_rank.loc[:, [CONTEXT_NAMES[con]]])

        heatmap_corr_df = all_rule_rank[0].join(all_rule_rank[1:], how="outer")
        all_valid_rules_df = pd.concat(all_valid_rules, axis=0, sort=False)
        all_sel_features_df = pd.concat(all_sel_features, axis=0, sort=False)
        city_rules[city] = {"heatmap": heatmap_corr_df,
                            "valid_rules": all_valid_rules_df,
                            "sel_features": all_sel_features_df}

        counts[(city_labels[city], "pos")] = (heatmap_corr_df > 0).sum(axis=0)
        counts[(city_labels[city], "neg")] = (heatmap_corr_df < 0).sum(axis=0)
        counts[(city_labels[city], "net")] = (heatmap_corr_df > 0).sum(axis=0) - (heatmap_corr_df < 0).sum(axis=0)
        collect_counts.append(counts)

    return counts, city_rules #heatmap_corr_df, all_valid_rules_df, all_sel_features_df


def get_corr(x, metric="confidence"):
    """
    Calculates the Spearman Correlation coefficient between the mean of the context variable and the confidence or
    lift value of association rules derived from multiple subsets of parks. A valid coefficient is only returned if
    the association rules were derived for at least 4 subsets.
    :param x:
    :param metric:
    :return:
    """
    if len(x) <= 4:
        return np.nan
    corr, sig = spearmanr(x["context_mean"], x[metric])
    return corr


def split_df(dfs, context_var):
    """
    Splits a dataframe in multiple subsets based on a context variable
    :param dfs:
    :param context_var:
    :return:
    """
    chunks = []
    for df in dfs:
        if len(df) <= 1:
            continue
        threshold = df[context_var].median()
        chunks.extend([df.loc[df[context_var] <= threshold], df.loc[df[context_var] > threshold]])
    return chunks


def filter_rules(rules_all, cities):
    """
    Filter association rules by comparing rules of size 3 and higher to rules of 2. If the confidence value of the rule
    of size 2 is only 10 percent lower, the large rule is dismissed.
    :param rules_all:
    :param cities:
    :return:
    """
    rules_big = rules_all[rules_all.apply(lambda x: (len(x["consequents"]) > 1) | (len(x["antecedents"]) > 1), axis=1)]
    rules_two = rules_all[
        rules_all.apply(lambda x: (len(x["consequents"]) == 1) & (len(x["antecedents"]) == 1), axis=1)]
    rules_three = rules_all[
        rules_all.apply(lambda x: (len(x["consequents"]) == 1) & (len(x["antecedents"]) == 2), axis=1)]

    filtered_rules = []

    for c in cities:
        print(c)
        rules_two_city = rules_two[rules_two["city"] == c]
        rules_big_city = rules_big[rules_big["city"] == c]
        rules_three_city = rules_three[rules_three["city"] == c]

        for i, ex in rules_big_city.iterrows():
            ant = ex["antecedents"]
            con = ex["consequents"]
            if len(con) == 1:
                res = rules_two_city[rules_two_city.apply(lambda x: any([a in x["antecedents"] for a in ant])
                                                                    & (x["consequents"] == con), axis=1)]
            elif len(con) == 2:
                res = rules_three_city[rules_three_city.apply(lambda x: any([a in x["antecedents"] for a in ant])
                                                                        & (x["consequents"] == con),
                                                              axis=1)]  # Case 1: Found a comparable rule of size 2
            else:
                # print("rules with three consequents are excluded.")
                continue
            if (len(res) > 0) and ((ex["confidence"] - res["confidence"].max()) > 0.10):
                filtered_rules.append(ex)
            elif len(res) == 0:
                filtered_rules.append(ex)

    filtered_rules_df = pd.DataFrame(filtered_rules)
    filtered_rules_df = pd.concat([filtered_rules_df, rules_two])

    return filtered_rules_df