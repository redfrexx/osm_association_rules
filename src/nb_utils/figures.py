#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Functions used to create figures of the paper
"""

__author__ = "Christina Ludwig, GIScience Research Group, Heidelberg University"
__email__ = "christina.ludwig@uni-heidelberg.de"

import matplotlib.pyplot as plt
import seaborn as sns
import os
import nb_utils as nbu

def fig_1(all_features, pretty_city_names, figures_dir=None):
    """

    :param all_features:
    :param pretty_city_names:
    :param figures_dir:
    :return:
    """
    fontsize = 9
    linewidth_box = 0.8
    linewidth = 0.6

    all_features_boxplot = all_features.copy()
    all_features_boxplot["area"] = all_features_boxplot["area"] / 10000.

    fig, axes = plt.subplots(1,2, figsize=(8.5, 3.5))
    fig.tight_layout(pad=2)

    b = sns.boxplot(x="city", y="area",data=all_features_boxplot, ax=axes[0], color='0.85', width=0.7, linewidth=linewidth_box)
    axes[0].set_yscale("log")
    axes[0].set_xticklabels(pretty_city_names.values(), rotation=45, fontsize=fontsize)
    axes[0].set_ylabel("Area [ha]", fontsize=fontsize)
    axes[0].set_xlabel("", fontsize=fontsize)

    sns.boxplot(x="city", y="feature_count",data=all_features_boxplot, ax=axes[1], color='0.85', width=0.7, linewidth=linewidth_box)
    axes[1].set_yscale("log")
    axes[1].set_xticklabels(pretty_city_names.values(), rotation=45, fontsize=fontsize)
    axes[1].set_ylabel("Number of OSM features", fontsize=fontsize)
    axes[1].set_xlabel("", fontsize=fontsize)

    [i.set_linewidth(linewidth) for _, i in axes[0].spines.items()]
    [i.set_linewidth(linewidth) for _, i in axes[1].spines.items()]
    plt.gcf().subplots_adjust(left=0.08, top=0.95, bottom=0.2)

    if figures_dir is not None:
        plt.savefig(os.path.join(figures_dir, "figure_1.pdf"), dpi=600, bottom=0.4)


def fig_2(ohsome_data, figures_dir=None):
    """
    Plot figure 2: Number of parks over time
    :param cities_df:
    :param figures_dir:
    :return:
    """
    fig, axes = plt.subplots(1,1, figsize=(10, 4))
    linestyles = ['-', '--', '-.', ':']
    for col, style in zip(ohsome_data.columns, linestyles):
        ohsome_data[col].plot(style=style, ax=axes, color="black")
    plt.xlabel("Year")
    plt.ylabel("Number of parks in OSM")
    plt.legend()
    if figures_dir is not None:
        plt.savefig(os.path.join(figures_dir, "figure_2.pdf"), dpi=600, bottom=0.4)


def fig_4(current_rule, col, valid_rules, selected_features, figures_dir=None):
    """
    Create figure showing the relationship between confidence and lift depending on a context variable
    :param current_rule:
    :param col:
    :param valid_rules:
    :param selected_features:
    :return:
    """
    print(current_rule)
    sel_valid_rules = valid_rules.loc[
        (valid_rules["context"] == col) & (valid_rules["rule"] == current_rule)]
    sel_valid_rules["cond_range"] = sel_valid_rules["context_max"] - sel_valid_rules["context_min"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
    nbu.plot_graph(selected_features, sel_valid_rules.loc[sel_valid_rules["cond_range"] >= 2], col=col,
                   metric="confidence", ax=axes[0], cond_max=10)
    nbu.plot_graph(selected_features, sel_valid_rules.loc[sel_valid_rules["cond_range"] >= 2], col=col, metric="lift",
                   ax=axes[1], cond_max=10)
    axes[1].get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.5)
    if figures_dir is not None:
        plt.savefig(os.path.join(figures_dir, "figure_4.pdf"), dpi=600, layout="tight")


def fig_users_tokyo(parks_tokyo, users_tokyo, figures_dir=None):
    """
    Plot supplemental figure of number of parks and users in Tokyo
    :param parks_over_time:
    :param users_tokyo:
    :param figures_dir:
    :return:
    """
    fig, axes = plt.subplots(1,1, figsize=(10, 4))

    color = 'tab:red'
    parks_tokyo.plot(kind="line", ax=axes, color=color, legend=True, label="Number of Parks")
    axes.set_ylabel('Number of parks')

    ax2 = axes.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Number of users')  # we already handled the x-label with ax1
    users_tokyo["All active users"].plot(kind="line", color="green", ax=ax2, legend=False)
    users_tokyo["Users mapping parks"].plot(kind="line", color=color, ax=ax2, legend=False)

    ax2.legend(loc=(0.01, 0.72))
    plt.xlabel("Year")
    plt.title("Mapping of leisure=park in Tokyo")
    if figures_dir is not None and os.path.exists(figures_dir):
        plt.savefig(os.path.join(figures_dir, "figure_supplement_1.pdf"), dpi=600, bottom=0.4)

