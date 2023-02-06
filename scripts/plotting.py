import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


def plot_morans_i():
    np.random.seed(42)
    test = create_data(500)
    w = get_weights_as_array(test[:, :2], 0.2)
    for t in range(5):
        morans = morans_i(test[:, t + 2], w)
        plt.scatter(test[:, 0], test[:, 1], c=test[:, t + 2])
        plt.colorbar()
        plt.title(f"Morans I: {morans}")
        plt.axis("off")
        plt.show()


def main_plot(
    results,
    nr_data=500,
    noise_type="uniformly distributed",
    save_path="outputs/main_plot.pdf",
    score_col="RMSE",
):
    noise_level_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    locality_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    include_models = [
        "OLS",
        "SLX",
        "GWR",
        "RF",
        "RF (coordinates)",
        "spatial RF",
        "Kriging",
    ]
    include_models = [
        m for m in include_models if m in results["model"].unique()
    ]
    # [model for model in results["model"].unique() if "geo" not in model]
    nr_models = len(include_models)
    fig = plt.figure(figsize=(16, 6.5))
    for mode_ind, mode in enumerate(
        ["linear", "non-linear (simple)", "non-linear"]
    ):
        #     print("----------------")
        for model_ind, model in enumerate(include_models):
            #         print(mode, "data, --> model:", model)
            results_filter = results[
                (results["data mode"] == mode)
                & (results["model"] == model)
                & (results["nr_data"] == nr_data)
                & (results["noise_type"] == noise_type)
            ]
            results_filter.set_index(["noise", "locality"], inplace=True)
            visualize_scores = np.zeros(
                (len(noise_level_range), len(locality_range))
            )
            for i, noise in enumerate(noise_level_range):
                for j, locality in enumerate(locality_range):
                    score = results_filter.loc[noise, locality][
                        score_col
                    ].mean()
                    visualize_scores[i, j] = score

            ax1 = fig.add_subplot(
                3, nr_models + 1, ((nr_models + 1) * mode_ind) + model_ind + 1
            )
            imshow_plot = ax1.imshow(visualize_scores, vmin=0, vmax=0.6)
            #         plt.axis("off")
            #             if model_ind==0:
            #                 ax1.set_ylabel("$\longleftarrow$ Increasing \n noise", fontsize=15)
            #             ax1.yaxis.set_label_position("right")
            #             ax1.yaxis.tick_right()
            plt.xticks([])
            plt.yticks([])
            #             ax1.set_xlabel("$\longrightarrow$ decreasing \n stationarity", fontsize=10)
            if model_ind == 0:
                #             ax2 = ax1.twinx()
                #             ax2.set_ylabel(mode)
                #             ax2.yaxis.set_label_position("right")
                #                 pad = 2
                mode_new = (
                    "non-linear\n(simple)  "
                    if mode == "non-linear (simple)"
                    else mode
                )
                ax1.annotate(
                    mode_new,
                    xy=(0, 0.5),
                    xytext=(-50, 0),  # ax1.yaxis.labelpad - pad
                    xycoords=ax1.yaxis.label,
                    textcoords="offset points",
                    size=18,
                    ha="right",
                    va="center",
                    rotation=90,
                    weight="bold",
                )
            if mode_ind == 0:
                ax1.set_title(model, weight="bold", fontsize=15)

    fig.text(0.5, 0.0, "$\longrightarrow$ decreasing stationarity", ha="center")
    #     fig.text(0.5, 0.36, "$\longrightarrow$ decreasing stationarity", ha='center')
    #     fig.text(0.5, 0.7, "$\longrightarrow$ decreasing stationarity", ha='center')

    fig.text(
        0.07,
        0.5,
        "$\longleftarrow$ Increasing noise",
        va="center",
        rotation="vertical",
    )
    # make colorbar
    # fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.88, 0.05, 0.02, 0.9])
    fig.colorbar(imshow_plot, cax=cbar_ax, label=score_col)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def barplot_synthetic(results, score_col="RMSE"):
    # only look at local models
    subset = results[
        results["model"].isin(
            ["GWR", "RF", "spatial RF", "Kriging"]
        )  # "spatial RF",
    ]
    # subset.groupby(["nr_data", "data mode", "model", "noise_discrete", "locality_discrete"]).agg({"R2 score": "mean"})
    plt.figure(figsize=(18, 6))
    counter = 1
    modes = ["linear", "non-linear (simple)", "non-linear"]
    for mode, save_name in zip(
        modes, ["linear", "non_linear_simple", "non_linear"]
    ):
        plt.subplot(1, len(modes), counter)
        counter += 1
        subset_2 = subset[
            (subset["data mode"] == mode)
            & (subset["noise_discrete"] == "low")
            & (subset["locality_discrete"] == "high")
            & (subset["noise_type"] == "uniformly distributed")
        ]
        subset_2 = subset_2.groupby(["nr_data", "model"]).agg(
            {score_col: "mean"}
        )

        ax = sns.barplot(
            data=subset_2.reset_index(), x="nr_data", y=score_col, hue="model"
        )
        #     plt.ylim(0, 1)
        plt.xlabel("Number of samples")
        #     if mode == "non-linear (simple)":
        #         plt.legend(title="Model", loc="lower right", framealpha=1, ncol=2)
        #     else:
        plt.legend([], [], frameon=False)
        plt.title(mode + " DGP", fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    plt.figlegend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        labelspacing=0.0,
        bbox_to_anchor=(0.5, 1.09),
    )
    plt.savefig(f"outputs/barplot_main.pdf", bbox_inches="tight")
    plt.show()


def noise_analysis(results):
    fig = plt.figure(figsize=(12, 9))
    subset = results[
        results["model"].isin(
            ["OLS", "GWR", "RF (coordinates)", "Kriging"]
        )  # "spatial RF",
    ]
    counter = 1
    for i, mode in enumerate(
        ["linear", "non-linear (simple)"]
    ):  #  "non-linear (simple)" # linear",
        for j, model in enumerate(["GWR", "Kriging"]):
            subset2 = subset[
                (subset["model"] == model)
                & (subset["data mode"] == mode)
                &
                #         (subset["noise_discrete"] == "low") &
                (subset["locality_discrete"] == "high")
                &
                #         (subset["noise"] == 0.3) &
                #                 (subset["locality"] == 0.4) &
                #         (subset["noise_type"] == "constant") &
                (subset["nr_data"] == 500)
            ]
            subset2["noise_type"] = subset2["noise_type"] + " noise"
            ax = fig.add_subplot(2, 2, counter)
            sns.lineplot(
                ax=ax,
                data=subset2.reset_index(),
                x="noise",
                y="RMSE",
                hue="noise_type",
            )
            #         if counter == 1:
            #             plt.legend(title="Noise (spatial distribution)")# , loc=(1, 1))
            #         else:
            plt.legend([], [], frameon=False)
            if j == 0:
                ax.annotate(
                    mode,
                    xy=(0, 0.5),
                    xytext=(-20, 0),  # ax1.yaxis.labelpad - pad
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    size="large",
                    ha="right",
                    va="center",
                    rotation=90,
                    weight="bold",
                )
            ax.set_xlabel("Noise level ($\sigma$)")
            ax.set_ylim(0, 0.7)
            if counter in [2, 4]:
                plt.ylabel("")
            if counter in [1, 2]:
                plt.title(model, weight="bold", fontsize=16)
            counter += 1

    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    plt.figlegend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        labelspacing=0.0,
        bbox_to_anchor=(0.5, 1.07),
    )
    plt.savefig("outputs/noise_analysis.pdf", bbox_inches="tight")
    plt.show()
