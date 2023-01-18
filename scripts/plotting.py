import matplotlib.pyplot as plt
import numpy as np


def main_plot(
    results,
    nr_data=500,
    noise_level_range=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    locality_range=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
):
    results.loc[results["model"] == "linear regression", "model"] = "OLS"
    include_models = [
        "OLS",
        "SAR",
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
    fig = plt.figure(figsize=(16, 9))
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
            ]
            results_filter.set_index(["noise", "locality"], inplace=True)
            visualize_scores = np.zeros(
                (len(noise_level_range), len(locality_range))
            )
            for i, noise in enumerate(noise_level_range):
                for j, locality in enumerate(locality_range):
                    score = results_filter.loc[noise, locality][
                        "R2 score"
                    ].mean()
                    visualize_scores[i, j] = score

            ax1 = fig.add_subplot(
                3, nr_models + 1, ((nr_models + 1) * mode_ind) + model_ind + 1
            )
            imshow_plot = ax1.imshow(visualize_scores, vmin=0, vmax=1)
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
                ax1.annotate(
                    mode,
                    xy=(0, 0.5),
                    xytext=(-40, 0),  # ax1.yaxis.labelpad - pad
                    xycoords=ax1.yaxis.label,
                    textcoords="offset points",
                    size="large",
                    ha="right",
                    va="center",
                    rotation=90,
                    weight="bold",
                )
            #             if mode_ind == 0:
            ax1.set_title(model, weight="bold", fontsize=15)

    fig.text(0.5, 0.0, "$\longrightarrow$ decreasing stationarity", ha="center")
    fig.text(
        0.5, 0.36, "$\longrightarrow$ decreasing stationarity", ha="center"
    )
    fig.text(0.5, 0.7, "$\longrightarrow$ decreasing stationarity", ha="center")

    fig.text(
        0.04,
        0.5,
        "$\longleftarrow$ Increasing noise",
        va="center",
        rotation="vertical",
    )
    # make colorbar
    # fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.88, 0.05, 0.02, 0.9])
    fig.colorbar(imshow_plot, cax=cbar_ax)
    plt.tight_layout()
    plt.savefig("outputs/main_plot.pdf")
    plt.show()
