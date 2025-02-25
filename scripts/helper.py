import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from abtester.optimizers import *


def summarize_outcomes(y1, y0):
    mean_square_y0, mean_square_y1 = np.mean(y0**2), np.mean(y1**2)
    p_star = (1 + np.sqrt(mean_square_y0 / mean_square_y1)) ** (-1)
    correlation = np.mean(y0 * y1) / np.sqrt(mean_square_y0 * mean_square_y1)

    # Moments
    y0_second = np.sqrt(np.mean(y0**2))
    y1_second = np.sqrt(np.mean(y1**2))
    y0_fourth = (np.mean(y0**4)) ** 0.25
    y1_fourth = (np.mean(y1**4)) ** 0.25

    upper_bound = max(y0_fourth, y1_fourth)
    lower_bound = min(y0_second, y1_second)

    print("Mean ATE:", np.mean(y1 - y0))
    print("Upper moment bound:", upper_bound)
    print("Lower moment bound:", lower_bound)
    print("C/c ratio:", upper_bound / lower_bound)
    print("Rho:", correlation)

    print(
        "Check bounding assumptions:",
        0 < lower_bound,
        lower_bound < upper_bound,
        correlation > -(1 - lower_bound),
        lower_bound < 1,
    )

    print(
        "Check moment inequalities (should always hold):",
        y0_second < y0_fourth,
        y1_second < y1_fourth,
    )

    plt.hist(y1, bins=20, color="r", alpha=0.7, label="y1's Distribution")
    plt.hist(y0, bins=20, color="g", alpha=0.7, label="y0's Distribution")
    plt.title("Outcome Sequence Distributions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()


def get_sets(y1, y0):
    feature_functions = {
        # "difference": lambda x,y : x-y,
        # "variance": lambda x,y : x**2 + y**2,
        # 'ratio': lambda x,y : np.maximum(x**2/(y**2+1e-3),y**2/(x**2+1e-3))
        "neyman_p": lambda x, y: 1
        / (1 + np.sqrt((x**2 + 1e-6) / (y**2 + 1e-6)))
    }

    sets = [np.ones_like(y1, dtype="bool")[None, ...]]
    quantiles = [[0, 0.66], [0.33, 1.0]]
    for feature_name, feature_function in feature_functions.items():
        # Create sets based on quantile information
        feature = feature_function(y1, y0)
        for q_lb, q_ub in quantiles:
            sets.append(
                np.logical_and(
                    feature >= np.quantile(feature, q_lb),
                    feature <= np.quantile(feature, q_ub),
                )[None, ...]
            )
    sets = np.concatenate(sets, axis=0)
    return sets


################################################################################
# Plot results
################################################################################


def plot_neyman_regret(
    y1,
    y0,
    Pt_history_list,
    HT_history_list,
    label_list=["Our Method", "Dai et al. (2023)", "Our Method (AMGATE)"],
    ST=0,
    T_prime=None,
    subsample=1,
    window_size=1,
    save_dir=None,
    dataset_name=None,
):
    plt.figure(figsize=(8, 5))

    T = y1.shape[0]
    T_prime = T if T_prime is None else T_prime
    T_sub = int(T / subsample)

    pstar = 1 / (1 + np.sqrt(np.cumsum(y0**2) / np.cumsum(y1**2)))

    y1_arr = np.array(y1)
    y0_arr = np.array(y0)

    ftopt = np.array(
        [
            np.sum(y1_arr[: i + 1] ** 2) / pstar[i]
            + np.sum(y0_arr[: i + 1] ** 2) / (1 - pstar[i])
            for i in range(pstar.shape[0])
        ]
    )

    df_regret_rows = []

    # for Pt_history, label in zip(Pt_history_list, label_list):
    for Pt_history, HT_history, label in zip(
        Pt_history_list, HT_history_list, label_list
    ):

        ftp = np.cumsum(
            y1_arr.reshape(-1) ** 2 * np.mean(1 / Pt_history, axis=1)
        ) + np.cumsum(y0_arr.reshape(-1) ** 2 * np.mean(1 / (1 - Pt_history), axis=1))

        reg = ftp - ftopt

        smoothed_reg = np.convolve(
            reg / np.arange(1, T + 1),
            np.ones(window_size) / window_size,
            mode="valid",
        )

        plt.plot(
            smoothed_reg[ST : min(T_prime, T_sub)],
            label=label,
        )

        for i, val in enumerate(smoothed_reg):
            df_regret_rows.append([i + 1, label, val])

    plt.axhline(y=0, color="black", linestyle="--", alpha=0.3)

    plt.xlabel("Rounds (T)")
    plt.ylabel("Neyman regret")
    # plt.yscale("log")
    plt.legend()

    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir)  # saves as PNG by default
        plt.close()
    else:
        plt.show()

    # Save CSV with regret data
    save_dir_raw = "/".join(save_dir.split("/")[:-1]).replace("outputs", "outputs-raw")
    csv_name = (
        f"{dataset_name}_plot_neyman_regret_data.csv"
        if dataset_name
        else "plot_neyman_regret_data.csv"
    )
    save_dir_raw = os.path.join(save_dir_raw, csv_name)
    os.makedirs(os.path.dirname(save_dir_raw), exist_ok=True)
    df_regret = pd.DataFrame(df_regret_rows, columns=["Round", "Label", "NeymanRegret"])
    df_regret.to_csv(save_dir_raw, index=False)


def get_rescaled_neyman_variance(y1, y0):
    """Short helper: returns the 'neyman_rescaled' array used in plot_variance."""
    T = y1.shape[0]
    pstar = 1 / (1 + np.sqrt(np.cumsum(y0**2) / np.cumsum(y1**2)))
    y1_arr = np.array(y1)
    y0_arr = np.array(y0)
    ftopt = np.array(
        [
            np.sum(y1_arr[: i + 1] ** 2) / pstar[i]
            + np.sum(y0_arr[: i + 1] ** 2) / (1 - pstar[i])
            for i in range(pstar.shape[0])
        ]
    )
    # The expression for the theoretical rescaled variance
    var_rescaled = (ftopt - np.cumsum((y1_arr - y0_arr) ** 2)) / np.arange(1, T + 1)
    return var_rescaled


def plot_variance(
    y1,
    y0,
    Pt_history_list,
    HT_history_list,
    label_list=["Our Method", "Dai et al. (2023)", "Our Method (AMGATE)"],
    ST=500,
    T_prime=None,
    window_size=1,
    subsample=1,
    save_dir=None,
    dataset_name=None,
):
    T = y1.shape[0]
    T_prime = T if T_prime is None else T_prime
    T_sub = int(T / subsample)

    neyman_rescaled = np.array(get_rescaled_neyman_variance(y1, y0))

    plt.figure(figsize=(8, 5))
    plt.plot(neyman_rescaled[ST : min(T_prime, T_sub)], label="Neyman")

    # Collect data for CSV export
    df_variance_rows = []
    # First store the Neyman line data
    neyman_slice = neyman_rescaled[ST : min(T_prime, T_sub)]
    neyman_conv = np.convolve(
        neyman_slice, np.ones(window_size) / window_size, mode="valid"
    )
    for i, val in enumerate(neyman_conv):
        # Round starts at ST in the slice, then offset by i
        round_t = ST + i
        df_variance_rows.append([round_t, "Neyman", val])

    for Pt_history, HT_history, label in zip(
        Pt_history_list, HT_history_list, label_list
    ):
        var_rescaled = (
            np.cumsum(
                y1**2 * np.mean(1 / Pt_history, axis=1)
                + y0**2 * np.mean(1 / (1 - Pt_history), axis=1)
            )
            - np.cumsum((y1 - y0) ** 2)
        ) / np.arange(1, T + 1)

        var_slice = var_rescaled[ST : min(T_prime, T_sub)]
        var_conv = np.convolve(
            var_slice, np.ones(window_size) / window_size, mode="valid"
        )

        for i, val in enumerate(var_conv):
            round_t = ST + i
            df_variance_rows.append([round_t, label, val])

        plt.plot(
            var_conv,
            label=label,
        )

    plt.xlabel("Rounds (T)")
    plt.ylabel("Rescaled Variance")
    plt.legend()

    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir)  # saves as PNG by default
        plt.close()
    else:
        plt.show()

    # Save CSV with regret data
    save_dir_raw = "/".join(save_dir.split("/")[:-1]).replace("outputs", "outputs-raw")
    csv_name = (
        f"{dataset_name}_plot_variance_data.csv"
        if dataset_name
        else "plot_variance_data.csv"
    )
    save_dir_raw = os.path.join(save_dir_raw, csv_name)
    os.makedirs(os.path.dirname(save_dir_raw), exist_ok=True)
    df_regret = pd.DataFrame(df_variance_rows, columns=["Round", "Label", "Variance"])
    df_regret.to_csv(save_dir_raw, index=False)


def plot_combined_treatment_prob(
    y1,
    y0,
    Pt_history_list,
    label_list=["Our Method", "Dai et al. (2023)", "Our Method (AMGATE)"],
    save_dir=None,
    dataset_name=None,
):
    plt.figure(figsize=(8, 5))
    df_prob_rows = []

    # Plot each method's treatment probability (smoothed)
    for Pt_history, label in zip(Pt_history_list, label_list):
        # Smooth
        smoothed_prob = np.convolve(
            Pt_history.T[0],
            1,
            # np.ones(100) / 100,
            mode="valid",
        )
        # Plot
        plt.plot(smoothed_prob, label=label)
        # Collect data for CSV
        for i, val in enumerate(smoothed_prob):
            df_prob_rows.append([i + 1, label, val])

    # Plot Neyman (not smoothed)
    neyman_probs = get_neyman_prob(y1, y0)
    plt.plot(neyman_probs, label="Neyman")
    for i, val in enumerate(neyman_probs):
        df_prob_rows.append([i + 1, "Neyman", val])

    plt.xlabel("Rounds (T)")
    plt.ylabel("Treatment Probability")
    plt.legend()

    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir)  # saves as PNG by default
        plt.close()
    else:
        plt.show()

    # Save CSV with treatment probability data
    if save_dir:
        # Match the style of saving used in other plots
        save_dir_raw = "/".join(save_dir.split("/")[:-1]).replace(
            "outputs", "outputs-raw"
        )
        csv_name = (
            f"{dataset_name}_plot_combined_treatment_prob_data.csv"
            if dataset_name
            else "plot_combined_treatment_prob_data.csv"
        )
        save_dir_raw = os.path.join(save_dir_raw, csv_name)
        os.makedirs(os.path.dirname(save_dir_raw), exist_ok=True)
        df_prob = pd.DataFrame(
            df_prob_rows, columns=["Round", "Label", "TreatmentProbability"]
        )
        df_prob.to_csv(save_dir_raw, index=False)


def plot_variance_regret_multi(
    y1,
    y0,
    HT_history_list,
    label_list=None,
    ST=0,
    T_prime=None,
    subsample=1,
    save_dir=None,
    dataset_name=None,
):
    T = len(y1)
    T_prime = T if T_prime is None else T_prime
    T_sub = int(T / subsample)

    neyman_rescaled = get_rescaled_neyman_variance(y1, y0)
    plt.plot(
        neyman_rescaled[ST : min(T_prime, T_sub)], "r", alpha=0.7, label="Opt (Neyman)"
    )

    for HT_history, label in zip(HT_history_list, label_list):
        var_history = np.var(HT_history, axis=0)
        var_rescaled = var_history * np.arange(1, T_sub + 1)
        plt.plot(var_rescaled[ST : min(T_prime, T_sub)], alpha=0.7, label=label)

    plt.xlabel("Time Steps")
    plt.ylabel("Rescaled Variance")
    plt.legend()

    title_suffix = f"{dataset_name}" if dataset_name else ""
    plt.title(f"Variance Regret on Dataset: {title_suffix}")
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir)
        plt.cla()
        plt.close()
    else:
        plt.show()


################################################################################
# Subroutines testing for dataset
################################################################################


def run_experiment(y1, y0, sets, reps, dataset_name=None, savefig_dir=None):
    # if dataset_name:
    # print(f"Running new method on dataset: {dataset_name}")

    # shuffle indices
    idx = np.random.permutation(len(y1))
    y1, y0 = y1[idx], y0[idx]

    Ht_history_list, Pt_history_list, Zt_history_list, label_list = [], [], [], []

    if sets is not None:
        # Group-conditional ATE
        Pt_history, Zt_history, Yt_history, Ptg_raw, _ = gd_amgate(
            y1, y0, sets, dim=reps
        )
        Ht_history = HT_eval_iter(
            Pt_history, Yt_history, Zt_history
        )  # Running HT estimate

        Ht_history_list.append(Ht_history)
        Pt_history_list.append(Pt_history)
        Zt_history_list.append(Zt_history)
        label_list.append("MGATE")

    # Group-agnostic ATE
    Pt_history, Zt_history, Yt_history = gd(y1, y0, dim=reps)
    Ht_history = HT_eval_iter(Pt_history, Yt_history, Zt_history)  # Running HT estimate

    Ht_history_list.append(Ht_history)
    Pt_history_list.append(Pt_history)
    Zt_history_list.append(Zt_history)
    label_list.append(r"ClipOGD$^\mathrm{SC}$")

    # Clip-OGD
    Pt_history, Zt_history, Yt_history = gd(y1, y0, dim=reps, old=True)
    Ht_history = HT_eval_iter(Pt_history, Yt_history, Zt_history)

    Ht_history_list.append(Ht_history)
    Pt_history_list.append(Pt_history)
    Zt_history_list.append(Zt_history)
    label_list.append(r"ClipOGD$^0$")

    if sets is not None:
        # Adding per group probability plots here
        for g in range(sets.shape[0]):
            Pt_history_list_g = [
                Pt_history[sets[g], :] for Pt_history in Pt_history_list
            ]
            Zt_history_list_g = [
                Zt_history[sets[g], :] for Zt_history in Zt_history_list
            ]
            Pt_history_list_g += [Ptg_raw[sets[g], g, :]]

            # HT_history_list_g = HT_eval_iter(Pt_history, Yt_history, Zt_history)
            y1_g = y1[sets[g]]
            y0_g = y0[sets[g]]

            Ht_history_list_g = [
                HT_eval_iter(
                    Pt_history,
                    y1_g[..., None] * Zt_history + y0_g[..., None] * (1 - Zt_history),
                    Zt_history,
                )
                for Pt_history, Zt_history in zip(Pt_history_list_g, Zt_history_list_g)
            ]

            # Recreate running average treatment effect estimates using Horvitz-Thompson weighting.
            # group_weights_g=group_weights_raw[sets[g],...].mean(-1)
            # g_sets = sets[:,sets[g]]
            plot_combined_treatment_prob(
                y1_g,
                y0_g,
                Pt_history_list_g,
                label_list=label_list + ["Group-specific expert"],
                save_dir=os.path.join(
                    savefig_dir, f"grouped/{dataset_name}_pt_combined_{g}.png"
                ),
                dataset_name=f"group_{g}",
            )
            plot_neyman_regret(
                y1_g,
                y0_g,
                Pt_history_list_g,
                Ht_history_list_g,
                label_list=label_list + ["Group-specific expert"],
                ST=100,
                T_prime=None,
                subsample=1,
                window_size=100,
                save_dir=os.path.join(
                    savefig_dir, f"grouped/{dataset_name}_regret_{g}.png"
                ),
                dataset_name=f"group_{g}",
            )
            plot_variance(
                y1_g,
                y0_g,
                Pt_history_list_g,
                Ht_history_list_g,
                label_list=label_list,
                save_dir=os.path.join(
                    savefig_dir, f"grouped/{dataset_name}_variance_{g}.png"
                ),
                dataset_name=f"group_{g}",
            )

    plot_combined_treatment_prob(
        y1,
        y0,
        Pt_history_list,
        label_list=label_list,
        save_dir=os.path.join(
            savefig_dir, f"{dataset_name}_pt_combined_multigroup.png"
        ),
        dataset_name=dataset_name,
    )
    plot_variance(
        y1,
        y0,
        Pt_history_list,
        Ht_history_list,
        label_list=label_list,
        save_dir=os.path.join(savefig_dir, f"{dataset_name}_variance_multigroup.png"),
        dataset_name=dataset_name,
    )
    plot_neyman_regret(
        y1,
        y0,
        Pt_history_list,
        Ht_history_list,
        label_list=label_list,
        ST=100,
        T_prime=None,
        subsample=1,
        window_size=100,
        save_dir=os.path.join(savefig_dir, f"{dataset_name}_regret_multigroup.png"),
        dataset_name=None,
    )

    return Ht_history_list, label_list


def run_experiment_ppi(y1, y0, hint1, hint0, reps, dataset_name=None, savefig_dir=None):
    y1_ppi, y0_ppi = y1 - hint1, y0 - hint0

    neyman_orig = get_rescaled_neyman_variance(y1, y0)
    neyman_ppi = get_rescaled_neyman_variance(y1_ppi, y0_ppi)

    Pt_new, Zt_new, Yt_new = gd(y1, y0, dim=reps)
    HT_new = HT_eval_iter(Pt_new, Yt_new, Zt_new)

    Pt_old, Zt_old, Yt_old = gd(y1, y0, dim=reps, old=True)
    HT_old = HT_eval_iter(Pt_old, Yt_old, Zt_old)

    Pt_new_ppi, Zt_new_ppi, Yt_new_ppi = gd(y1_ppi, y0_ppi, dim=reps)
    HT_new_ppi = HT_eval_iter(Pt_new_ppi, Yt_new_ppi, Zt_new_ppi)

    Pt_old_ppi, Zt_old_ppi, Yt_old_ppi = gd(y1_ppi, y0_ppi, dim=reps, old=True)
    HT_old_ppi = HT_eval_iter(Pt_old_ppi, Yt_old_ppi, Zt_old_ppi)

    var_new = np.var(HT_new, axis=0)
    var_old = np.var(HT_old, axis=0)
    var_new_ppi = np.var(HT_new_ppi, axis=0)
    var_old_ppi = np.var(HT_old_ppi, axis=0)

    scale = np.arange(1, len(y1) + 1)
    var_new_rescaled = var_new * scale
    var_old_rescaled = var_old * scale
    var_new_ppi_rescaled = var_new_ppi * scale
    var_old_ppi_rescaled = var_old_ppi * scale

    fig, ax = plt.subplots()

    ax.plot(neyman_orig, "r", alpha=0.7, label="Optimal (Neyman)")
    ax.plot(var_new_rescaled, "g", alpha=0.7, label="Our method")
    ax.plot(var_old_rescaled, "b", alpha=0.7, label="Dai et al. (2023)")

    ax.plot(neyman_ppi, "r", linestyle="dashed", alpha=0.7, label="Opt (Neyman) PPI")
    ax.plot(
        var_new_ppi_rescaled, "g", linestyle="dashed", alpha=0.7, label="Our method PPI"
    )
    ax.plot(
        var_old_ppi_rescaled,
        "b",
        linestyle="dashed",
        alpha=0.7,
        label="Dai et al. (2023) PPI",
    )

    title_suffix = dataset_name if dataset_name else ""
    plt.title(f"PPI vs. no PPI for LLM Accuracy: {title_suffix}")
    ax.set_xlabel("Rounds (T)")
    ax.set_ylabel("Rescaled Variance")
    ax.legend()

    if savefig_dir:
        out_file = os.path.join(savefig_dir, f"{dataset_name}_regret_ppi_comp.png")
        plt.savefig(out_file, bbox_inches="tight")
    else:
        plt.show()

    return (
        Pt_new,
        Pt_old,
        HT_new,
        HT_old,
        Pt_new_ppi,
        Pt_old_ppi,
        HT_new_ppi,
        HT_old_ppi,
    )


################################################################################
# Testing
################################################################################


def test_microfinance(data_dir, outputs_dir, reps, add_sets=True):
    csv_path = os.path.join(data_dir, "microfinance", "po-df.csv")
    df = pd.read_csv(csv_path).dropna()
    y1, y0 = df["y1"].values, df["y0"].values
    idx = np.random.permutation(len(y1))
    y1, y0 = y1[idx], y0[idx]
    sets = get_sets(y1, y0) if add_sets else None
    run_experiment(
        y1,
        y0,
        sets=sets,
        reps=reps,
        dataset_name="microfinance",
        savefig_dir=os.path.join(outputs_dir, "microfinance"),
    )


def test_gaussian(data_dir, outputs_dir, reps, T, add_sets=True):
    data_dir = os.path.join(data_dir, "gaussian")
    outputs_dir = os.path.join(outputs_dir, "gaussian")

    for dataset_name in tqdm(os.listdir(data_dir)):
        path_csv = os.path.join(data_dir, dataset_name)
        df = pd.read_csv(path_csv)
        y1, y0 = df["y1"][:T], df["y0"][:T]

        save_dir = os.path.join(outputs_dir, dataset_name)
        sets = get_sets(y1, y0) if add_sets else None
        run_experiment(
            y1,
            y0,
            sets=sets,
            reps=reps,
            dataset_name=dataset_name,
            savefig_dir=save_dir,
        )


def test_synthetic(data_dir, outputs_dir, reps, add_sets=True):
    data_dir = os.path.join(data_dir, "synthetic")
    outputs_dir = os.path.join(outputs_dir, "synthetic")

    for dataset_name in os.listdir(data_dir):
        path_csv = os.path.join(data_dir, dataset_name)
        df = pd.read_csv(path_csv)
        y1, y0 = df["y1"], df["y0"]

        save_dir = os.path.join(outputs_dir, dataset_name)
        sets = get_sets(y1, y0) if add_sets else None
        run_experiment(
            y1,
            y0,
            sets=sets,
            reps=reps,
            dataset_name=dataset_name,
            savefig_dir=save_dir,
        )


def test_asos(data_dir, outputs_dir, reps, add_sets=True):
    data_dir = os.path.join(data_dir, "asos")
    outputs_dir = os.path.join(outputs_dir, "asos")

    num_metrics = 4
    for i in range(num_metrics):
        csv_path = os.path.join(data_dir, f"metric_{i}.csv")
        df = pd.read_csv(csv_path)
        y1, y0 = df["y1"], df["y0"]

        save_dir = os.path.join(outputs_dir, f"metric_{i}")

        sets = get_sets(y1, y0) if add_sets else None
        run_experiment(
            y1, y0, sets=sets, dataset_name=f"ASOS_Metric_{i}", savefig_dir=save_dir
        )


def test_upworthy(data_dir, outputs_dir, reps, add_sets=True):
    data_dir = os.path.join(data_dir, "upworthy")
    outputs_dir = os.path.join(outputs_dir, "upworthy")

    tasks = ["PE", "NE", "Q"]
    for task in tasks:
        for k in [1, 10]:
            csv_path = os.path.join(data_dir, f"upworthy_{task}.csv")
            df = pd.read_csv(csv_path)
            y1, y0 = k * df["y1"], k * df["y0"]

            save_dir = os.path.join(outputs_dir, task)
            sets = get_sets(y1, y0) if add_sets else None
            run_experiment(
                y1,
                y0,
                sets=sets,
                dataset_name=f"Upworthy_{task}_{k}",
                savefig_dir=save_dir,
            )


def test_adaptive(data_dir, outputs_dir, reps, add_sets=True):
    # data_dir = os.path.join(data_dir, "adaptive_sequence")
    outputs_dir = os.path.join(outputs_dir, "adaptive_sequence")

    df = pd.read_csv(
        os.path.join(data_dir, "adaptive-experiment", "adaptive_sequence.csv")
    )
    y1, y0 = df["y1"], df["y0"]
    save_dir = os.path.join(outputs_dir, f"metric")
    sets = get_sets(y1, y0) if add_sets else None
    run_experiment(
        y1,
        y0,
        sets=sets,
        dataset_name=f"test",
        savefig_dir=save_dir,
    )


def test_llmbench(data_dir, outputs_dir, reps, add_sets=True):
    llmbench_dir = os.path.join(data_dir, "llmbench")
    out_dir = os.path.join(outputs_dir, "llmbench")
    dataset_names = sorted(os.listdir(llmbench_dir))

    for dataset_name in tqdm(dataset_names):
        # Accuracy
        acc_dir = os.path.join(llmbench_dir, dataset_name, "accuracies")
        acc_csv = os.path.join(acc_dir, os.listdir(acc_dir)[0])
        df_acc = pd.read_csv(acc_csv)
        y1_acc, y0_acc = df_acc["y1"] - 0.5, df_acc["y0"] - 0.5

        save_dir_acc = os.path.join(out_dir, dataset_name, "accuracies")
        # print(f"Accuracy ATE on dataset: {dataset_name}")
        if len(y1_acc) < 700:
            continue
        sets = None
        run_experiment(
            y1_acc,
            y0_acc,
            sets=sets,
            dataset_name=dataset_name,
            savefig_dir=save_dir_acc,
        )

        # Confidence
        conf_dir = os.path.join(llmbench_dir, dataset_name, "confidences")
        conf_csv = os.path.join(conf_dir, os.listdir(conf_dir)[0])
        df_conf = pd.read_csv(conf_csv)
        y1_conf, y0_conf = df_conf["y1"], df_conf["y0"]

        save_dir_conf = os.path.join(out_dir, dataset_name, "confidences")
        sets = get_sets(y1_conf, y0_conf) if add_sets else None
        run_experiment(
            y1_conf,
            y0_conf,
            sets=sets,
            reps=reps,
            dataset_name=dataset_name,
            savefig_dir=save_dir_conf,
        )
