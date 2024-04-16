from pathlib import Path
from collections import defaultdict

from local2global_embedding.run.utils import ScriptParser, ResultsDict, load_data
from local2global_embedding.utils import flatten
from local2global_embedding.run.plot import mean_and_deviation, plot_with_errorbars
from matplotlib import pyplot as plt

key_to_label = {
    "scale": "l2g",
    "notranslate": "rotate-only",
    "norotate": "translate-only",
    "norotate_notranslate": "no-l2g"
}

plot_options = {
    "scale": dict(fmt='-', label='l2g', marker='>', color='tab:red', zorder=5),
    "notranslate": dict(fmt='--', marker='s', markersize=3, label='rotate-only', color='tab:orange', linewidth=0.5, zorder=3),
    "norotate": dict(fmt='-.', marker='d', markersize=3, label='translate-only', color='tab:purple', linewidth=0.5, zorder=2),
    "norotate_notranslate": dict(fmt=':', label='no-l2g', color='tab:pink', linewidth=0.5, zorder=1)
}


def plot(folder, dims=(8, 16, 128)):
    for dim in dims:
        folder = Path(folder)
        experiments_auc = defaultdict(lambda: defaultdict(dict))
        experiments_acc = defaultdict(lambda: defaultdict(dict))
        network_data = load_data(folder.name)
        num_labels = network_data.y.max().item() + 1
        patch_folders = folder.glob("*_patches")
        for pf in patch_folders:
            n = int(pf.name.split("_n", 1)[1].split("_", 1)[0])
            for ef in pf.iterdir():
                if ef.is_dir():
                    for data_file in ef.glob("*_l2g_*.json"):
                        model, key_part = data_file.stem.split("_l2g_", 1)
                        label = key_part.split("_eval", 1)[0]
                        experiment = ef.name + "_" + model
                        with ResultsDict(data_file, lock=False) as f:
                            experiments_acc[experiment][label][n] = f.get("acc", dim)
                            experiments_auc[experiment][label][n] = f.get("auc", dim)
        for key1, value in experiments_acc.items():
            plt.figure()
            for key2, opts in plot_options.items():
                value2 = value[key2]
                ns = sorted(value2.keys())
                v_mean, v_std = mean_and_deviation(value2[n] for n in ns)
                plot_with_errorbars(ns, v_mean, v_std, **opts)
            plt.legend()
            plt.xlabel("number of patches")
            plt.ylabel("classification accuracy")
            plt.gca().set_ylim(0.98 / num_labels, 1.02)
            plt.legend(ncol=2 , frameon=False)
            plt.savefig(folder / f"cl_d{dim}_{key1}.pdf")

        for key1, value in experiments_auc.items():
            plt.figure()
            for key2, opts in plot_options.items():
                value2 = value[key2]
                ns = sorted(value2.keys())
                v_mean, v_std = mean_and_deviation(value2[n] for n in ns)
                plot_with_errorbars(ns, v_mean, v_std, **opts)
            plt.legend()
            plt.xlabel("number of patches")
            plt.ylabel("AUC")
            plt.gca().set_ylim(0.48, 1.02)
            plt.legend(ncol=2, frameon=False)
            plt.savefig(folder / f"auc_d{dim}_{key1}.pdf")
            # print(key2)


if __name__ == "__main__":
    ScriptParser(plot).run()
