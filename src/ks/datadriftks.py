import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from scipy.stats import ks_2samp

__all__ = ["DataDritDetectionKS"]


class DataDritDetectionKS(object):
    def __init__(self, p_value, save_fig_path, nbins=10):
        self._p_value = p_value
        self._save_fig_path = save_fig_path
        self._nbins = nbins

    def make_hist_info(self, data, use_prev_hist=False):
        if use_prev_hist:
            self._after_hist_info = self._prev_hist_info.copy()
            after_top = np.digitize(data, bins=self._prev_hist_info["bins"][1:-1], right=False)
            bin_count = {i: 0 for i in range(0, self._nbins)}
            unique_n, top = np.unique(after_top, return_counts=True)
            bin_count.update(dict(zip(unique_n, top)))
            self._after_hist_info["top"] = np.array(list(bin_count.values()))
            hist_info = self._after_hist_info
        else:
            n, bins = np.histogram(data, self._nbins)
            hist_info = self._make_vis_info(n, bins)
        return hist_info

    def add_prev_hist_info(self, hist_info):
        self._prev_hist_info = hist_info
        self._bins = len(self._prev_hist_info["bins"][1:])

    def _make_vis_info(self, n, bins):
        left = bins[:-1]
        right = bins[1:]
        bottom = np.zeros(len(left))
        top = bottom + n
        hist_info = dict(
            top=top,
            bottom=bottom,
            right=right,
            left=left,
            bins=bins,
        )
        return hist_info

    def _get_hist(self, hist_info, **kwargs):
        left = hist_info["left"]
        right = hist_info["right"]
        top = hist_info["top"] / sum(hist_info["top"])
        bottom = hist_info["bottom"]
        XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T
        barpath = path.Path.make_compound_path_from_polys(XY)
        patch = patches.PathPatch(barpath, **kwargs)
        return patch

    def get_min_max_x_y(self, prev_hist_info, after_hist_info):
        min_left = min(prev_hist_info["left"][0], after_hist_info["left"][0])
        max_right = max(prev_hist_info["right"][-1], after_hist_info["right"][-1])
        min_bottom = min(prev_hist_info["bottom"].min(), after_hist_info["bottom"].min())
        max_top = max(
            prev_hist_info["top"].max() / sum(prev_hist_info["top"]),
            after_hist_info["top"].max() / sum(after_hist_info["top"]),
        )
        return dict(left=min_left, right=max_right, bottom=min_bottom, top=max_top)

    def run_ks_test(self, prev_hist_info, after_hist_info):
        after_bins = after_hist_info["bins"][:-1][np.where(after_hist_info["top"] > 0, True, False)]
        ks_test = ks_2samp(prev_hist_info["bins"], after_bins)
        return ks_test

    def is_data_drift(self, prev_hist_info, after_hist_info):
        stat = self.run_ks_test(prev_hist_info, after_hist_info)
        if stat.pvalue < self._p_value:
            return True
        else:
            return False

    def visualize(self, prev_hist_info, after_hist_info, show=True, fig_kwargs=dict()):
        fig, ax = plt.subplots(**fig_kwargs)
        patch = self._get_hist(prev_hist_info, color="b", alpha=0.5)
        ax.add_patch(patch)

        patch = self._get_hist(after_hist_info, color="r", alpha=0.5)
        ax.add_patch(patch)
        hist_info = self.get_min_max_x_y(after_hist_info=after_hist_info, prev_hist_info=prev_hist_info)
        ax.set_xlim(hist_info["left"], hist_info["right"])
        ax.set_ylim(hist_info["bottom"], hist_info["top"])  # hist_info["top"]
        ks_test = self.run_ks_test(prev_hist_info, after_hist_info)
        ax.set_title(f"KS Statistics : {ks_test.statistic:.5f}, P-VALUE : {ks_test.pvalue:.5f}")
        plt.savefig(self._save_fig_path)
        if show:
            plt.show()
        else:
            plt.close()
