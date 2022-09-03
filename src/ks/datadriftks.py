import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from scipy.stats import ks_2samp

__all__ = ["DataDritDetectionKS"]


class DataDritDetectionKS(object):
    def __init__(self, p_value, save_fig_path):
        self._p_value = p_value
        self._save_fig_path = save_fig_path

    def make_hist_info(self, data, nbins=10):
        n, bins = np.histogram(data, nbins)
        hist_info = self._make_vis_info(n, bins)
        return hist_info

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

    def _get_hist(self, hist_info, **kwargs):
        left = hist_info["left"]
        right = hist_info["right"]
        top = hist_info["top"]
        bottom = hist_info["bottom"]
        XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T
        barpath = path.Path.make_compound_path_from_polys(XY)
        patch = patches.PathPatch(barpath, **kwargs)
        return patch

    def get_min_max_x_y(self, prev_hist_info, after_hist_info):
        min_left = min(prev_hist_info["left"][0], after_hist_info["left"][0])
        max_right = max(prev_hist_info["right"][-1], after_hist_info["right"][-1])
        min_bottom = min(prev_hist_info["bottom"].min(), after_hist_info["bottom"].min())
        max_top = max(prev_hist_info["top"].max(), after_hist_info["top"].max())
        return dict(left=min_left, right=max_right, bottom=min_bottom, top=max_top)

    def run_ks_test(self, prev_hist_info, after_hist_info):
        ks_test = ks_2samp(prev_hist_info["bins"], after_hist_info["bins"])
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
        ax.set_ylim(hist_info["bottom"], hist_info["top"])
        ks_test = self.run_ks_test(prev_hist_info, after_hist_info)
        ax.set_title(f"KS Statistics : {ks_test.statistic:.5f}, P-VALUE : {ks_test.pvalue:.5f}")
        plt.savefig(self._save_fig_path)
        if show:
            plt.show()
        else:
            plt.close()
