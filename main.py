import sys
import numpy as np

sys.path.append("./")
from src.ks.datadriftks import DataDritDetectionKS
from src.utils import PickleHandler


if __name__ == "__main__":
    ks = DataDritDetectionKS(p_value=0.05, save_fig_path="./hist.png")

    ## past 1 month ago
    prev_data = np.random.normal(loc=5, scale=1.5, size=1000)
    prev_hist_info = ks.make_hist_info(prev_data, nbins=10)
    ## save information any format
    PickleHandler.save_pickle("./past_1_month_hist_info.pkl", prev_hist_info)
    del prev_hist_info
    ## current month
    after_data = np.random.normal(loc=6, scale=1.0, size=10000)
    after_hist_info = ks.make_hist_info(after_data, nbins=10)

    prev_hist_info = PickleHandler.load_pickle("./past_1_month_hist_info.pkl")
    print(ks.is_data_drift(prev_hist_info=prev_hist_info, after_hist_info=after_hist_info))
    ks.visualize(prev_hist_info=prev_hist_info, after_hist_info=after_hist_info, show=False)
