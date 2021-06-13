import os
import gc
import math
import numpy as np
import pandas
from common import *
import sf_calc

try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except:
    from tqdm import tqdm

class WeightManager:
    @staticmethod
    def CreateBins():
        pt_bins = [ ]
        pt_bins.extend(list(np.arange(20, 50, 5)))
        pt_bins.extend(list(np.arange(50, 100, 10)))
        pt_bins.extend(list(np.arange(100, 200, 20)))
        pt_bins.extend(list(np.arange(200, 500, 50)))
        pt_bins.extend(list(np.arange(500, 1000, 100)))
        pt_bins.append(1000)

        eta_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3]

        pteta_bins = []
        for pt_bin in range(len(pt_bins) - 1):
            for eta_bin in range(len(eta_bins) - 1):
                pteta_bins.append([ pt_bins[pt_bin], pt_bins[pt_bin + 1], eta_bins[eta_bin], eta_bins[eta_bin + 1] ])

        return np.array(pt_bins), np.array(eta_bins), np.array(pteta_bins)

    @staticmethod
    def CreateWeightDataFrame(full_file_name, Y, pt_bins, eta_bins):
        weight_df = ReadBrancesToDataFrame(full_file_name, 'taus', ['pt', 'eta'])
        weight_result = sf_calc.ApplyUniformWeights(pt_bins, eta_bins, weight_df.pt.values, weight_df.eta.values, Y)
        weight_df['weight'] = pandas.Series(weight_result[0], index=weight_df.index)
        bin_ids = weight_result[1]
        weight_df["pt_bin_ids"] = pandas.Series(bin_ids[:, 0], index=weight_df.index, dtype=int)
        weight_df["eta_bin_ids"] = pandas.Series(bin_ids[:, 1], index=weight_df.index, dtype=int)

        for n in range(len(match_suffixes)):
            br_suff = match_suffixes[n]
            weight_df['gen_'+br_suff] = pandas.Series(Y[:, n], index=weight_df.index)

        return weight_df

    def __init__(self, weight_file_name, calc_weights=False, full_file_name = None, Y = None, first_block = True):
        self.pt_bins, self.eta_bins, self.pteta_bins = WeightManager.CreateBins()
        if calc_weights:
            if (full_file_name is None) or (Y is None):
                raise RuntimeError("Missing information which is needed to calculate the weights.")
            self.weight_df = WeightManager.CreateWeightDataFrame(full_file_name, Y, self.pt_bins, self.eta_bins)
            self.SaveWeights(weight_file_name)
        else:
            self.weight_df = pandas.read_hdf(weight_file_name, 'weights')

        if first_block:
            for cl in ['e', 'mu', 'jet']:
                self.weight_df["tau_vs_" + cl] = pandas.Series(np.zeros(self.weight_df.shape[0]),
                                                               index=self.weight_df.index)
                self.weight_df["weight_" + cl] = pandas.Series(np.copy(self.weight_df.weight.values),
                                                               index=self.weight_df.index)

        self.sum_tau_weights = self.weight_df[self.weight_df.gen_tau == 1].weight.sum()

        gc.collect()

    def GetWeights(self, start, stop):
        return self.weight_df[["weight_e", "weight_mu", "weight_jet"]].values[start:stop, :]

    def SetHistFileName(self, hist_file_name, overwrite=True):
        self.hist_file_name = hist_file_name
        if hist_file_name is not None and os.path.isfile(hist_file_name):
            os.remove(hist_file_name)

    def SaveWeights(self, weight_file_name):
        self.weight_df.to_hdf(weight_file_name, 'weights', mode='w', format='fixed', complevel=1)

    def UpdateWeights(self, model, epoch, X, test_start, n_test, sf_inputs, class_target_eff, batch_size=100000):
        pred = model.predict([X[test_start:test_start+n_test], self.GetWeights(test_start, test_start+n_test),
                              sf_inputs[test_start:test_start+n_test]],
                             batch_size = batch_size, verbose=0)
        print("\tpredictions has been calculated.")

        n_bins = self.pteta_bins.shape[0]
        n_updates = n_bins * len(class_target_eff)
        df_update = pandas.DataFrame(data ={
            'epoch': np.ones(n_updates, dtype=int) * epoch,
            'cl_idx': np.ones(n_updates, dtype=int) * (-1),
            'target_eff': np.zeros(n_updates),
            'threashold': np.zeros(n_updates),
            'pt_bin_id': np.zeros(n_updates, dtype=int),
            'eta_bin_id': np.zeros(n_updates, dtype=int),
            'pt_min': np.zeros(n_updates),
            'pt_max': np.zeros(n_updates),
            'eta_min': np.zeros(n_updates),
            'eta_max': np.zeros(n_updates),
            'is_updated': np.ones(n_updates, dtype=int),
            'sf': np.ones(n_updates),
            'eff': np.zeros(n_updates),
            'eff_err': np.zeros(n_updates),
            'n_taus': np.zeros(n_updates, dtype=int),
            'n_passed': np.ones(n_updates),
        })

        upd_idx = 0

        test_sel = (self.weight_df.index >= test_start) & (self.weight_df.index < test_start + n_test)
        tau_sel = test_sel & (self.weight_df.gen_tau == 1)

        thr = np.zeros(3)
        all_target_eff = np.zeros(3)

        for cl, target_eff in class_target_eff:
            br_loc = self.weight_df.columns.get_loc('tau_vs_'+cl)
            cl_idx = match_suffixes.index(cl)
            tau_vs_cl = TauLosses.tau_vs_other(pred[:, tau], pred[:, cl_idx])
            self.weight_df.iloc[test_start:test_start+n_test, br_loc] = tau_vs_cl
            df_tau = self.weight_df[tau_sel]
            cl_idx = min(cl_idx, 2)
            thr[cl_idx] = np.percentile(df_tau["tau_vs_" + cl], (1 - target_eff) * 100)
            #thr[cl_idx] = quantile_ex(df_tau["tau_vs_" + cl].values, 1 - target_eff, df_tau.weight.values)
            all_target_eff[cl_idx] = target_eff

        sf_results = sf_calc.CalculateScaleFactors(self.pt_bins, self.eta_bins,
                self.weight_df.tau_vs_e.values, self.weight_df.tau_vs_mu.values, self.weight_df.tau_vs_jet.values,
                self.weight_df.gen_tau.values, self.weight_df.weight.values, thr, all_target_eff, test_start,
                n_test, self.weight_df.pt_bin_ids.values, self.weight_df.eta_bin_ids.values)
        weights_changed = np.count_nonzero(sf_results[:, :, :, 0]) > 0

        if weights_changed:
            new_weights = sf_calc.ApplyScaleFactors(self.weight_df.pt_bin_ids.values,
                    self.weight_df.eta_bin_ids.values, sf_results, self.weight_df.gen_tau.values,
                    self.GetWeights(0, self.weight_df.shape[0]), self.weight_df.weight.values,
                    self.sum_tau_weights, 10)
            for cl, target_eff in class_target_eff:
                cl_idx = min(match_suffixes.index(cl), 2)
                if np.count_nonzero(sf_results[cl_idx, :, :, 0]) > 0:
                    w_br_loc = self.weight_df.columns.get_loc('weight_' + cl)
                    self.weight_df.iloc[:, w_br_loc] = new_weights[:, cl_idx]

        for cl, target_eff in class_target_eff:
            cl_idx = min(match_suffixes.index(cl), 2)
            for pt_bin_id in range(len(self.pt_bins) - 1):
                for eta_bin_id in range(len(self.eta_bins) - 1):
                    df_update.loc[upd_idx, 'cl_idx'] = cl_idx
                    df_update.loc[upd_idx, 'target_eff'] = target_eff
                    df_update.loc[upd_idx, 'threashold'] = thr[cl_idx]
                    df_update.loc[upd_idx, 'pt_bin_id'] = pt_bin_id
                    df_update.loc[upd_idx, 'eta_bin_id'] = eta_bin_id
                    df_update.loc[upd_idx, 'pt_min'] = self.pt_bins[pt_bin_id]
                    df_update.loc[upd_idx, 'pt_max'] = self.pt_bins[pt_bin_id+1]
                    df_update.loc[upd_idx, 'eta_min'] = self.eta_bins[eta_bin_id]
                    df_update.loc[upd_idx, 'eta_max'] = self.eta_bins[eta_bin_id+1]

                    df_update.loc[upd_idx, 'is_updated'] = sf_results[cl_idx, pt_bin_id, eta_bin_id, 0]
                    df_update.loc[upd_idx, 'sf'] = sf_results[cl_idx, pt_bin_id, eta_bin_id, 1]
                    df_update.loc[upd_idx, 'n_taus'] = sf_results[cl_idx, pt_bin_id, eta_bin_id, 4]
                    df_update.loc[upd_idx, 'n_passed'] = sf_results[cl_idx, pt_bin_id, eta_bin_id, 7]
                    df_update.loc[upd_idx, 'eff'] = sf_results[cl_idx, pt_bin_id, eta_bin_id, 2]
                    df_update.loc[upd_idx, 'eff_err'] = sf_results[cl_idx, pt_bin_id, eta_bin_id, 3]

                    upd_idx += 1


            cl_update = df_update[df_update.cl_idx == cl_idx]
            cl_updated = cl_update[cl_update.is_updated > 0]
            if cl_updated.shape[0] > 0:
                average_sf = np.average(cl_updated.sf, weights=cl_updated.n_taus)
            else:
                average_sf = 0
            print('tau_vs_{}: bins changed = {}, average sf = {}'.format(cl, cl_updated.shape[0], average_sf))
        if self.hist_file_name is not None:
            df_update.to_hdf(self.hist_file_name, "weight_updates", append=True, complevel=1, complib='zlib')
        gc.collect()
