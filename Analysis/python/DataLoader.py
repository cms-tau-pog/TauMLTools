import glob
import math
from concurrent.futures import ThreadPoolExecutor
#from itertools import izip
import numpy as np
import uproot
from common import *

class DataLoader:
    @staticmethod
    def GetNumberOfEntries(file_name, tree_name):
        with uproot.open(file_name) as file:
            tree = file[tree_name]
            return tree.numentries

    def __init__(self, file_name_pattern, batch_size, n_cells, has_validation_set = True, n_passes = -1,
                 allow_partial_batches = False):
        all_files = [ f.replace('\\', '/') for f in sorted(glob.glob(file_name_pattern)) ]
        if (has_validation_set and len(all_files) < 2) or len(all_files) < 1:
            raise RuntimeError("Too few input files")
        if batch_size <= 0:
            raise RuntimeError("batch_size should be a positive number")
        self.batch_size = batch_size
        self.n_cells = n_cells
        self.has_validation_set = has_validation_set
        self.n_passes = n_passes
        self.allow_partial_batches = allow_partial_batches
        first_data_file_index = 0
        if has_validation_set:
            self.validation_files = all_files[0:1]
            self.validation_size = DataLoader.GetNumberOfEntries(self.validation_files[0], 'taus')
            first_data_file_index = 1
        self.data_files = all_files[first_data_file_index:]
        self.data_file_sizes = np.zeros(len(self.data_files), dtype=int)
        for n in range(len(self.data_files)):
            self.data_file_sizes[n] = DataLoader.GetNumberOfEntries(self.data_files[n], 'taus')
            if self.data_file_sizes[n] < batch_size:
                raise RuntimeError("File {} has too few events.".format(self.data_files[n]))
        self.data_size = self.data_file_sizes.sum()
        self.steps_per_epoch = math.floor(self.data_size / self.batch_size)
        if has_validation_set:
            self.validation_steps = math.floor(self.validation_size / self.batch_size)
        self.executor = ThreadPoolExecutor(12)

    def generator(self, primary_set = True, return_truth = True, return_weights = True):
        if primary_set:
            files = self.data_files
        else:
            files = self.validation_files

        def df_taus_gen():
            for df_taus in uproot.iterate(files, 'taus', df_tau_branches,
                                          entrysteps=self.batch_size, executor=self.executor):
                #df_taus.columns = [ c.decode('utf-8') for c in df_taus.columns ]
                yield df_taus

        def df_cells_gen():
            for df_cells in uproot.iterate(files, 'cells', df_cell_branches,
                                          entrysteps=self.batch_size * self.n_cells, executor=self.executor):
                #df_cells.columns = [ c.decode('utf-8') for c in df_cells.columns ]
                yield df_cells

        pfCand_branches = input_cell_common_branches + input_cell_pfCand_branches
        ele_branches = input_cell_common_branches + input_cell_ele_branches
        muon_branches = input_cell_common_branches + input_cell_muon_branches

        min_value, max_value = -1000, 1000

        current_pass = 0
        while self.n_passes < 0 or current_pass < self.n_passes:
            for df_taus, df_cells in zip(df_taus_gen(), df_cells_gen()):
                if not self.allow_partial_batches and df_taus[b'tau_pt'].shape[0] != self.batch_size: continue

                b_size = df_taus[b'tau_pt'].shape[0]
                X_taus = np.empty((b_size, len(input_tau_branches)))
                Y = np.empty((b_size, n_outputs), dtype=int)
                if return_weights:
                    weights = np.empty(b_size)

                for n in range(len(input_tau_branches)):
                    X_taus[:, n] = df_taus[input_tau_branches[n].encode()].clip(min_value, max_value)

                for n in range(len(truth_branches)):
                    Y[:, n] = df_taus[truth_branches[n].encode()]

                if return_weights:
                    weights[:] = df_taus[weight_branches[0].encode()]
                    weights = weights / np.sum(weights) * weights.shape[0]

                X_cells_pfCand = np.empty((b_size, n_cells_eta, n_cells_phi, len(pfCand_branches)))
                X_cells_ele = np.empty((b_size, n_cells_eta, n_cells_phi, len(ele_branches)))
                X_cells_muon = np.empty((b_size, n_cells_eta, n_cells_phi, len(muon_branches)))

                for n in range(len(pfCand_branches)):
                    X_cells_pfCand[:, :, :, n] = \
                        df_cells[pfCand_branches[n].encode()].clip(min_value, max_value) \
                        .reshape(b_size, n_cells_eta, n_cells_phi)

                for n in range(len(ele_branches)):
                    X_cells_ele[:, :, :, n] = \
                        df_cells[ele_branches[n].encode()].clip(min_value, max_value) \
                        .reshape(b_size, n_cells_eta, n_cells_phi)

                for n in range(len(muon_branches)):
                    X_cells_muon[:, :, :, n] = \
                        df_cells[muon_branches[n].encode()].clip(min_value, max_value) \
                        .reshape(b_size, n_cells_eta, n_cells_phi)
                X_all = [ X_taus, X_cells_pfCand, X_cells_ele, X_cells_muon ]
                if return_truth and return_weights:
                    yield (X_all, Y, weights)
                elif return_truth:
                    yield (X_all, Y)
                elif return_weights:
                    yield (X_all, weights)
                else:
                    yield X_all
            current_pass += 1
