import glob
import math
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
import numpy as np
import uproot
from common import *
from fill_grid import FillGrid

def LoaderThread(file_name, queue, executor, batch_size, chunk_size, tau_begin, tau_end, return_truth, return_weights):
    root_file = uproot.open(file_name)
    taus_tree = root_file['taus']
    cells_tree = {}
    for loc in cell_locations:
        cells_tree[loc] = root_file[loc + '_cells']
    component_branches = [ input_cell_common_branches + input_cell_pfCand_branches,
                           input_cell_common_branches + input_cell_ele_branches,
                           input_cell_common_branches + input_cell_muon_branches ]

    tau_current = tau_begin
    while tau_current < tau_end:
        entry_stop = min(tau_current + chunk_size, tau_end)
        df_taus = taus_tree.arrays(branches=df_tau_branches, outputtype=pandas.DataFrame, namedecode='utf-8',
                                  entrystart=tau_current, entrystop=entry_stop, executor=executor)

        df_cells = {}
        cells_begin_ref = {}
        for loc in cell_locations:
            cells_begin = df_taus[loc + 'Cells_begin'].values[0]
            cells_end = df_taus[loc + 'Cells_end'].values[-1]
            df_cells[loc] = cells_tree[loc].arrays(branches=df_cell_branches, outputtype=pandas.DataFrame,
                                                   namedecode='utf-8', entrystart=cells_begin,
                                                   entrystop=cells_end, executor=executor)
            cells_begin_ref[loc] = cells_begin
        n_batches = math.ceil(chunk_size / float(batch_size))
        for batch_id in range(n_batches):
            b_tau_begin = batch_id * batch_size
            b_tau_end = min(df_taus.shape[0], (batch_id + 1) * batch_size)
            b_size = b_tau_end - b_tau_begin

            X_taus = np.empty((b_size, len(input_tau_branches)))
            Y = np.empty((b_size, n_outputs), dtype=int)
            if return_weights:
                weights = np.empty(b_size)

            X_taus[:, :] = df_taus[input_tau_branches].values[b_tau_begin:b_tau_end, :]
            Y[:, :] = df_taus[truth_branches].values[b_tau_begin:b_tau_end, :]

            if return_weights:
                weights[:] = df_taus[weight_branches[0]].values[b_tau_begin:b_tau_end]

            X_all = [ X_taus ]
            for loc in cell_locations:
                b_cells_begins = df_taus[loc + 'Cells_begin'].values[b_tau_begin:b_tau_end]
                b_cells_ends = df_taus[loc + 'Cells_end'].values[b_tau_begin:b_tau_end]
                b_cells_begin = b_cells_begins[0] - cells_begin_ref[loc]
                b_cells_end = b_cells_ends[-1] - cells_begin_ref[loc]
                for cmp_branches in component_branches:
                    X_cells_comp = np.zeros((b_size, n_cells_eta[loc], n_cells_phi[loc], len(cmp_branches)))
                    FillGrid(b_cells_begins, b_cells_ends, (n_cells_eta[loc] - 1) / 2, (n_cells_phi[loc] - 1) / 2,
                             df_cells[loc][cmp_branches].values[b_cells_begin:b_cells_end, :], X_cells_comp)
                    X_all.append(X_cells_comp)

            if return_truth and return_weights:
                item = (X_all, Y, weights)
            elif return_truth:
                item = (X_all, Y)
            elif return_weights:
                item = (X_all, weights)
            else:
                item = X_all
            queue.put(item)

class DataLoader:
    @staticmethod
    def GetNumberOfEntries(file_name, tree_name):
        with uproot.open(file_name) as file:
            tree = file[tree_name]
            return tree.numentries

    def __init__(self, file_name, batch_size, chunk_size, validation_size = None, n_passes = -1):
        if batch_size <= 0 or chunk_size <= 0:
            raise RuntimeError("batch_size and chunk_size should be positive numbers")
        self.file_name = file_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.n_passes = n_passes
        self.file_size = DataLoader.GetNumberOfEntries(self.file_name, 'taus')
        self.has_validation_set = validation_size is not None

        if self.has_validation_set:
            if validation_size <= 0 or validation_size >= self.file_size:
                raise RuntimeError("invalid validation_size")
            self.validation_size = validation_size
            self.first_tau_index = validation_size
            self.data_size = self.file_size - self.validation_size
        else:
            self.first_tau_index = 0
            self.data_size = self.file_size

        self.steps_per_epoch = math.ceil(self.data_size / self.batch_size)
        if self.has_validation_set:
            self.validation_steps = math.ceil(self.validation_size / self.batch_size)
        self.executor = ThreadPoolExecutor(8)

    def generator(self, primary_set = True, return_truth = True, return_weights = True):
        if primary_set:
            tau_begin = self.first_tau_index
            tau_end = self.file_size
            steps_per_epoch = self.steps_per_epoch
        else:
            tau_begin = 0
            tau_end = self.validation_size
            steps_per_epoch = self.validation_steps

        queue = Queue(maxsize=10)
        current_pass = 0
        while self.n_passes < 0 or current_pass < self.n_passes:
            thread = Thread(target=LoaderThread,
                     args=( self.file_name, queue, self.executor, self.batch_size, self.chunk_size, tau_begin, tau_end,
                            return_truth, return_weights ))
            thread.daemon = True
            thread.start()
            for step_id in range(steps_per_epoch):
                item = queue.get()
                yield item

            current_pass += 1
