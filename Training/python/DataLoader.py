import glob
import math
import gc
from queue import Queue
from threading import Thread, Lock
import numpy as np
import pandas
import uproot
from common import *
from fill_grid import FillGrid, FillSequence

read_hdf_lock = Lock()

def read_hdf(file_name, key, columns, start, stop):
    read_hdf_lock.acquire()
    df = pandas.read_hdf(file_name, key, columns=columns, start=start, stop=stop)
    read_hdf_lock.release()
    return df

def LoaderThread(file_entries, queue, net_config, batch_size, chunk_size, return_truth, return_weights, return_grid):
    FillFn = FillGrid if return_grid else FillSequence
    for file_name, tau_begin, tau_end in file_entries:
        root_input = file_name.endswith('.root')
        if root_input:
            root_file = uproot.open(file_name)
            taus_tree = root_file['taus']
            cells_tree = {}

            for loc in net_conf.cell_locations:
                cells_tree[loc] = root_file[loc + '_cells']

        expected_n_batches = int(math.ceil((tau_end - tau_begin) / float(batch_size)))
        tau_current = tau_begin
        global_batch_id = 0
        while tau_current < tau_end:
            entry_stop = min(tau_current + chunk_size, tau_end)
            if root_input:
                df_taus = taus_tree.arrays(branches=df_tau_branches, outputtype=pandas.DataFrame, namedecode='utf-8',
                                          entrystart=tau_current, entrystop=entry_stop)
            else:
                df_taus = read_hdf(file_name, 'taus', df_tau_branches, tau_current, entry_stop)

            df_cells = {}
            cells_begin_ref = {}
            for loc in net_config.cell_locations:
                cells_begin = df_taus[loc + 'Cells_begin'].values[0]
                cells_end = df_taus[loc + 'Cells_end'].values[-1]
                if root_input:
                    df_cells[loc] = cells_tree[loc].arrays(branches=df_cell_branches, outputtype=pandas.DataFrame,
                                                           namedecode='utf-8', entrystart=cells_begin,
                                                           entrystop=cells_end, executor=executor)
                else:
                    df_cells[loc] = read_hdf(file_name, loc + '_cells', df_cell_branches, cells_begin, cells_end)
                cells_begin_ref[loc] = cells_begin
            current_chunk_size = entry_stop - tau_current
            n_batches = int(math.ceil(current_chunk_size / float(batch_size)))
            for batch_id in range(n_batches):
                if global_batch_id >= expected_n_batches:
                    raise RuntimeError("Too many batches")
                global_batch_id += 1
                b_tau_begin = batch_id * batch_size
                b_tau_end = min(df_taus.shape[0], (batch_id + 1) * batch_size)
                b_size = b_tau_end - b_tau_begin

                X_all = [ ]
                if len(net_config.tau_branches):
                    X_taus = np.empty((b_size, len(net_config.tau_branches)), dtype=np.float32)
                    X_taus[:, :] = df_taus[net_config.tau_branches].values[b_tau_begin:b_tau_end, :]
                    X_all.append(X_taus)

                Y = np.empty((b_size, n_outputs), dtype=np.int)
                Y[:, :] = df_taus[truth_branches].values[b_tau_begin:b_tau_end, :]

                if return_weights:
                    weights = np.empty(b_size, dtype=np.float32)
                    weights[:] = df_taus[weight_branches[0]].values[b_tau_begin:b_tau_end]

                for loc in net_config.cell_locations:
                    b_cells_begins = df_taus[loc + 'Cells_begin'].values[b_tau_begin:b_tau_end]
                    b_cells_ends = df_taus[loc + 'Cells_end'].values[b_tau_begin:b_tau_end]
                    b_cells_begin = b_cells_begins[0] - cells_begin_ref[loc]
                    b_cells_end = b_cells_ends[-1] - cells_begin_ref[loc]
                    for cmp_branches in net_config.comp_branches:
                        X_cells_comp = FillFn(b_cells_begins, b_cells_ends, n_cells_eta[loc], n_cells_phi[loc],
                            df_cells[loc][cell_index_branches].values[b_cells_begin:b_cells_end, :],
                            df_cells[loc][cmp_branches].values[b_cells_begin:b_cells_end, :],
                            df_taus[input_cell_external_branches].values[b_tau_begin:b_tau_end, :])
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
            tau_current = entry_stop
            del df_taus
            for loc in net_config.cell_locations:
                del df_cells[loc]
            gc.collect()

class FileEntry:
    @staticmethod
    def GetNumberOfSteps(data_size, batch_size):
        return int(math.ceil(data_size / float(batch_size)))

    def __init__(self, file_name, n_entries, batch_size, evt_left, val_evt_left):
        self.file_name = file_name
        self.tau_begin = None
        self.tau_end = None
        self.val_tau_begin = None
        self.val_tau_end = None
        self.size = 0
        self.val_size = 0
        self.steps = 0
        self.val_steps = 0
        if n_entries is None or n_entries <= 0: return
        if val_evt_left is not None and val_evt_left > 0:
            self.val_tau_begin = 0
            self.val_tau_end = min(n_entries, val_evt_left)
            self.val_size = self.val_tau_end - self.val_tau_begin
            self.val_steps = FileEntry.GetNumberOfSteps(self.val_size, batch_size)
        if (evt_left is None or evt_left > 0) and (self.val_tau_end is None or self.val_tau_end < n_entries):
            self.tau_begin = self.val_tau_end if self.val_tau_end is not None else 0
            self.tau_end = min(n_entries, evt_left) if evt_left is not None else n_entries
            self.size = self.tau_end - self.tau_begin
            self.steps = FileEntry.GetNumberOfSteps(self.size, batch_size)

    def __repr__(self):
        return str(self)
    def __str__(self):
        return '{}: size = {}, val_size = {}'.format(self.file_name, self.size, self.val_size)

class DataLoader:
    @staticmethod
    def GetNumberOfEntries(file_name, tree_name):
        if file_name.endswith('.root'):
            with uproot.open(file_name) as file:
                tree = file[tree_name]
                return tree.numentries
        else:
            with pandas.HDFStore(file_name, mode='r') as h5:
                return h5.get_storer('taus').nrows

    def __init__(self, file_name_pattern, net_config, batch_size, chunk_size, validation_size = None,
                 max_data_size = None, max_queue_size = 8, n_passes = -1, return_grid = True):
        if type(batch_size) != int or type(chunk_size) != int or batch_size <= 0 or chunk_size <= 0:
            raise RuntimeError("batch_size and chunk_size should be positive integer numbers")
        if batch_size > chunk_size or chunk_size % batch_size != 0:
            raise RuntimeError("chunk_size should be greater or equal to batch_size, and be proportional to it")


        self.net_config = net_config
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.n_passes = n_passes
        self.max_queue_size = max_queue_size
        self.return_grid = return_grid
        self.has_validation_set = validation_size is not None

        all_files = [ f.replace('\\', '/') for f in sorted(glob.glob(file_name_pattern)) ]
        evt_left = max_data_size
        val_evt_left = validation_size
        self.file_entries = []
        self.data_size = 0
        self.validation_size = 0
        self.steps_per_epoch = 0
        self.validation_steps = 0
        for file_name in all_files:
            file_size = DataLoader.GetNumberOfEntries(file_name, 'taus')
            file_entry = FileEntry(file_name, file_size, batch_size, evt_left, val_evt_left)
            if evt_left is not None:
                evt_left -= file_entry.size
            if val_evt_left is not None:
                val_evt_left -= file_entry.val_size
            self.data_size += file_entry.size
            self.validation_size += file_entry.val_size
            self.steps_per_epoch += file_entry.steps
            self.validation_steps += file_entry.val_steps
            self.file_entries.append(file_entry)
        self.total_size = self.data_size + self.validation_size

        if val_evt_left is not None and val_evt_left != 0:
            raise RuntimeError("Insufficient number of events to create validation set.")
        if self.data_size == 0:
            raise RuntimeError("Insufficent number of events to create data set.")

    def generator(self, primary_set = True, return_truth = True, return_weights = True):
        file_entries = []
        if primary_set:
            for entry in self.file_entries:
                if entry.size > 0:
                    file_entries.append( (entry.file_name, entry.tau_begin, entry.tau_end) )
            steps_per_epoch = self.steps_per_epoch
        else:
            for entry in self.file_entries:
                if entry.val_size > 0:
                    file_entries.append( (entry.file_name, entry.val_tau_begin, entry.val_tau_end) )
            steps_per_epoch = self.validation_steps

        queue = Queue(maxsize=self.max_queue_size)
        current_pass = 0
        while self.n_passes < 0 or current_pass < self.n_passes:
            thread = Thread(target=LoaderThread,
                     args=( file_entries, queue, self.net_config, self.batch_size, self.chunk_size,
                            return_truth, return_weights, self.return_grid ))
            thread.daemon = True
            thread.start()
            for step_id in range(steps_per_epoch):
                item = queue.get()
                yield item
            thread.join()
            gc.collect()
            current_pass += 1
