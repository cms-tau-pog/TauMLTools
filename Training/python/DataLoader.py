import gc
import glob

from DataLoaderBase import *

def LoaderThread(queue_out,
                 queue_files,
                 terminators,
                 identifier,
                 input_grids,
                 batch_size,
                 n_inner_cells,
                 n_outer_cells,
                 n_flat_features,
                 n_grid_features,
                 tau_types,
                 return_truth,
                 return_weights):

    data_source = DataSource(queue_files)
    put_next = True

    while put_next:

        data = data_source.get()
        if data is None:
            break

        X_all = GetData.getX(data, batch_size, n_grid_features, n_flat_features,
                             input_grids, n_inner_cells, n_outer_cells)

        X_all = tuple(X_all)

        if return_weights:
            weights = GetData.getdata(data.weight, -1)
        if return_truth:
            Y = GetData.getdata(data.y_onehot, (batch_size, tau_types))

        if return_truth and return_weights:
            item = (X_all, Y, weights)
        elif return_truth:
            item = (X_all, Y)
        elif return_weights:
            item = (X_all, weights)
        else:
            item = X_all

        put_next = queue_out.put(item)

    queue_out.put_terminate(identifier)
    terminators[identifier].wait()

class DataLoader (DataLoaderBase):

    def __init__(self, config, file_scaling):

        self.dataloader_core = config["Setup"]["dataloader_core"]
        self.compile_classes(config, file_scaling, self.dataloader_core)

        self.config = config

        self.n_grid_features = {}
        for celltype in self.config["Features_all"]:
            if celltype!="TauFlat":
                self.n_grid_features[str(celltype)] = len(self.config["Features_all"][celltype]) - \
                                                      len(self.config["Features_disable"][celltype])

        self.n_flat_features = len(self.config["Features_all"]["TauFlat"]) - \
                               len(self.config["Features_disable"]["TauFlat"])

        # global variables after compile are read out here
        self.batch_size     = self.config["Setup"]["n_tau"]
        self.n_inner_cells  = self.config["Setup"]["n_inner_cells"]
        self.n_outer_cells  = self.config["Setup"]["n_outer_cells"]
        self.tau_types      = len(self.config["Setup"]["tau_types_names"])
        self.n_load_workers   = self.config["SetupNN"]["n_load_workers"]
        self.n_batches        = self.config["SetupNN"]["n_batches"]
        self.n_batches_val    = self.config["SetupNN"]["n_batches_val"]
        self.n_batches_log    = self.config["SetupNN"]["n_batches_log"]
        self.validation_split = self.config["SetupNN"]["validation_split"]
        self.max_queue_size   = self.config["SetupNN"]["max_queue_size"]
        self.n_epochs         = self.config["SetupNN"]["n_epochs"]
        self.epoch         = self.config["SetupNN"]["epoch"]
        self.input_grids        = self.config["SetupNN"]["input_grids"]
        self.n_cells = { 'inner': self.n_inner_cells, 'outer': self.n_outer_cells }
        self.model_name       = self.config["SetupNN"]["model_name"]

        data_files = glob.glob(f'{self.config["Setup"]["input_dir"]}/*.root')
        self.train_files, self.val_files = \
             np.split(data_files, [int(len(data_files)*(1-self.validation_split))])

        print("Files for training:", len(self.train_files))
        print("Files for validation:", len(self.val_files))


    def get_generator(self, primary_set = True, return_truth = True, return_weights = True):

        _files = self.train_files if primary_set else self.val_files
        if len(_files)==0:
            raise RuntimeError(("Taining" if primary_set else "Validation")+\
                               " file list is empty.")

        n_batches = self.n_batches if primary_set else self.n_batches_val
        print("Number of workers in DataLoader: ", self.n_load_workers)
        converter = torch_to_tf(return_truth, return_weights)

        def _generator():

            finish_counter = 0

            queue_files = mp.Queue()
            [ queue_files.put(file) for file in _files ]

            queue_out = QueueEx(max_size = self.max_queue_size, max_n_puts = n_batches)
            terminators = [ mp.Event() for _ in range(self.n_load_workers) ]

            processes = []
            for i in range(self.n_load_workers):
                processes.append(
                mp.Process(target = LoaderThread,
                        args = (queue_out, queue_files, terminators, i,
                                self.input_grids, self.batch_size, self.n_inner_cells,
                                self.n_outer_cells, self.n_flat_features, self.n_grid_features,
                                self.tau_types, return_truth, return_weights,)))
                processes[-1].start()

            while finish_counter < self.n_load_workers:
                item = queue_out.get()
                if isinstance(item, int):
                    finish_counter+=1
                    terminators[item].set()
                else:
                    yield converter(item)

            queue_out.clear()
            ugly_clean(queue_files)

            for i, pr in enumerate(processes):
                pr.join()
            gc.collect()

        return _generator

    def get_predict_generator(self):
        '''
        The implementation of the deterministic generator
        for suitable use of performance evaluation.
        The use example:
        >gen_file = dataloader.get_eval_generator()
        >for file in files:
        >   for x,y in en_file(file):
        >       y_pred = ...
        '''
        assert self.batch_size == 1
        data_loader = R.DataLoader()
        def read_from_file(file_path):
            data_loader.ReadFile(R.std.string(file_path), 0, -1)
            while data_loader.MoveNext():
                data = data_loader.LoadData()
                x = GetData.getX(data, self.batch_size, self.n_grid_features,
                                 self.n_flat_features, self.input_grids,
                                 self.n_inner_cells, self.n_outer_cells)
                y = GetData.getdata(data.y_onehot, (self.batch_size, self.tau_types))
                yield tuple(x), y
        return read_from_file

    @staticmethod
    def get_branches(config, group):
        return list(
            set(sum( [list(d.keys()) for d in config["Features_all"][group]],[])) - \
            set(config["Features_disable"][group])
        )

    def get_net_config(self):

        # copy of feature lists is not necessery
        input_tau_branches = self.get_branches(self.config,"TauFlat")
        input_cell_external_branches = self.get_branches(self.config,"GridGlobal")
        input_cell_pfCand_ele_branches = self.get_branches(self.config,"PfCand_electron")
        input_cell_pfCand_muon_branches = self.get_branches(self.config,"PfCand_muon")
        input_cell_pfCand_chHad_branches = self.get_branches(self.config,"PfCand_chHad")
        input_cell_pfCand_nHad_branches = self.get_branches(self.config,"PfCand_nHad")
        input_cell_pfCand_gamma_branches = self.get_branches(self.config,"PfCand_gamma")
        input_cell_ele_branches = self.get_branches(self.config,"Electron")
        input_cell_muon_branches = self.get_branches(self.config,"Muon")

        class NetConf:
            pass

        netConf = NetConf()
        netConf.tau_net = self.config["SetupNN"]["tau_net"]
        netConf.comp_net = self.config["SetupNN"]["comp_net"]
        netConf.comp_merge_net = self.config["SetupNN"]["comp_merge_net"]
        netConf.conv_2d_net = self.config["SetupNN"]["conv_2d_net"]
        netConf.dense_net = self.config["SetupNN"]["dense_net"]
        netConf.n_tau_branches = len(input_tau_branches)
        netConf.cell_locations = ['inner', 'outer']
        netConf.comp_names = ['egamma', 'muon', 'hadrons']
        netConf.n_comp_branches = [
            len(input_cell_external_branches + input_cell_pfCand_ele_branches + input_cell_ele_branches + input_cell_pfCand_gamma_branches),
            len(input_cell_external_branches + input_cell_pfCand_muon_branches + input_cell_muon_branches),
            len(input_cell_external_branches + input_cell_pfCand_chHad_branches + input_cell_pfCand_nHad_branches)
        ]
        netConf.n_cells = self.n_cells
        netConf.n_outputs = self.tau_types

        return netConf

    def get_input_config(self, return_truth = True, return_weights = True):
        # Input tensor shape and type
        input_shape, input_types = [], []
        input_shape.append(tuple([None, len(self.get_branches(self.config,"TauFlat"))]))
        input_types.append(tf.float32)
        for grid in ["inner","outer"]:
            for f_group in self.input_grids:
                n_f = sum([len(self.get_branches(self.config,cell_type)) for cell_type in f_group])
                input_shape.append(tuple([None, self.n_cells[grid], self.n_cells[grid], n_f]))
                input_types.append(tf.float32)
        input_shape=(tuple(input_shape),)
        input_types=(tuple(input_types),)
        if return_truth:
            input_shape = input_shape + ((None, self.tau_types),)
            input_types = input_types + (tf.float32,)
        if return_weights:
            input_shape = input_shape + ((None),)
            input_types = input_types + (tf.float32,)

        return input_shape, input_types
