import gc
import glob
from tqdm import tqdm

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
                 return_weights,
                 active_features,
                 cell_locations):

    def DataProcess(data):

        X_all = GetData.getX(data, data.tau_i, batch_size, n_grid_features, n_flat_features,
                             input_grids, n_inner_cells, n_outer_cells, active_features, cell_locations)
        if return_weights:
            weights = GetData.getdata(data.weight, data.tau_i, -1, debug_area="weights")
        if return_truth:
            Y = GetData.getdata(data.y_onehot, data.tau_i, (batch_size, tau_types), debug_area="truth")

        if return_truth and return_weights:
            item = [X_all, Y, weights]
        elif return_truth:
            item = [X_all, Y]
        elif return_weights:
            item = [X_all, weights]
        else:
            item = X_all

        return item

    data_source = DataSource(queue_files)
    put_next = True

    while put_next:

        data = data_source.get()
        if data is None: break
        item = DataProcess(data)
        put_next = queue_out.put(item)

    queue_out.put_terminate(identifier)
    terminators[identifier][0].wait()

    if (data := data_source.get_remains()) is not None:
        item = DataProcess(data)
        _ = queue_out.put(item)

    queue_out.put_terminate(identifier)
    terminators[identifier][1].wait()

class DataLoader (DataLoaderBase):

    def __init__(self, config, file_scaling):

        self.dataloader_core = config["Setup"]["dataloader_core"]

        self.config = config

        for block_name, features in self.config["Features_all"].items():
            if block_name not in self.config["SetupNN"]["active_features"]:
                self.config["Features_disable"][block_name] = [list(x.keys())[0] for x in self.config["Features_all"][block_name]]

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
        self.inner_cell_size = self.config["Setup"]["inner_cell_size"]
        self.outer_cell_size = self.config["Setup"]["outer_cell_size"]
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
        self.use_weights = self.config["Setup"]["use_weights"]
        self.DeepTauVSjet_cut  = self.config["Setup"]["DeepTauVSjet_cut"]
        self.cell_locations = self.config["SetupNN"]["cell_locations"]
        self.rm_inner_from_outer = self.config["Setup"]["rm_inner_from_outer"]
        self.active_features = self.config["SetupNN"]["active_features"]
        self.input_type = self.config["Setup"]["input_type"]
        self.tf_input_dir = self.config["Setup"]["tf_input_dir"]
        self.tf_dataset_x_order = self.config["Setup"]["tf_dataset_x_order"]
        self.adversarial_dataset = self.config["Setup"]["adversarial_dataset"]
        self.adversarial_parameter = self.config["Setup"]["adv_parameter"]
        self.adv_batch_size = self.config["Setup"]["n_adv_tau"]
        self.adv_learning_rate = self.config["Setup"]["adv_learning_rate"]
        self.use_previous_opt = self.config["Setup"]["use_previous_opt"]
        
        if self.input_type == "ROOT" or self.input_type == "Adversarial":
            data_files = glob.glob(f'{self.config["Setup"]["input_dir"]}/*.root') 
            self.train_files, self.val_files = \
                np.split(data_files, [int(len(data_files)*(1-self.validation_split))])
            print("Files for training:", len(self.train_files))
            print("Files for validation:", len(self.val_files))
            

        self.compile_classes(config, file_scaling, self.dataloader_core, data_files)



    def get_generator(self, primary_set = True, return_truth = True, return_weights = True, show_progress = False, adversarial = False):

        _files = self.train_files if primary_set else self.val_files
        if len(_files)==0:
            raise RuntimeError(("Training" if primary_set else "Validation")+\
                               " file list is empty.")

        n_batches = self.n_batches if primary_set else self.n_batches_val
        print("Number of workers in DataLoader: ", self.n_load_workers)
        converter = torch_to_tf(return_truth, return_weights)

        def _generator():
            
            if show_progress and n_batches>0:
                pbar = tqdm(total = n_batches)

            queue_files = mp.Queue()
            [ queue_files.put(file) for file in _files ]

            queue_out = QueueEx(max_size = self.max_queue_size, max_n_puts = n_batches)
            terminators = [ [mp.Event(),mp.Event()] for _ in range(self.n_load_workers) ]

            processes = []
            for i in range(self.n_load_workers):
                processes.append(
                mp.Process(target = LoaderThread,
                        args = (queue_out, queue_files, terminators, i,
                                self.input_grids, self.batch_size, self.n_inner_cells,
                                self.n_outer_cells, self.n_flat_features, self.n_grid_features,
                                self.tau_types, return_truth, return_weights, self.active_features, self.cell_locations)))
                processes[-1].start()

            if adversarial:
                if primary_set:
                    adv_ds = tf.data.experimental.load(self.adversarial_dataset, compression="GZIP").take(750)
                else:
                    adv_ds = tf.data.experimental.load(self.adversarial_dataset, compression="GZIP").skip(750).take(375)
                adv_iter = iter(adv_ds)


            # First part to iterate through the main part
            finish_counter = 0
            while finish_counter < self.n_load_workers:
                item = queue_out.get()
                if isinstance(item, int):
                    finish_counter+=1
                else:
                    if show_progress and n_batches>0:
                        pbar.update(1)
                    if adversarial:
                        x, y, sample_weight = converter(item)
                        w_zero = tf.zeros(self.batch_size)
                        y_zero = tf.zeros((self.batch_size, 4))
                        try: # adv iterator not exhausted
                            x_adv, y_adv, sample_weight_adv = next(adv_iter)
                        except: #reset iterator
                            adv_iter = iter(adv_ds)
                            x_adv, y_adv, sample_weight_adv = next(adv_iter)
                        x_out = tuple(tf.concat([x[i], x_adv[i]], 0) for i in range(len(x)))
                        y_out = tf.concat([y, y_zero],0)
                        y_adv_out = tf.expand_dims(tf.concat([w_zero, y_adv[:,0]],0), axis=1) 
                        w_out = tf.concat([sample_weight, w_zero],0)
                        w_adv_out = tf.expand_dims(tf.concat([w_zero, sample_weight_adv],0), axis=1) 
                        yield (x_out, y_out, y_adv_out, w_out, w_adv_out)
                    else:
                        yield converter(item)
            for i in range(self.n_load_workers):
                terminators[i][0].set()

            # Second part to collect remains
            collector = Collector(self.batch_size)
            finish_counter = 0
            while finish_counter < self.n_load_workers:
                item = queue_out.get()
                if isinstance(item, int):
                    finish_counter+=1
                else:
                    collector.fill(item)
            remains = collector.get()
            if remains is not None:
                for item in remains:
                    yield item
            for i in range(self.n_load_workers):
                terminators[i][1].set()

            queue_out.clear()
            ugly_clean(queue_files)

            for i, pr in enumerate(processes):
                pr.join()
            gc.collect()

        return _generator

    def get_predict_generator(self, return_truth=True, return_weights=False):
        '''
        The implementation of the deterministic generator
        for suitable use of performance evaluation.
        The use example:
        >gen_file = dataloader.get_eval_generator()
        >for file in files:
        >   for x,y in en_file(file):
        >       y_pred = ...
        '''
        converter = torch_to_tf(return_truth, return_weights)
        data_loader = R.DataLoader()
        def read_from_file(file_path):
            data_loader.ReadFile(R.std.string(file_path), 0, -1)
            while True:
                full_tensor = data_loader.MoveNext()
                data = data_loader.LoadData(full_tensor)
                x = GetData.getX(data, data.tau_i, self.batch_size, self.n_grid_features,
                                 self.n_flat_features, self.input_grids,
                                 self.n_inner_cells, self.n_outer_cells,
                                 self.active_features, self.cell_locations)
                y = GetData.getdata(data.y_onehot, data.tau_i, (self.batch_size, self.tau_types))
                uncompress_index = np.copy(np.frombuffer(data.uncompress_index.data(),
                                                         dtype=np.int,
                                                         count=data.uncompress_index.size()))
                yield converter((tuple(x), y)), uncompress_index[:data.tau_i], data.uncompress_size
                if full_tensor==False: break
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
        netConf.cell_locations = self.config["SetupNN"]["cell_locations"]
        netConf.comp_names = ['egamma', 'muon', 'hadrons']
        netConf.n_comp_branches = [
            len(input_cell_external_branches + input_cell_pfCand_ele_branches + input_cell_ele_branches + input_cell_pfCand_gamma_branches),
            len(input_cell_external_branches + input_cell_pfCand_muon_branches + input_cell_muon_branches),
            len(input_cell_external_branches + input_cell_pfCand_chHad_branches + input_cell_pfCand_nHad_branches)
        ]
        netConf.n_cells = self.n_cells
        netConf.n_outputs = self.tau_types
        netConf.first_layer_reg = self.config["SetupNN"]["first_layer_reg"]
        return netConf

    def get_input_config(self, return_truth = True, return_weights = True):
        # Input tensor shape and type
        input_shape, input_types = [], []
        if 'TauFlat' in self.active_features:
            input_shape.append(tuple([None, len(self.get_branches(self.config,"TauFlat"))]))
            input_types.append(tf.float32)
        for grid in self.cell_locations:
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
