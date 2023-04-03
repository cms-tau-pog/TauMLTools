import gc
import glob
from DataLoaderBase import *

def LoaderThread(queue_out,
                 queue_files,
                 terminators,
                 identifier,
                 batch_size,
                 input_grids,
                 n_sequence,
                 n_features,
                 output_classes,
                 return_truth,
                 return_weights):

    def DataProcess(data):

        X_all = GetData.getsequence(data.x, data.tau_i, batch_size, input_grids, n_sequence, n_features)

        if return_weights:
            weights = GetData.getdata(data.weights, data.tau_i, -1, debug_area="weights")
        if return_truth:
            Y = GetData.getdata(data.y, data.tau_i, (batch_size, output_classes), debug_area="truth")

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

        self.config = config
        
        dataloader_core = config["Setup"]["dataloader_core"]
        data_files = glob.glob(f'{config["Setup"]["input_dir"]}/*.root')
        self.compile_classes(config, file_scaling, dataloader_core, data_files)

        # Computing additional variables: 
        self.config['input_map'] = {}
        self.config['n_features'] = {}
        self.config['embedded_param'] = {}

        for pfCand_type in self.config["Features_all"]:
            self.config['input_map'][pfCand_type] = {}
            self.config['n_features'][pfCand_type] = \
                len(self.config["Features_all"][pfCand_type]) - \
                len(self.config["Features_disable"][pfCand_type])
            if pfCand_type in self.config["EmbeddedCellObjectType"]:
                self.config['embedded_param'][pfCand_type] = {}
            for f_dict in self.config["Features_all"][pfCand_type]:
                f = next(iter(f_dict))
                if f not in self.config["Features_disable"][pfCand_type]:
                    self.config['input_map'][pfCand_type][f] = \
                        getattr(getattr(R,pfCand_type+"_Features"),f)
                if pfCand_type in self.config["EmbeddedCellObjectType"]:
                    self.config['embedded_param'][pfCand_type][f] = f_dict[f][-2:]

        self.train_files, self.val_files = \
            np.split(data_files, [int(len(data_files)*(1-self.config["SetupBaseNN"]["validation_split"]))])

        print("Files for training:", len(self.train_files))
        print("Files for validation:", len(self.val_files))

    def get_predict_generator(self, return_truth=True, return_weights=False):
        '''
        The implementation of the deterministic generator
        for suitable use of performance evaluation.
        The use example:
        >gen_file = dataloader.get_predict_generator()
        >for file in files:
        >   for x,y in en_file(file):
        >       y_pred = ...
        '''
        print(self.config['n_features'])
        converter = torch_to_tf(return_truth, return_weights)
        data_loader = R.DataLoader()
        def read_from_file(file_path):
            data_loader.ReadFile(R.std.string(file_path), 0, -1)
            while True:
                full_tensor = data_loader.MoveNext()
                data = data_loader.LoadData(full_tensor)
                x = GetData.getsequence(data.x, data.tau_i,
                                            self.config["Setup"]["n_tau"],
                                            self.config["CellObjectType"],
                                            self.config["SequenceLength"],
                                            self.config['n_features'])

                y = GetData.getdata(data.y, data.tau_i,
                                    (self.config["Setup"]["n_tau"],
                                    self.config["Setup"]["output_classes"]),
                                    debug_area="truth")
                uncompress_index = np.copy(np.frombuffer(data.uncompress_index.data(),
                                                         dtype=np.int,
                                                         count=data.uncompress_index.size()))
                if(self.config["Setup"]["to_propagate_glob"]==True):
                    # Needed to propagate global variables from DataLoader to the apply_training
                    x_glob = GetData.getdata(data.x_glob, data.tau_i,
                                    (self.config["Setup"]["n_tau"],
                                     self.config['n_features']["Global"]),
                                    debug_area="global")
                    yield converter((tuple(x), y)), x_glob.clone().numpy(), uncompress_index[:data.tau_i], data.uncompress_size
                else:
                    yield converter((tuple(x), y)), uncompress_index[:data.tau_i], data.uncompress_size
                if full_tensor==False: break
        return read_from_file

    def get_generator(self, primary_set = True, return_truth = True, return_weights = False):

        _files = self.train_files if primary_set else self.val_files
        if len(_files)==0:
            raise RuntimeError(("Taining" if primary_set else "Validation")+\
                               " file list is empty.")

        n_batches = self.config["SetupBaseNN"]["n_batches"] if primary_set \
                    else self.config["SetupBaseNN"]["n_batches_val"]
        print("Number of workers in DataLoader: ",
                self.config["SetupBaseNN"]["n_load_workers"])

        converter = torch_to_tf(return_truth, return_weights)

        def _generator():

            finish_counter = 0
            
            queue_files = mp.Queue()
            [ queue_files.put(file) for file in _files ]

            queue_out = QueueEx(max_size = self.config["SetupBaseNN"]["max_queue_size"], max_n_puts = n_batches)

            processes = []
            n_load_workers = self.config["SetupBaseNN"]["n_load_workers"]
            terminators = [ [mp.Event(),mp.Event()] for _ in range(n_load_workers) ]
            for i in range(n_load_workers):
                processes.append(
                mp.Process(target = LoaderThread, 
                           args = (queue_out, queue_files,
                                   terminators, i,
                                   self.config["Setup"]["n_tau"],
                                   self.config["CellObjectType"],
                                   self.config["SequenceLength"],
                                   self.config['n_features'],
                                   self.config["Setup"]["output_classes"],
                                   return_truth,
                                   return_weights)))
                processes[-1].start()
            
            # First part to iterate through the main part
            finish_counter = 0
            while finish_counter < n_load_workers:
                item = queue_out.get()
                if isinstance(item, int):
                    finish_counter+=1
                else:
                    yield converter(item)

            for i in range(n_load_workers):
                terminators[i][0].set()

            # Second part to collect remains
            collector = Collector(self.config["Setup"]["n_tau"])
            finish_counter = 0
            while finish_counter < n_load_workers:
                item = queue_out.get()
                if isinstance(item, int):
                    finish_counter+=1
                else:
                    collector.fill(item)
            remains = collector.get()
            if remains is not None:
                for item in remains:
                    yield item
            for i in range(n_load_workers):
                terminators[i][1].set()
  
            queue_out.clear()
            ugly_clean(queue_files)

            for i, pr in enumerate(processes):
                pr.join()
            gc.collect()

        return _generator


    def get_shape(self):

        input_shape, input_types = [], []
        for comp in self.config["CellObjectType"]:
            input_shape.append((None,
                                self.config["SequenceLength"][comp],
                                self.config['n_features'][comp]))
            input_types.append(tf.float32)
        input_shape = [tuple(input_shape)]
        input_shape.append((None,
                            self.config["Setup"]["output_classes"]))
        input_types = [tuple(input_types)]
        input_types.append(tf.float32)

        return tuple(input_shape), tuple(input_types)
