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


    data_source = DataSource(queue_files)
    put_next = True

    while put_next:

        data = data_source.get()
        if data is None:
            break

        X_all = tuple(GetData.getsequence(data.x, batch_size, input_grids, n_sequence, n_features))
        if return_truth:
            Y = GetData.getdata(data.y, (batch_size, output_classes))
            item = (X_all, Y)
        else:
            item = X_all
        
        put_next = queue_out.put(item)

    queue_out.put_terminate(identifier)
    terminators[identifier].wait()

class DataLoader (DataLoaderBase):

    def __init__(self, config, file_scaling):

        dataloader_core = config["Setup"]["dataloader_core"]
        self.compile_classes(config, file_scaling, dataloader_core)

        self.config = config

        # Computing additional variables: 
        self.config['input_map'] = {}
        self.config['n_features'] = {}
        self.config['embedded_param'] = {}

        for pfCand_type in self.config["CellObjectType"]:
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

        data_files = glob.glob(f'{self.config["Setup"]["input_dir"]}/*.root')

        self.train_files, self.val_files = \
             np.split(data_files, [int(len(data_files)*(1-self.config["SetupNN"]["validation_split"]))])

        print("Files for training:", len(self.train_files))
        print("Files for validation:", len(self.val_files))


    def get_generator(self, primary_set = True, return_truth = True, return_weights = False):

        _files = self.train_files if primary_set else self.val_files
        if len(_files)==0:
            raise RuntimeError(("Taining" if primary_set else "Validation")+\
                               " file list is empty.")

        n_batches = self.config["SetupNN"]["n_batches"] if primary_set \
                    else self.config["SetupNN"]["n_batches_val"]
        print("Number of workers in DataLoader: ",
                self.config["SetupNN"]["n_load_workers"])

        converter = torch_to_tf(return_truth, return_weights)

        def _generator():

            finish_counter = 0
            
            queue_files = mp.Queue()
            [ queue_files.put(file) for file in _files ]

            queue_out = QueueEx(max_size = self.config["SetupNN"]["max_queue_size"], max_n_puts = n_batches)

            processes = []
            n_load_workers = self.config["SetupNN"]["n_load_workers"]
            terminators = [ mp.Event() for _ in range(n_load_workers) ]
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

            while finish_counter < n_load_workers:
                item = queue_out.get()
                if isinstance(item, int):
                    finish_counter+=1
                    terminators[item].set()
                else:
                    yield converter(item)
                    
            ugly_clean(queue_files)
            queue_out.clear()

            for i, pr in enumerate(processes):
                pr.join()
            gc.collect()

        return _generator


    def get_shape(self):

        input_shape, input_types = [], []
        for comp in self.config["CellObjectType"]:
            input_shape.append((self.config["Setup"]["n_tau"],
                                self.config["SequenceLength"][comp],
                                self.config['n_features'][comp]))
            input_types.append(tf.float32)
        input_shape = [tuple(input_shape)]
        input_shape.append((self.config["Setup"]["n_tau"],
                            self.config["Setup"]["output_classes"]))
        input_types = [tuple(input_types)]
        input_types.append(tf.float32)

        return tuple(input_shape), tuple(input_types)
