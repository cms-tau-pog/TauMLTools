class MixBin:
  @staticmethod
  def Load(cfg):
    mix_bins = []
    pt_bin_edges = cfg['bin_edges']['pt']
    eta_bin_edges = cfg['bin_edges']['eta']
    batch_size = 0
    for step_idx, output_step in enumerate(cfg['output']):
      step_bins = []
      for bin_idx, bin in enumerate(output_step['bins']):
        if len(bin['counts']) != len(pt_bin_edges) - 1:
          raise ValueError("Number of counts does not match number of pt bins")
        for pt_bin_idx, count in enumerate(bin['counts']):
          if count == 0: continue
          mix_bin = MixBin()
          mix_bin.input_setups = output_step['input_setups']
          mix_bin.inputs = []
          for input_setup in output_step['input_setups']:
            mix_bin.inputs.extend(cfg['inputs'][input_setup])
          selection = cfg['bin_selection'].format(pt_low=pt_bin_edges[pt_bin_idx],
                                                  pt_high=pt_bin_edges[pt_bin_idx + 1],
                                                  eta_low=eta_bin_edges[bin['eta_bin']],
                                                  eta_high=eta_bin_edges[bin['eta_bin'] + 1])
          mix_bin.selection = f'L1Tau_Gen_type == static_cast<int>(TauType::{bin["tau_type"]}) && ({selection})'
          mix_bin.tau_type = bin['tau_type']
          mix_bin.eta_bin = bin['eta_bin']
          mix_bin.pt_bin = pt_bin_idx
          mix_bin.step_idx = step_idx
          mix_bin.bin_idx = bin_idx
          mix_bin.start_idx = batch_size
          mix_bin.stop_idx = batch_size + count
          mix_bin.count = count
          mix_bin.allow_duplicates = bin.get('allow_duplicates', False)
          step_bins.append(mix_bin)
          batch_size += count
      mix_bins.append(step_bins)
    return mix_bins, batch_size

  def Print(self):
    print(f'taus/batch: {self.count}')
    print(f'inputs: {self.input_setups}')
    print(f'eta bin: {self.eta_bin}')
    print(f'pt bin: {self.pt_bin}')
    print(f'step idx: {self.bin_idx}')
    print(f'bin idx: {self.bin_idx}')
    print(f'tau_type: {self.tau_type}')
    print(f'selection: {self.selection}')
    print(f'allow duplicates: {self.allow_duplicates}')

