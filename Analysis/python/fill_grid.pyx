cimport numpy as np
import numpy as np
import math

cpdef void FillGrid(np.ndarray cells_begin, np.ndarray cells_end, int max_eta_index, int max_phi_index,
                    np.ndarray packed_cells, np.ndarray unpacked_cells):
    begin_ref = cells_begin[0]
    for tau_index in range(cells_begin.shape[0]):
        begin = cells_begin[tau_index] - begin_ref
        end = cells_end[tau_index] - begin_ref
        for cell_index in range(begin, end):
            eta_index = packed_cells[cell_index, 0] + max_eta_index
            phi_index = packed_cells[cell_index, 1] + max_phi_index
            unpacked_cells[tau_index, eta_index, phi_index, :] = packed_cells[cell_index, :]
