cimport numpy as np
import numpy as np
import math

cpdef np.ndarray[float, ndim=4] FillGrid(np.ndarray cells_begin, np.ndarray cells_end, int n_cells_eta,
                                          int n_cells_phi, np.ndarray packed_cells):
    cdef np.ndarray[float, ndim=4] unpacked_cells = np.zeros((cells_begin.shape[0], n_cells_eta, n_cells_phi,
                                                              packed_cells.shape[1]), dtype=np.float32)
    cdef int max_eta_index = (n_cells_eta - 1) / 2
    cdef int max_phi_index = (n_cells_phi - 1) / 2
    begin_ref = cells_begin[0]
    for tau_index in range(cells_begin.shape[0]):
        begin = cells_begin[tau_index] - begin_ref
        end = cells_end[tau_index] - begin_ref
        for cell_index in range(begin, end):
            eta_index = int(packed_cells[cell_index, 0] + max_eta_index)
            phi_index = int(packed_cells[cell_index, 1] + max_phi_index)
            unpacked_cells[tau_index, eta_index, phi_index, :] = packed_cells[cell_index, :]
    return unpacked_cells
