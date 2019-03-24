cimport numpy as np
import numpy as np
import math

cpdef np.ndarray[float, ndim=4] FillGrid(np.ndarray cells_begin, np.ndarray cells_end, int n_cells_eta,
                                         int n_cells_phi, np.ndarray cell_indices, np.ndarray packed_cells,
                                         np.ndarray other_inputs):
    cdef np.ndarray[float, ndim=4] unpacked_cells = np.zeros(
        ( cells_begin.shape[0], n_cells_eta, n_cells_phi, other_inputs.shape[1] + packed_cells.shape[1] ),
        dtype=np.float32)
    cdef int max_eta_index = (n_cells_eta - 1) / 2
    cdef int max_phi_index = (n_cells_phi - 1) / 2
    cdef int n_other_inputs = other_inputs.shape[1]
    begin_ref = cells_begin[0]
    for tau_index in range(cells_begin.shape[0]):
        begin = cells_begin[tau_index] - begin_ref
        end = cells_end[tau_index] - begin_ref
        for flat_cell_index in range(begin, end):
            eta_index = cell_indices[flat_cell_index, 0] + max_eta_index
            phi_index = cell_indices[flat_cell_index, 1] + max_phi_index
            unpacked_cells[tau_index, eta_index, phi_index, 0:n_other_inputs] = other_inputs[tau_index, :]
            unpacked_cells[tau_index, eta_index, phi_index, n_other_inputs:] = packed_cells[flat_cell_index, :]
    return unpacked_cells

cpdef np.ndarray[float, ndim=3] FillSequence(np.ndarray cells_begin, np.ndarray cells_end, int n_cells_eta,
                                             int n_cells_phi, np.ndarray cell_indices, np.ndarray packed_cells,
                                             np.ndarray other_inputs):
    cdef np.ndarray[float, ndim=3] unpacked_cells = np.zeros(
        ( cells_begin.shape[0], n_cells_eta * n_cells_phi, other_inputs.shape[1] + packed_cells.shape[1] ),
        dtype=np.float32)
    cdef int max_eta_index = (n_cells_eta - 1) / 2
    cdef int max_phi_index = (n_cells_phi - 1) / 2
    cdef int n_other_inputs = other_inputs.shape[1]
    begin_ref = cells_begin[0]
    for tau_index in range(cells_begin.shape[0]):
        begin = cells_begin[tau_index] - begin_ref
        end = cells_end[tau_index] - begin_ref
        for flat_cell_index in range(begin, end):
            unpacked_cells[tau_index, flat_cell_index - begin, 0:n_other_inputs] = other_inputs[tau_index, :]
            unpacked_cells[tau_index, flat_cell_index - begin, n_other_inputs:] = packed_cells[flat_cell_index, :]
    return unpacked_cells
