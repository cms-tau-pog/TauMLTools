cimport numpy as np
import numpy as np
import math

cpdef np.ndarray[int, ndim=2] CalculatePtEtaBinIds(np.ndarray pt_bins, np.ndarray eta_bins,
                np.ndarray pt, np.ndarray eta, np.ndarray gen_tau):
    cdef Py_ssize_t n_entries = pt.shape[0], n
    cdef int pt_bin_id = 0, eta_bin_id = 0
    cdef np.ndarray[int, ndim=2] result = np.empty((n_entries, 2), dtype=np.int)

    for n in range(n_entries):
        if gen_tau[n] != 1: continue

        pt_bin_id = 0
        eta_bin_id = 0
        while pt_bins[pt_bin_id + 1] < pt[n]:
            pt_bin_id += 1
        abs_eta = abs(eta[n])
        while eta_bins[eta_bin_id + 1] < abs_eta:
            eta_bin_id += 1

        result[n, 0] = pt_bin_id
        result[n, 1] = eta_bin_id

    return result

cpdef np.ndarray[double, ndim=4] CalculateScaleFactors(np.ndarray pt_bins, np.ndarray eta_bins, np.ndarray tau_vs_e,
                np.ndarray tau_vs_mu, np.ndarray tau_vs_jet, np.ndarray gen_tau, np.ndarray weights, np.ndarray thr,
                np.ndarray target_eff, Py_ssize_t start_entry, Py_ssize_t n_entries, np.ndarray pt_bin_ids,
                np.ndarray eta_bin_ids):
    cdef Py_ssize_t i, n, k, n_pt_bins = len(pt_bins) - 1, n_eta_bins = len(eta_bins) - 1
    cdef Py_ssize_t pt_bin_id = 0, eta_bin_id = 0
    cdef double bin_eff, bin_eff_rel_err2, bin_eff_err, sf
    cdef int need_update
    cdef np.ndarray[double, ndim=4] result = np.zeros((3, n_pt_bins, n_eta_bins, 10))
    cdef np.ndarray[int] passed = np.zeros(3, dtype=np.int)

    for i in range(n_entries):
        n = start_entry + i
        if gen_tau[n] != 1: continue

        pt_bin_id = pt_bin_ids[n]
        eta_bin_id = eta_bin_ids[n]

        result[:, pt_bin_id, eta_bin_id, 4] += 1
        result[:, pt_bin_id, eta_bin_id, 5] += weights[n]
        result[:, pt_bin_id, eta_bin_id, 6] += weights[n] ** 2
        passed[0] = tau_vs_e[n] > thr[0]
        passed[1] = tau_vs_mu[n] > thr[1]
        passed[2] = tau_vs_jet[n] > thr[2]
        for k in range(3):
            if passed[k]:
                result[k, pt_bin_id, eta_bin_id, 7] += 1
                result[k, pt_bin_id, eta_bin_id, 8] += weights[n]
                result[k, pt_bin_id, eta_bin_id, 9] += weights[n] ** 2

    for k in range(3):
        for pt_bin_id in range(n_pt_bins):
            for eta_bin_id in range(n_eta_bins):
                if result[k, pt_bin_id, eta_bin_id, 5] == 0: continue
                bin_eff = result[k, pt_bin_id, eta_bin_id, 8] / result[k, pt_bin_id, eta_bin_id, 5]
                bin_eff_rel_err2 = result[k, pt_bin_id, eta_bin_id, 6] / (result[k, pt_bin_id, eta_bin_id, 5] ** 2)
                if result[k, pt_bin_id, eta_bin_id, 8] > 0:
                    bin_eff_rel_err2 += result[k, pt_bin_id, eta_bin_id, 9] / (result[k, pt_bin_id, eta_bin_id, 8] ** 2)
                bin_eff_err = bin_eff * math.sqrt(bin_eff_rel_err2)
                need_update = target_eff[k] < bin_eff - bin_eff_err or target_eff[k] > bin_eff + bin_eff_err
                if result[k, pt_bin_id, eta_bin_id, 7] == 0 and \
                        target_eff[k] * result[k, pt_bin_id, eta_bin_id, 4] < 1:
                    need_update = False
                sf = 1
                if need_update:
                    sf = 1 + (target_eff[k] - bin_eff) / min(target_eff[k], 1 - target_eff[k])
                    sf = min(2, max(0.5, sf))

                result[k, pt_bin_id, eta_bin_id, 0] = need_update
                result[k, pt_bin_id, eta_bin_id, 1] = sf
                result[k, pt_bin_id, eta_bin_id, 2] = bin_eff
                result[k, pt_bin_id, eta_bin_id, 3] = bin_eff_err

    return result

cpdef np.ndarray[double, ndim=2] ApplyScaleFactors(np.ndarray pt_bin_ids, np.ndarray eta_bin_ids, np.ndarray sf_results,
                    np.ndarray gen_tau, np.ndarray weights, np.ndarray orig_weights, double sum_w, double max_diff):
    cdef Py_ssize_t n_entries = weights.shape[0], n, k
    cdef Py_ssize_t pt_bin_id = 0, eta_bin_id = 0
    cdef res
    cdef np.ndarray[double, ndim=2] result = np.empty((n_entries, 3))
    cdef np.ndarray[double] sum_w_new = np.zeros(3)
    cdef np.ndarray[double] norm_sf = np.zeros(3)

    for n in range(n_entries):
        if gen_tau[n] != 1:
            result[n, :] = weights[n, :]
            continue

        pt_bin_id = pt_bin_ids[n]
        eta_bin_id = eta_bin_ids[n]

        for k in range(3):
            if sf_results[k, pt_bin_id, eta_bin_id, 0] < 0.5:
                res = weights[n, k]
            else:
                res = weights[n, k] * sf_results[k, pt_bin_id, eta_bin_id, 1]
            result[n, k] = max(orig_weights[n] / max_diff, min(orig_weights[n] * max_diff, res))
            sum_w_new[k] += result[n, k]

    norm_sf = sum_w / sum_w_new

    for n in range(n_entries):
        if gen_tau[n] != 1: continue
        for k in range(3):
            result[n, k] *= norm_sf[k]

    return result
