import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import pandas as pd
from numba import njit, prange
import math
import itertools
import time


@njit(cache=True, fastmath=True)
def curvature_equidistribution(t_nodes, y_nodes, max_nodes=500):
    if len(t_nodes) == max_nodes: return t_nodes, y_nodes
    n_states = y_nodes.shape[0]
    n_nodes = len(t_nodes)
    dy_dt = np.zeros((n_states, n_nodes - 1), dtype=np.float64)
    dt_diff = np.zeros(n_nodes - 1, dtype=np.float64)
    for j in range(n_nodes - 1): dt_diff[j] = t_nodes[j+1] - t_nodes[j]
    for i in range(n_states):
        for j in range(n_nodes - 1): dy_dt[i, j] = abs((y_nodes[i, j+1] - y_nodes[i, j]) / dt_diff[j])
    d2y_dt2 = np.zeros(n_nodes - 2, dtype=np.float64)
    for i in range(n_states):
        for j in range(n_nodes - 2):
            dt_mid = (t_nodes[j+2] - t_nodes[j]) / 2.0
            d2y_dt2[j] += abs((dy_dt[i, j+1] - dy_dt[i, j]) / dt_mid)
    curvature = np.zeros(n_nodes - 1, dtype=np.float64)
    if len(d2y_dt2) > 0:
        curvature[1:-1] = d2y_dt2
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
    alpha = 0.5 
    slope_sum = np.zeros(n_nodes - 1, dtype=np.float64)
    for i in range(n_states):
        for j in range(n_nodes - 1): slope_sum[j] += dy_dt[i, j]
    monitor = np.zeros(n_nodes - 1, dtype=np.float64)
    for j in range(n_nodes - 1): monitor[j] = math.sqrt(1.0 + slope_sum[j]**2 + alpha * curvature[j]**2)
    arc_length = np.zeros(n_nodes, dtype=np.float64)
    for i in range(1, n_nodes): arc_length[i] = arc_length[i-1] + monitor[i-1] * dt_diff[i-1]
    for i in range(n_nodes): arc_length[i] /= arc_length[-1]
    target_arc = np.linspace(0.0, 1.0, max_nodes)
    t_new = np.interp(target_arc, arc_length, t_nodes)
    t_new = np.sort(t_new)
    min_dt = 1e-5  
    for i in range(1, len(t_new) - 1):
        if t_new[i] <= t_new[i-1] + min_dt: t_new[i] = t_new[i-1] + min_dt
    t_new[0] = t_nodes[0]
    t_new[-1] = max(t_nodes[-1], t_new[-2] + min_dt)
    y_new = np.empty((n_states, max_nodes), dtype=np.float64)
    for i in range(n_states): y_new[i, :] = np.interp(t_new, t_nodes, y_nodes[i, :])
    return t_new, y_new

@njit(cache=True, fastmath=True)
def smoothed_max(x, delta=1e-3):
    return 0.5 * (math.sqrt(x**2 + delta) + x)

# ======================================================
# EXACT PMP SOLVER CORE [mi-Alt-PMP]
# ======================================================
@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, parallel=True)
def jitted_ode_n_sources(t, y, p_net, Eb_min, Eb_max, S, Z, B, Delta, eta_s, eta_dg, eta_g, X_g, C_grd, inv_eps, eps_softmax, dt_step):
    m, n_p = t.shape[0], p_net.shape[0]
    s, z, g, k_modes = Eb_min.shape[0], Z.shape[0], eta_g.shape[0], X_g.shape[0]
    dy = np.empty((2*s + 2*z, m), dtype=np.float64)
    inv_B, inv_Delta, inv_S = 1.0 / B, 1.0 / Delta, 1.0 / S
    dep_idx = s - 1
    
    tau_g_arr = np.zeros(k_modes, dtype=np.float64)
    for m_idx in range(k_modes):
        for j in range(g): tau_g_arr[m_idx] += (eta_g[j] * X_g[m_idx, j])**2 * inv_Delta[j]
            
    H_m_all = np.empty((m, k_modes), dtype=np.float64)
    p_s_m_all = np.empty((m, k_modes, s), dtype=np.float64)
    x_c_all = np.empty((m, k_modes), dtype=np.float64)
    W_all = np.empty((m, k_modes), dtype=np.float64)

    for i in prange(m):
        ti = t[i]
        pt = p_net[0] if ti <= 0.0 else p_net[-1] if ti >= n_p - 1 else np.interp(ti, np.arange(n_p), p_net)
        P_dg_term = 0.0
        for j in range(z): P_dg_term += eta_dg[j] * y[s+j, i]
            
        gamma_smooth = 10.0  
        eta_k1_local = np.empty(s - 1, dtype=np.float64)
        tau_1, tau_2 = 0.0, 0.0
        for j in range(s - 1):
            k_s_j = math.tanh(gamma_smooth * (pt - P_dg_term))
            eta_k1_local[j] = eta_s[j] ** k_s_j
            tau_1 += y[s+z+j, i] * eta_k1_local[j] * inv_S[j]
            tau_2 += (eta_s[j] ** (2 * k_s_j)) * inv_S[j]
            
        k_s_dep = math.tanh(gamma_smooth * (pt - P_dg_term))
        eta_k1_dep = eta_s[dep_idx] ** k_s_dep
        eta_2k_dep = eta_s[dep_idx] ** (2 * k_s_dep)
        inv_eta_k1_dep = 1.0 / eta_k1_dep
        lam_e_dep = y[s+z+dep_idx, i]
        max_H = -1e30
        
        for m_idx in range(k_modes):
            tau_g_m = tau_g_arr[m_idx]
            denom = eta_2k_dep + S[dep_idx] * (tau_2 + tau_g_m)
            
            # --- Active-Set Projection (Algorithm 1) ---
            P_s_trial = (eta_k1_dep * (pt - P_dg_term - tau_1) + lam_e_dep * (tau_2 + tau_g_m)) / denom
            
            if P_s_trial > 500.0:
                P_s_star = 500.0
                lam_mod = (500.0 * denom - eta_k1_dep * (pt - P_dg_term - tau_1)) / (tau_2 + tau_g_m + 1e-12)
            elif P_s_trial < -500.0:
                P_s_star = -500.0
                lam_mod = (-500.0 * denom - eta_k1_dep * (pt - P_dg_term - tau_1)) / (tau_2 + tau_g_m + 1e-12)
            else:
                P_s_star = P_s_trial
                lam_mod = lam_e_dep
                
            p_s_m_all[i, m_idx, dep_idx] = P_s_star
            
            # Formulate modified substitution chain W
            W = inv_eta_k1_dep * (S[dep_idx] * P_s_star - lam_mod)
            W_all[i, m_idx] = W
            
            H_cost = 0.5 * S[dep_idx] * P_s_star**2
            lam_p_s_term = lam_e_dep * P_s_star
            
            for j in range(g):
                p_g_val = W * inv_Delta[j] * X_g[m_idx, j] * eta_g[j]
                p_g_val = min(max(p_g_val, -2000.0), 2000.0)
                H_cost += 0.5 * Delta[j] * p_g_val**2
                
            for j in range(s - 1):
                p_s_val = inv_S[j] * (y[s+z+j, i] + eta_k1_local[j] * W)
                p_s_val = min(max(p_s_val, -500.0), 500.0)
                p_s_m_all[i, m_idx, j] = p_s_val
                H_cost += 0.5 * S[j] * p_s_val**2
                lam_p_s_term += y[s+z+j, i] * p_s_val
                
            lam_u_term = 0.0
            for j in range(z):
                u_opt = -inv_B[j] * y[2*s+z+j, i]
                u_opt = min(max(u_opt, -4000.0), 4000.0)
                H_cost += 0.5 * Z[j] * y[s+j, i]**2 + 0.5 * B[j] * u_opt**2
                lam_u_term += y[2*s+z+j, i] * u_opt
                
            H_m_val = H_cost + C_grd[m_idx] - lam_p_s_term + lam_u_term
            H_m_all[i, m_idx] = H_m_val
            if -H_m_val / eps_softmax > max_H: max_H = -H_m_val / eps_softmax
                
        sum_exp = 0.0
        for m_idx in range(k_modes):
            val = math.exp(-H_m_all[i, m_idx] / eps_softmax - max_H)
            x_c_all[i, m_idx] = val
            sum_exp += val
            
        A_t = 0.0
        for j in range(s): dy[j, i] = 0.0
            
        for m_idx in range(k_modes):
            prob = (x_c_all[i, m_idx] + 1e-12) / (sum_exp + k_modes * 1e-12)
            for j in range(s): dy[j, i] += prob * p_s_m_all[i, m_idx, j]
            A_t += prob * W_all[i, m_idx]
            
        # --- Formulate Adjoint State Derivatives ---
        # 1. Energy Storage Costate \dot{\lambda}_e (Eq 20)
        for j in range(s):
            dy[j, i] = -dy[j, i] * dt_step
            
            # Softplus Formulation (Restored)
            viol_l = 25.0 * (Eb_min[j] - y[j, i])
            viol_u = 25.0 * (y[j, i] - Eb_max[j])
            
            # Softplus approx for max(0, x) with your overflow protection
            v_l = viol_l / 25.0 if viol_l > 20.0 else math.log(1.0 + math.exp(viol_l)) / 25.0
            v_u = viol_u / 25.0 if viol_u > 20.0 else math.log(1.0 + math.exp(viol_u)) / 25.0
            
            # REMOVED the math.tanh() clamp to strictly preserve Eq 20 mechanics
            dy[s+z+j, i] = (v_u - v_l) * inv_eps 

        # 2. Generator Costate \dot{\lambda}_z (Eq 21)
        for j in range(z):
            dy[s+j, i] = -inv_B[j] * y[2*s+z+j, i]
            
            # Apply consistent Softplus EPFM to the Diesel Generator bounds
            viol_z_l = 25.0 * (0.0 - y[s+j, i]) 
            viol_z_u = 25.0 * (y[s+j, i] - 2000.0)
            
            v_z_l = viol_z_l / 25.0 if viol_z_l > 20.0 else math.log(1.0 + math.exp(viol_z_l)) / 25.0
            v_z_u = viol_z_u / 25.0 if viol_z_u > 20.0 else math.log(1.0 + math.exp(viol_z_u)) / 25.0
            
            penalty_z = (v_z_u - v_z_l) * inv_eps
            
            # Substituted slack stationarity + Fuel penalty + EPFM
            dy[2*s+z+j, i] = A_t * eta_dg[j] - Z[j] * y[s+j, i] + penalty_z
            
    return dy

@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def jitted_bc(ya, yb, e_init, e_target, p_dg_init):
    s, z = e_init.shape[0], p_dg_init.shape[0]
    res = np.empty(2*s + 2*z, dtype=np.float64)
    for j in range(s):
        res[j], res[s+z+j] = ya[j] - e_init[j], yb[j] - e_target[j]
    for j in range(z):
        res[s+j], res[2*s+z+j] = ya[s+j] - p_dg_init[j], yb[2*s+z+j] - 0.0
    return res

@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, parallel=True)
def jitted_jacobian_n_sources(t, y, p_net, Eb_min, Eb_max, S, Z, B, Delta, eta_s, eta_dg, eta_g, X_g, C_grd, inv_eps, eps_softmax, dt_step):
    m_nodes = t.shape[0]
    n_y = y.shape[0] 
    jac = np.zeros((n_y, n_y, m_nodes), dtype=np.float64)
    delta = 1e-8 
    f0 = jitted_ode_n_sources(t, y, p_net, Eb_min, Eb_max, S, Z, B, Delta, eta_s, eta_dg, eta_g, X_g, C_grd, inv_eps, eps_softmax, dt_step)
    for j in range(n_y):
        y_perturbed = y.copy()
        for i in range(m_nodes): y_perturbed[j, i] += delta
        f_perturbed = jitted_ode_n_sources(t, y_perturbed, p_net, Eb_min, Eb_max, S, Z, B, Delta, eta_s, eta_dg, eta_g, X_g, C_grd, inv_eps, eps_softmax, dt_step)
        for i in range(n_y):
            for k in prange(m_nodes): jac[i, j, k] = (f_perturbed[i, k] - f0[i, k]) / delta
    return jac

@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, parallel=True)
def jitted_extract_core(p_net, lam_e_prof, p_dg_prof, S, Delta, eta_s, eta_dg, eta_g, X_g, C_grd):
    m = p_net.shape[0]
    s, z, g, k_modes = lam_e_prof.shape[0], p_dg_prof.shape[0], eta_g.shape[0], X_g.shape[0]
    Ps_final, Pg_final, m_star = np.empty((s, m), dtype=np.float64), np.empty((g, m), dtype=np.float64), np.empty(m, dtype=np.int64)
    inv_S, inv_Delta = 1.0 / S, 1.0 / Delta
    dep_idx = s - 1
    
    tau_g_arr = np.zeros(k_modes, dtype=np.float64)
    for m_idx in range(k_modes):
        for j in range(g): tau_g_arr[m_idx] += (eta_g[j] * X_g[m_idx, j])**2 * inv_Delta[j]

    for i in prange(m):
        pt = p_net[i]
        P_dg_term = 0.0
        for j in range(z): P_dg_term += eta_dg[j] * p_dg_prof[j, i]
            
        eta_k1_local = np.empty(s - 1, dtype=np.float64)
        tau_1, tau_2 = 0.0, 0.0
        for j in range(s - 1):
            k_s_j = math.tanh(10.0 * (pt - P_dg_term))
            eta_k1_local[j] = eta_s[j] ** k_s_j
            tau_1 += lam_e_prof[j, i] * eta_k1_local[j] * inv_S[j]
            tau_2 += (eta_s[j] ** (2 * k_s_j)) * inv_S[j]

        k_s_dep = math.tanh(10.0 * (pt - P_dg_term))
        eta_k1_dep = eta_s[dep_idx] ** k_s_dep
        eta_2k_dep = eta_s[dep_idx] ** (2 * k_s_dep)
        inv_eta_k1_dep = 1.0 / eta_k1_dep
        lam_e_dep = lam_e_prof[dep_idx, i]

        H_m_vals = np.empty(k_modes, dtype=np.float64)
        p_s_m, p_g_m = np.empty((k_modes, s), dtype=np.float64), np.empty((k_modes, g), dtype=np.float64)

        for m_idx in range(k_modes):
            tau_g_m = tau_g_arr[m_idx]
            denom = eta_2k_dep + S[dep_idx] * (tau_2 + tau_g_m)
            
            # --- Active-Set Projection Extraction ---
            P_s_trial = (eta_k1_dep * (pt - P_dg_term - tau_1) + lam_e_dep * (tau_2 + tau_g_m)) / denom
            if P_s_trial > 500.0:
                P_s_star = 500.0
                lam_mod = (500.0 * denom - eta_k1_dep * (pt - P_dg_term - tau_1)) / (tau_2 + tau_g_m + 1e-12)
            elif P_s_trial < -500.0:
                P_s_star = -500.0
                lam_mod = (-500.0 * denom - eta_k1_dep * (pt - P_dg_term - tau_1)) / (tau_2 + tau_g_m + 1e-12)
            else:
                P_s_star = P_s_trial
                lam_mod = lam_e_dep
                
            p_s_m[m_idx, dep_idx] = P_s_star
            W = inv_eta_k1_dep * (S[dep_idx] * P_s_star - lam_mod)
            
            H_cost = 0.5 * S[dep_idx] * P_s_star**2
            lam_p_s_term = lam_e_dep * P_s_star
            
            for j in range(g):
                p_g_val = W * inv_Delta[j] * X_g[m_idx, j] * eta_g[j]
                p_g_m[m_idx, j] = min(max(p_g_val, -2000.0), 2000.0)
                H_cost += 0.5 * Delta[j] * p_g_m[m_idx, j]**2
                
            for j in range(s - 1):
                p_s_val = inv_S[j] * (lam_e_prof[j, i] + eta_k1_local[j] * W)
                p_s_m[m_idx, j] = min(max(p_s_val, -500.0), 500.0)
                H_cost += 0.5 * S[j] * p_s_m[m_idx, j]**2
                lam_p_s_term += lam_e_prof[j, i] * p_s_m[m_idx, j]
                
            H_m_vals[m_idx] = H_cost + C_grd[m_idx] - lam_p_s_term

        max_H_scaled = -1e30
        for m_idx in range(k_modes):
            val = -H_m_vals[m_idx] / 1e-5
            if val > max_H_scaled: max_H_scaled = val
                
        best_m_prob, best_m = -1.0, 0
        for m_idx in range(k_modes):
            exp_val = math.exp(-H_m_vals[m_idx] / 1e-5 - max_H_scaled)
            if exp_val > best_m_prob:
                best_m_prob = exp_val
                best_m = m_idx
                
        m_star[i] = best_m
        for j in range(g): Pg_final[j, i] = p_g_m[best_m, j]
        for j in range(s): Ps_final[j, i] = p_s_m[best_m, j]

    return Ps_final, Pg_final, m_star


def solve_ems_n_sources(p_net, time_arr, E_init, E_target, E_min, E_max, P_dg_init, S_diag, Z_diag, B_diag, Delta_diag, eta_s, eta_dg, eta_g, X_g, C_grd, dt, eps_array=None):
    num_s = len(E_init)
    num_z = len(P_dg_init)
    t_nodes = time_arr.copy().astype(np.float64)
    y_guess = np.zeros((2*num_s + 2*num_z, len(t_nodes)), dtype=np.float64)
    
    for j in range(num_s): y_guess[j, :] = E_init[j]
    for j in range(num_z): y_guess[num_s+j, :] = P_dg_init[j]
    
    epsilons = eps_array if eps_array is not None else np.logspace(6, -9, 1000) 
    final_sol = None
    
    def bvp_bc(ya, yb): return jitted_bc(ya, yb, E_init, E_target, P_dg_init)

    for idx, eps in enumerate(epsilons):
        inv_eps = -1.0 / max(eps, 1e-2)  
        eps_softmax = eps * 1e4
        
        def bvp_ode(t, y):
            return jitted_ode_n_sources(t, y, p_net, E_min, E_max, S_diag, Z_diag, B_diag, Delta_diag, eta_s, eta_dg, eta_g, X_g, C_grd, inv_eps, eps_softmax, dt)
        
        def bvp_jacobian(t, y):
            return jitted_jacobian_n_sources(t, y, p_net, E_min, E_max, S_diag, Z_diag, B_diag, Delta_diag, eta_s, eta_dg, eta_g, X_g, C_grd, inv_eps, eps_softmax, dt)
        
        if eps_softmax > 5e3:
            tol = 10.0
        elif eps_softmax > 1e-2:
            tol = 5.0
        else:
            tol = 15.0 
            
        sol = solve_bvp(bvp_ode, bvp_bc, t_nodes, y_guess, fun_jac=bvp_jacobian, tol=tol, max_nodes=50000)
        viol = 0.0
        
        if sol.success:
            nodes = len(sol.x)
            for j in range(num_s): 
                viol = max(viol, np.max(sol.y[j, :] - E_max[j]), np.max(E_min[j] - sol.y[j, :]), 0.0)
            t_nodes, y_guess = sol.x, sol.y
            if eps_array is None:
                print(f"Alt-PMP | Temp: {eps_softmax:.1e} | Nodes: {nodes:<4} | Viol: {viol:.4f}")
            final_sol = sol
            eps*=0.65
            
    return final_sol

def extract_pmp_trajectories(sol, time_cycle, p_net_full, num_s, num_z, S_diag, Delta_diag, eta_s, eta_dg, eta_g, X_g, C_grd):
    y_eval = sol.sol(time_cycle)
    Ps, Pg, m_star = jitted_extract_core(p_net_full, y_eval[num_s+num_z:2*num_s+num_z, :], y_eval[num_s:num_s+num_z, :], S_diag, Delta_diag, eta_s, eta_dg, eta_g, X_g, C_grd)
    return {"p_s": Ps, "p_dg": y_eval[num_s:num_s+num_z, :], "e": y_eval[0:num_s, :], "p_g": Pg, "m_star": m_star}
