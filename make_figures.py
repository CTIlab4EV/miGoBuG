"""Figures of the paper from ONE results folder (needs LaTeX, text.usetex):

    python make_figures.py reference_results

Writes figures/:
  Figs. 3-4  Results_ExecTime_*.svg/.pdf          execution time per dataset
  Figs. 5-7  Results_Benchmark_Config_<C>_*.svg/.pdf  trajectories (I: dataset 1, III: dataset 3, V: dataset 10)
  preview_*.png  stacked previews of Figs. 5-7, and the data plotted (*.csv)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

import run_benchmarks as R

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "reference_results")
out = Path("figures"); out.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(folder / "main.csv")))
by = {(r["config"], int(r["case"])): r for r in rows}
CFG = ["I", "II", "III", "IV", "V"]
NAMES = {"I": "Isolated Microgrid", "II": "Single Grid Node", "III": "Dual Grid Nodes",
         "IV": "Three Grid Nodes", "V": "Four Grid Nodes"}
cases = list(range(1, 11))
num = lambda r, k: float(r[k]) if r.get(k) not in ("", None) else np.nan


def save(fig, name):
    fig.savefig(out / f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Common style
# =====================================================================
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
                     'font.size': 22, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.linewidth': 1.7})   # frame as thick on the page as in Figs. 5-7 (1.5 at their scale)
xt = [rf"\textsc{{Case}} {i}" for i in cases]
x = np.arange(len(cases))

# =====================================================================
# Figs. 3-4: execution time, Configurations I-V
# =====================================================================
with open(out / "runtimes.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config", "dataset", "gurobi_status", "gurobi_time_s", "migobug_time_s"])
    for c in CFG:
        for i in cases:
            r = by[(c, i)]; w.writerow([c, i, r["MIQP_status"], num(r, "MIQP_time"), num(r, "t_total")])

tstyles = {'ALT_PMP': {'c': '#d1e5f0', 'h': '\\\\', 'lbl': 'miGoBuG'},
           'MIP':     {'c': '#fddbc7', 'h': 'o',    'lbl': 'MINLP'}}
width_time = 0.25
offsets_time_2 = np.array([-0.5, 0.5]) * width_time
MIP_SCALE_I = 100                               # Configuration I: MINLP bars scaled for visibility
CAP = 15.0                                      # visual cap [s]; longer bars are labelled with their time
def time_figure(cfgs, name):
    """one panel per configuration, 14 x 3.4 in each, one legend on top"""
    fig_time, axes_time = plt.subplots(len(cfgs), 1, figsize=(14, 3.6 * len(cfgs)), facecolor='white')
    for idx, c in enumerate(cfgs):
        ax_t = axes_time[idx]
        ax_t.grid(True, axis='y', linestyle=':', color='gray', alpha=0.6, zorder=0)
        time_pmp = np.array([num(by[(c, i)], "t_total") for i in cases])
        time_mip = np.array([num(by[(c, i)], "MIQP_time") for i in cases])
        # Configuration I: the MINLP takes ~0.005 s; scaled by MIP_SCALE_I so the bars are visible
        # (the scaled bars stay below every miGoBuG bar), and the legend says so
        mip_scale = MIP_SCALE_I if c == "I" else 1
        assert np.all(mip_scale * time_mip < time_pmp) or mip_scale == 1
        time_mip = mip_scale * time_mip
        limit = np.array([by[(c, i)]["MIQP_status"] != "optimal" for i in cases])
        capped = np.any(np.concatenate([time_pmp, time_mip]) > CAP)
        top = CAP if capped else max(time_pmp.max(), time_mip.max())

        def bars_of(offset_idx, t, key, lim=None):
            b = ax_t.bar(x + offsets_time_2[offset_idx], np.minimum(t, top), width_time,
                         facecolor=tstyles[key]['c'], edgecolor='black', hatch=tstyles[key]['h'], zorder=3)
            for i, val in enumerate(t):
                if val > top:
                    lab = r'$\ge$3600' if (lim is not None and lim[i]) else rf'{val:.0f}'
                    # horizontal label just above the truncated bar
                    ax_t.text(b[i].get_x() + b[i].get_width() / 2, top * 1.02, lab, rotation=0,
                              ha='center', va='bottom', fontsize=15, zorder=5)
        bars_of(0, time_pmp, 'ALT_PMP')
        bars_of(1, time_mip, 'MIP', limit)
        # 20 s when bars are truncated at 10 s; Configuration I (all times < 1 s) ends at 1.5 s so the
        # legend does not hide its bars
        ax_t.set_ylim(0, 25.0 if capped else 1.5 * np.ceil(top))
        if capped:
            ax_t.set_yticks(np.arange(0, 25.0 + 1e-9, 5.0))
        ax_t.set_ylabel(r'\textsc{Execution Time [s]}', fontsize=20)
        ax_t.set_title(rf'\textbf{{Configuration {c}: {NAMES[c]}}}', fontsize=24, pad=10)
        ax_t.set_xticks(x); ax_t.set_xticklabels(xt, fontsize=20, rotation=0, ha='center')
        ax_t.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax_t.tick_params(axis='y', labelsize=18)
        lbl_mip = tstyles['MIP']['lbl'] + (rf' ($\times${mip_scale})' if mip_scale != 1 else '')
        handles = [Patch(facecolor=tstyles['ALT_PMP']['c'], edgecolor='black', hatch=tstyles['ALT_PMP']['h'], label=tstyles['ALT_PMP']['lbl']),
                   Patch(facecolor=tstyles['MIP']['c'], edgecolor='black', hatch=tstyles['MIP']['h'], label=lbl_mip)]
        ax_t.legend(handles=handles, loc='upper left', ncol=2, edgecolor='black',
                    framealpha=1.0, fontsize=18)              # inside the panel, as in the original figure
    fig_time.tight_layout()
    fig_time.subplots_adjust(hspace=0.55)
    save(fig_time, name)


time_figure(["I", "II"], "Results_ExecTime_I_II")            # Fig. 3
time_figure(["III", "IV", "V"], "Results_ExecTime_III_V")  # Fig. 4


# =====================================================================
# Figs. 5-7: trajectories (separate figure per panel, and all panels stacked)
# =====================================================================
def darken_color(color, factor=0.7):
    rgb = mcolors.to_rgb(color)
    return tuple(val * factor for val in rgb)


def apply_sci_notation(ax, power=2):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((power, power))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(plt.rcParams['ytick.labelsize'])


def plot(t_grid, p_net_full, pmp_data, mip_data, Eb_min, Eb_max, X_g, tag, stacked=False):
    """stacked=False: one figure per panel (cells 0/2 of the notebook); True: one stacked figure (cells 1/3)."""
    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm', 'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 22,
        'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 19,
    })
    num_z = mip_data['p_dg'].shape[0]
    num_g = mip_data['p_g'].shape[0]
    num_s = mip_data['e'].shape[0]
    zoom_idx = len(t_grid)
    tz = t_grid[:zoom_idx]
    names = []
    n_panels = 3 + (2 if num_g > 0 else 0)
    if stacked:
        fig_s, axs_s = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels), sharex=True)
    k_panel = [0]

    def new_ax():
        if stacked:
            ax = axs_s[k_panel[0]]; k_panel[0] += 1
            return fig_s, ax
        return plt.subplots(figsize=(12, 4))

    def xlabel(ax):
        if not stacked or ax is axs_s[-1]:
            ax.set_xlabel(r'\textsc{Time [Hours]}')

    def apply_custom_style(ax):
        ax.set_facecolor('white')
        ax.grid(True, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', which='major')

    color_dg = ['#3182bd', '#4daf4a', '#e41a1c']
    color_grid = ['#ff7f00', '#00ced1', '#8c564b', '#9467bd']   # fourth color for Configuration V
    color_batt = ['#e7298a', '#7570b3', '#1b9e77']

    def apply_split_legends(ax, h_left, l_left, h_right, l_right, ncol_left=2, ncol_right=2, handlelength=2.0):
        leg1 = ax.legend(h_left, l_left, loc='upper left', borderaxespad=0.2, ncol=ncol_left,
                         edgecolor='black', facecolor='white', framealpha=0.5, columnspacing=1.0,
                         handlelength=handlelength)
        ax.add_artist(leg1)
        ax.legend(h_right, l_right, loc='upper right', borderaxespad=0.2, ncol=ncol_right,
                  edgecolor='black', facecolor='white', framealpha=0.5, columnspacing=1.0,
                  handlelength=handlelength)

    def finish(fig, ax, n, panel):
        ax.set_xlim(tz[0], tz[-1])
        if stacked:
            return
        fig.tight_layout()
        name = f"Results_Benchmark_Config_{tag}_{n}_{panel}"
        save(fig, name); names.append(name)

    n = 0
    # ---- Diesel generator power
    fig1, ax1 = new_ax()
    h_left, l_left, h_right, l_right = [], [], [], []
    line_ld, = ax1.plot(tz, p_net_full[:zoom_idx], color="#000000", linewidth=1.5, linestyle=':')
    h_right.append(line_ld); l_right.append(r'$P_{ld}$')
    for j in range(num_z):
        c_mip = color_dg[j % len(color_dg)]; c_pmp = darken_color(c_mip, factor=0.7)
        l1, = ax1.plot(tz, mip_data['p_dg'][j][:zoom_idx], color=c_mip, linestyle='-', linewidth=2.5, alpha=0.9)
        l2, = ax1.plot(tz, pmp_data['p_dg'][j][:zoom_idx], color=c_pmp, linestyle='--', linewidth=1.5, alpha=1.0)
        hl, ll = (h_left, l_left) if j == 0 else (h_right, l_right)
        hl.extend([l1, l2]); ll.extend([rf'$P_{{z,{j+1}}}^{{MINLP}}$', rf'$P_{{z,{j+1}}}^{{miGoBuG}}$'])
    ax1.set_title(r'\textbf{Diesel generator power}', fontweight='bold')
    ax1.set_ylabel(r'\textsc{Power [kW]}'); xlabel(ax1)
    apply_custom_style(ax1); apply_sci_notation(ax1, power=2)
    y_min, y_max = ax1.get_ylim(); ax1.set_ylim(y_min, y_max + (y_max - y_min) * 0.35)
    apply_split_legends(ax1, h_left, l_left, h_right, l_right, ncol_left=3, ncol_right=2)
    n += 1; finish(fig1, ax1, n, "Diesel")

    # ---- Grid power
    if num_g > 0:
        fig2, ax2 = new_ax()
        h_left, l_left, h_right, l_right = [], [], [], []
        for j in range(num_g):
            c_mip = color_grid[j % len(color_grid)]; c_pmp = darken_color(c_mip, factor=0.7)
            l1, = ax2.plot(tz, mip_data['p_g'][j][:zoom_idx], color=c_mip, linestyle='-', linewidth=2.5, alpha=0.9)
            l2, = ax2.plot(tz, pmp_data['p_g'][j][:zoom_idx], color=c_pmp, linestyle='--', linewidth=1.5, alpha=1.0)
            hl, ll = (h_left, l_left) if j < max(1, num_g // 2) else (h_right, l_right)
            hl.extend([l1, l2]); ll.extend([rf'$P_{{g,{j+1}}}^{{MINLP}}$', rf'$P_{{g,{j+1}}}^{{miGoBuG}}$'])
        ax2.set_title(r'\textbf{Grid power}', fontweight='bold')
        ax2.set_ylabel(r'\textsc{Power [kW]}'); xlabel(ax2)
        apply_custom_style(ax2); apply_sci_notation(ax2, power=2)
        y_min, y_max = ax2.get_ylim()
        many = num_g > 2                                   # Configuration V: more legend entries, more headroom
        ax2.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * (1.0 if many else 0.65))
        apply_split_legends(ax2, h_left, l_left, h_right, l_right, ncol_left=2, ncol_right=2)
        n += 1; finish(fig2, ax2, n, "Grid")

    # ---- Battery power
    fig3, ax3 = new_ax()
    h_left, l_left, h_right, l_right = [], [], [], []
    for j in range(num_s):
        c_mip = color_batt[j % len(color_batt)]; c_pmp = darken_color(c_mip, factor=0.7)
        l1, = ax3.plot(tz, mip_data['p_s'][j][:zoom_idx], color=c_mip, linestyle='-', linewidth=2.5, alpha=0.9)
        l2, = ax3.plot(tz, pmp_data['p_s'][j][:zoom_idx], color=c_pmp, linestyle='--', linewidth=1.5, alpha=1.0)
        hl, ll = (h_left, l_left) if j == 0 else (h_right, l_right)
        hl.extend([l1, l2]); ll.extend([rf'$P_{{s,{j+1}}}^{{MINLP}}$', rf'$P_{{s,{j+1}}}^{{miGoBuG}}$'])
    ax3.set_title(r'\textbf{Battery power}', fontweight='bold')
    ax3.set_ylabel(r'\textsc{Power [kW]}'); xlabel(ax3)
    apply_custom_style(ax3); apply_sci_notation(ax3, power=2)
    y_min, y_max = ax3.get_ylim(); ax3.set_ylim(y_min, y_max + (y_max - y_min) * 0.40)
    apply_split_legends(ax3, h_left, l_left, h_right, l_right, ncol_left=2, ncol_right=2)
    n += 1; finish(fig3, ax3, n, "BatteryPower")

    # ---- Grid connection (Gantt chart)
    if num_g > 0:
        fig4, ax4 = new_ax()
        h_left, l_left, h_right, l_right = [], [], [], []
        x_g_mip = X_g[mip_data['m_star'].astype(int)].T
        x_g_pmp = X_g[pmp_data['m_star'].astype(int)].T
        y_ticks, y_tick_labels = [], []
        for j in range(num_g):
            c_mip = color_grid[j % len(color_grid)]; c_pmp = darken_color(c_mip, factor=0.7)
            center_y = j * 2.5
            y_ticks.append(center_y); y_tick_labels.append(f'G{j+1}')
            mip_mask = (x_g_mip[j][:zoom_idx] > 0.5).astype(float)
            pmp_mask = (x_g_pmp[j][:zoom_idx] > 0.5).astype(float)
            m_poly = ax4.fill_between(tz, center_y, center_y + (0.9 * mip_mask), facecolor=c_mip, step='post', alpha=0.9)
            p_poly = ax4.fill_between(tz, center_y - (0.9 * pmp_mask), center_y, facecolor=c_pmp, step='post',
                                      alpha=1.0, hatch='//')
            hl, ll = (h_left, l_left) if j < max(1, num_g // 2) else (h_right, l_right)
            hl.extend([m_poly, p_poly]); ll.extend([rf'$x_{{g,{j+1}}}^{{MINLP}}$', rf'$x_{{g,{j+1}}}^{{miGoBuG}}$'])
            if j < num_g - 1:
                ax4.axhline(center_y + 1.25, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
        ax4.set_title(r'\textbf{Grid connection}', fontweight='bold')
        ax4.set_ylabel(r'\textsc{State \{0,1\}}'); xlabel(ax4)
        ax4.set_yticks(y_ticks); ax4.set_yticklabels(y_tick_labels)
        ax4.set_ylim(-2, (num_g - 1) * 2.5 + (9.0 if num_g > 2 else 2.5))   # room for the legend above the top node
        apply_custom_style(ax4)
        apply_split_legends(ax4, h_left, l_left, h_right, l_right, ncol_left=2, ncol_right=2)
        n += 1; finish(fig4, ax4, n, "GridState")

    # ---- Battery energy
    fig5, ax5 = new_ax()
    h_left, l_left, h_right, l_right = [], [], [], []
    for j in range(num_s):
        c_mip = color_batt[j % len(color_batt)]; c_pmp = darken_color(c_mip, factor=0.7)
        initial_val = mip_data['e'][j][0]
        l1, = ax5.plot(tz, mip_data['e'][j][:zoom_idx], color=c_mip, linestyle='-', linewidth=2.5, alpha=0.9)
        l2, = ax5.plot(tz, pmp_data['e'][j][:zoom_idx], color=c_pmp, linestyle='--', linewidth=1.5, alpha=1.0)
        h_right.append(l1); l_right.append(rf'$E_{{s,{j+1}}}^{{MINLP}}$')
        h_right.append(l2); l_right.append(rf'$E_{{s,{j+1}}}^{{miGoBuG}}$')
        ax5.axhline(initial_val, color=c_mip, linestyle='-.', linewidth=1.0, alpha=0.7)
        line_lim = ax5.axhline(Eb_max[j], color=c_pmp, linestyle=':', linewidth=2, alpha=0.8)
        ax5.axhline(Eb_min[j], color=c_pmp, linestyle=':', linewidth=2, alpha=0.8)
        h_left.append(line_lim); l_left.append(rf'$E_{{s,{j+1}}}^{{\pm}}$')
    ax5.set_title(r'\textbf{Battery energy}', fontweight='bold')
    ax5.set_ylabel(r'\textsc{Energy [kWh]}'); xlabel(ax5)
    apply_custom_style(ax5); apply_sci_notation(ax5, power=2)
    y_min, y_max = ax5.get_ylim(); ax5.set_ylim(y_min, y_max + (y_max - y_min) * 0.325)
    apply_split_legends(ax5, h_left, l_left, h_right, l_right, ncol_left=2, ncol_right=4,
                        handlelength=1.4)       # keeps the two legends apart
    n += 1; finish(fig5, ax5, n, "BatteryEnergy")
    if stacked:
        fig_s.tight_layout()
        fig_s.subplots_adjust(hspace=0.15)
        name = f"Results_Benchmark_Config_{tag}"                 # file name used in the manuscript
        save(fig_s, name); names.append(name)
    return names


cfg = dict(R.CFG); days = R.load_days(cfg)
for c, i in (("I", 1), ("III", 3), ("V", 10)):   # Figs. 5, 6 and 7
    case = R.make_case(cfg, R.case_load(cfg, days, i), c)
    G = np.load(folder / "traj" / f"main_{c}_case{i}_MIQP.npz")
    M = np.load(folder / "traj" / f"main_{c}_case{i}_miGoBuG.npz")
    N = M["e"].shape[1]
    t_grid = np.arange(N) * case.dt
    mip = dict(p_dg=G["p_dg"][:, :N], p_g=G["p_g"][:, :N], p_s=(G["p_dis"] - G["p_ch"])[:, :N],
               e=G["e"][:, :N], m_star=G["seq"][:N])
    pmp = dict(p_dg=M["p_dg"], p_g=M["p_g"], p_s=M["p_s"], e=M["e"], m_star=M["seq"])
    print(f"{c}-{i}: topology differs in {int(np.sum(G['seq'][:N] != M['seq']))} of {N} intervals; "
          f"J_MINLP={num(by[(c, i)], 'MIQP_J_inc'):.6g}, J_miGoBuG_ZOH={num(by[(c, i)], 'J_miGoBuG_ZOH'):.6g}")
    names = plot(t_grid, case.p_net, pmp, mip, case.E_min, case.E_max, case.X_g, c)
    plot(t_grid, case.p_net, pmp, mip, case.E_min, case.E_max, case.X_g, c, stacked=True)
    # stacked preview (as the panels appear in the paper)
    from PIL import Image
    ims = [Image.open(out / f"{nm}.png").convert("RGB") for nm in names]
    W = max(im.width for im in ims)
    canvas = Image.new("RGB", (W, sum(im.height for im in ims)), "white")
    yy = 0
    for im in ims:
        canvas.paste(im, (0, yy)); yy += im.height
    canvas.save(out / f"preview_Fig_Config_{c}_dataset{i}.png")
    with open(out / f"trajectories_{c}_dataset{i}.csv", "w", newline="") as f:
        cols = {"t_h": t_grid, "P_ld": case.p_net[:N]}
        for key in ("p_dg", "p_g", "p_s", "e"):
            for j in range(mip[key].shape[0]):
                cols[f"{key}{j+1}_MINLP"] = mip[key][j]; cols[f"{key}{j+1}_miGoBuG"] = pmp[key][j]
        cols["topology_MINLP"] = mip["m_star"]; cols["topology_miGoBuG"] = pmp["m_star"]
        w = csv.writer(f); w.writerow(list(cols)); w.writerows(zip(*cols.values()))
print("figures written to", out)
