"""Desktop GUI for GIOP: load field spectra, build R_rs, invert, inspect, export.

Tkinter + matplotlib, so there is nothing to install beyond the package itself.
Launch with ``giop-gui`` or ``python -m giop.gui``.

Workflow the GUI enforces, which is the workflow THEORY.md sect. 10 describes:

    1. load scans          (.sed triplet, or a CSV that is already R_rs)
    2. build R_rs          (panel reflectance, rho, residual correction)
    3. choose the window   (aph* is only defined 400-700 nm)
    4. invert              (GIOP-DC by default, every knob exposed)
    5. read the caveats    (the assumptions panel is not optional or hideable)
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "giop.gui needs tkinter. On Debian/Ubuntu: sudo apt install python3-tk"
    ) from exc

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure  # noqa: E402

from . import empirical, io as gio  # noqa: E402
from .inversion import giop  # noqa: E402
from .model import ConfigurationError, GiopConfig  # noqa: E402

CAVEATS = (
    "What these numbers are conditional on (THEORY.md sect. 11):\n"
    "• S_dg and eta are PRESCRIBED, not fitted. a_dg and a_ph are strongly\n"
    "  collinear in the blue, so their split is set by the assumed a_ph* shape.\n"
    "• GIOP returns NO uncertainty. The cost is unweighted least squares.\n"
    "• rho (sky-glint) is the dominant error above water and is sea-state\n"
    "  dependent. 0.028 assumes wind < 5 m/s, 40 deg view, 135 deg azimuth.\n"
    "• The retrieved aph amplitude equals chlorophyll only for Bricaud aph*.\n"
    "• No Raman or fluorescence term is in the forward model."
)


class GiopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GIOP - ocean colour inversion for field spectra")
        self.geometry("1280x860")

        self.wl = None
        self.rrs = None
        self.source = "none"
        self.result = None
        self.rrs_meta = None

        self._build_layout()

    # -- layout ---------------------------------------------------------------
    def _build_layout(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(left, width=380)
        nb.pack(fill=tk.Y, expand=True)
        self._tab_load(nb)
        self._tab_model(nb)

        ttk.Button(left, text="Run inversion", command=self.run).pack(
            fill=tk.X, pady=(8, 2))
        ttk.Button(left, text="Export result CSV", command=self.export).pack(fill=tk.X)

        self.status = tk.StringVar(value="No data loaded.")
        ttk.Label(left, textvariable=self.status, wraplength=360,
                  foreground="#004488").pack(fill=tk.X, pady=6)

        cav = tk.Text(left, height=11, wrap=tk.WORD, bg="#fff8e5",
                      relief=tk.FLAT, font=("TkDefaultFont", 8))
        cav.insert("1.0", CAVEATS)
        cav.configure(state=tk.DISABLED)
        cav.pack(fill=tk.X, pady=4)

        self.fig = Figure(figsize=(9, 8), dpi=100)
        self.ax_rrs = self.fig.add_subplot(311)
        self.ax_a = self.fig.add_subplot(312)
        self.ax_bb = self.fig.add_subplot(313)
        self.fig.tight_layout(pad=2.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def _tab_load(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="1. Data")

        ttk.Label(f, text="Spectral Evolution .sed (above-water triplet)",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        self.p_target = self._file_row(f, "Water target L_t")
        self.p_sky = self._file_row(f, "Sky L_sky")
        self.p_panel = self._file_row(f, "Panel L_p")

        g = ttk.Frame(f)
        g.pack(fill=tk.X, pady=4)
        self.panel_refl = self._entry(g, "Panel reflectance R_p", "0.99", 0)
        self.rho = self._entry(g, "rho (sky glint)", "0.028", 1)
        self.use_col = self._combo(g, "Use column", ["radiance", "ratio"], 2)
        self.residual = self._combo(
            g, "Residual correction", ["none", "nir_zero", "nir_similarity"], 3)

        ttk.Button(f, text="Build R_rs from triplet",
                   command=self.load_triplet).pack(fill=tk.X, pady=4)

        ttk.Separator(f).pack(fill=tk.X, pady=8)
        ttk.Label(f, text="Or load data that is already R_rs",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        ttk.Button(f, text="Load R_rs CSV...",
                   command=self.load_csv).pack(fill=tk.X, pady=2)
        ttk.Button(f, text="Load GIOP demo spectrum",
                   command=self.load_demo).pack(fill=tk.X, pady=2)
        return f

    def _tab_model(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="2. Model")

        g = ttk.Frame(f)
        g.pack(fill=tk.X)
        self.wl_lo = self._entry(g, "Window low (nm)", "400", 0)
        self.wl_hi = self._entry(g, "Window high (nm)", "700", 1)
        self.resample_to = self._combo(
            g, "Resample to",
            ["native"] + sorted(gio.SATELLITE_BANDS), 2)
        self.aph = self._combo(g, "aph*", ["bricaud", "ciotti", "gsm", "chase"], 3)
        self.sf = self._entry(g, "Ciotti S_f", "0.5", 4)
        self.sdg = self._combo(g, "S_dg", ["0.018", "qaa", "obpg", "gsm"], 5)
        self.eta = self._combo(g, "eta", ["qaa", "gsm", "1.0"], 6)
        self.fq = self._combo(g, "AOP-IOP", ["gordon", "morel"], 7)
        self.inv = self._combo(g, "Solver", ["fmin", "lmi"], 8)
        self.trans = self._combo(g, "Air-sea", ["lee", "flat"], 9)
        self.qc = self._entry(g, "QC max rel. diff (blank=off)", "0.33", 10)
        self.chl_alg = self._combo(
            g, "Seed chl algorithm", ["oc4", "oc3m", "oc3v", "manual"], 11)
        self.chl_manual = self._entry(g, "Manual seed chl", "0.2", 12)

        self.fit_shapes = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f, text="Also fit S_dg and eta (NOT GIOP-DC; 5 parameters)",
            variable=self.fit_shapes).pack(anchor="w", pady=6)
        return f

    def _file_row(self, parent, label):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        var = tk.StringVar()
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="...", width=3,
                   command=lambda: self._pick(var)).pack(side=tk.LEFT)
        return var

    def _pick(self, var):
        p = filedialog.askopenfilename(
            filetypes=[("Spectral Evolution", "*.sed"), ("All files", "*.*")])
        if p:
            var.set(p)

    def _entry(self, parent, label, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="e")
        return var

    def _combo(self, parent, label, values, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        var = tk.StringVar(value=values[0])
        ttk.Combobox(parent, textvariable=var, values=values, width=10,
                     state="readonly").grid(row=row, column=1, sticky="e")
        return var

    # -- actions --------------------------------------------------------------
    def load_triplet(self):
        try:
            paths = [self.p_target.get(), self.p_sky.get(), self.p_panel.get()]
            if not all(paths):
                messagebox.showwarning(
                    "Missing files",
                    "The above-water method needs all three scans: water target, "
                    "sky, and reference panel (THEORY.md G16/G17).")
                return
            t, s, p = (gio.read_sed(x) for x in paths)
            res = gio.rrs_from_sed_triplet(
                t, s, p,
                panel_reflectance=float(self.panel_refl.get()),
                rho=float(self.rho.get()),
                residual=self.residual.get(),
                use=self.use_col.get(),
            )
            self.wl, self.rrs = res.wavelength, res.rrs
            self.rrs_meta = res
            self.source = f"sed triplet ({Path(paths[0]).name})"
            note = " | ".join(res.notes) if res.notes else "no warnings"
            self.status.set(
                f"R_rs built from {len(self.wl)} bands, "
                f"{self.wl.min():.0f}-{self.wl.max():.0f} nm. rho={res.rho}. {note}")
            self.plot_input()
        except Exception as exc:
            self._error(exc)

    def load_csv(self):
        try:
            p = filedialog.askopenfilename(
                filetypes=[("CSV/TSV", "*.csv *.txt *.tsv"), ("All", "*.*")])
            if not p:
                return
            wl, spectra, labels = gio.read_csv_spectra(p)
            self.wl, self.rrs = wl, spectra[0]
            self.rrs_meta = None
            self.source = f"csv ({Path(p).name}, {labels[0] if labels else 'col 0'})"
            self.status.set(
                f"Loaded {spectra.shape[0]} spectra; showing the first. "
                f"{len(wl)} bands, {wl.min():.0f}-{wl.max():.0f} nm.")
            self.plot_input()
        except Exception as exc:
            self._error(exc)

    def load_demo(self):
        self.wl = np.array([412.0, 443, 490, 510, 555, 670])
        self.rrs = np.array([0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051])
        self.rrs_meta = None
        self.source = "GIOP demo (run_giop.m)"
        self.status.set(
            "Demo spectrum from run_giop.m. GIOP-DC should return "
            "adg=0.0441, bbp=0.0033, aph=0.3693.")
        self.plot_input()

    def run(self):
        if self.wl is None:
            messagebox.showwarning("No data", "Load a spectrum first.")
            return
        try:
            wl, rrs = self._prepare_grid()
            chl = self._seed_chl(wl, rrs)
            cfg = GiopConfig(
                fq=self.fq.get(),
                trans=self.trans.get(),
                eta=_num_or_str(self.eta.get()),
                sdg=_num_or_str(self.sdg.get()),
                aph=self.aph.get(),
                sf=float(self.sf.get()),
                inv=self.inv.get(),
                qc=float(self.qc.get()) if self.qc.get().strip() else None,
                fit_shapes=bool(self.fit_shapes.get()),
            )
            self.result = giop(wl, rrs, chl, cfg=cfg)
            self._report()
            self.plot_result()
        except ConfigurationError as exc:
            messagebox.showerror("Configuration refused", str(exc))
        except Exception as exc:
            self._error(exc)

    def _prepare_grid(self):
        lo, hi = float(self.wl_lo.get()), float(self.wl_hi.get())
        wl, rrs = self.wl, self.rrs
        target = self.resample_to.get()
        if target != "native":
            centres = np.array(gio.SATELLITE_BANDS[target], dtype=float)
            centres = centres[(centres >= lo) & (centres <= hi)]
            _, rrs = gio.gaussian_resample(wl, rrs, centres, fwhm=10.0)
            wl = centres
        m = (wl >= lo) & (wl <= hi) & np.isfinite(rrs)
        if m.sum() < 4:
            raise ValueError(
                f"only {m.sum()} finite bands in [{lo}, {hi}] nm; GIOP needs at "
                "least the three anchor bands plus one")
        return wl[m], rrs[m]

    def _seed_chl(self, wl, rrs):
        alg = self.chl_alg.get()
        if alg == "manual":
            return float(self.chl_manual.get())
        pick = lambda t: float(rrs[int(np.argmin(np.abs(wl - t)))])  # noqa: E731
        if alg == "oc4":
            return float(empirical.get_oc(pick(443), pick(490), pick(510),
                                          pick(555), "oc4"))
        if alg == "oc3m":
            return float(empirical.get_oc(pick(443), pick(488), -1, pick(547), "oc3m"))
        return float(empirical.get_oc(pick(443), pick(486), -1, pick(551), "oc3v"))

    def _report(self):
        r = self.result
        if not r.converged:
            self.status.set("Inversion did NOT converge; upstream returns -999 here.")
            return
        qc = {True: "pass", False: "FAIL", None: "off"}[r.qc_passed]
        self.status.set(
            f"a_dg(443) = {r.adg443:.4g} m^-1 | b_bp(443) = {r.bbp443:.4g} m^-1 | "
            f"aph amp = {r.chl:.4g} (seed chl {r.chl_seed:.3g}) | "
            f"S_dg = {r.sdg:.4g} nm^-1, eta = {r.eta:.3f} | "
            f"cost = {r.cost:.3g} | QC {qc} | source: {self.source}")

    def export(self):
        if self.result is None:
            messagebox.showwarning("Nothing to export", "Run an inversion first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        gio.write_result_csv(p, [self.result], [self.source])
        spec = Path(p).with_name(Path(p).stem + "_spectra.csv")
        r = self.result
        np.savetxt(
            spec,
            np.column_stack([r.wl, r.rrs_obs, r.rrs_model_above, r.aph, r.adg,
                             r.apg, r.bbp, r.aw, r.bbw]),
            delimiter=",", header=(
                "wavelength_nm,rrs_obs_sr-1,rrs_model_above_sr-1,aph_m-1,adg_m-1,"
                "apg_m-1,bbp_m-1,aw_m-1,bbw_m-1"), comments="")
        messagebox.showinfo("Exported", f"Wrote:\n{p}\n{spec}")

    # -- plotting -------------------------------------------------------------
    def plot_input(self):
        for ax in (self.ax_rrs, self.ax_a, self.ax_bb):
            ax.clear()
        self.ax_rrs.plot(self.wl, self.rrs, "k-", lw=1, label="R_rs measured")
        self.ax_rrs.axhline(0, color="0.7", lw=0.6)
        self.ax_rrs.set_ylabel("$R_{rs}$ (sr$^{-1}$)")
        self.ax_rrs.set_xlabel("wavelength (nm)")
        self.ax_rrs.legend(fontsize=8)
        self.ax_rrs.set_title(self.source, fontsize=9)
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def plot_result(self):
        r = self.result
        for ax in (self.ax_rrs, self.ax_a, self.ax_bb):
            ax.clear()

        self.ax_rrs.plot(r.wl, r.rrs_obs, "k-", lw=1.4, label="measured (above water)")
        self.ax_rrs.plot(r.wl, r.rrs_model_above, "b--", lw=1.2,
                         label="GIOP model (above water)")
        self.ax_rrs.axhline(0, color="0.7", lw=0.6)
        self.ax_rrs.set_ylabel("$R_{rs}$ (sr$^{-1}$)")
        self.ax_rrs.legend(fontsize=8)
        self.ax_rrs.set_title(
            f"{self.source}   |   both curves above water "
            f"(upstream plots the subsurface model here)", fontsize=8)

        self.ax_a.plot(r.wl, r.aw, color="0.5", lw=1, label="$a_w$ (fixed)")
        self.ax_a.plot(r.wl, r.aph, "g-", lw=1.4, label="$a_\\varphi$")
        self.ax_a.plot(r.wl, r.adg, "c-", lw=1.4, label="$a_{dg}$")
        self.ax_a.plot(r.wl, r.apg, "m:", lw=1.2, label="$a_{pg}$ (better constrained)")
        self.ax_a.set_ylabel("$a$ (m$^{-1}$)")
        self.ax_a.legend(fontsize=8)

        self.ax_bb.plot(r.wl, r.bbw, color="0.5", lw=1, label="$b_{bw}$ (fixed)")
        self.ax_bb.plot(r.wl, r.bbp, "r-", lw=1.4, label=f"$b_{{bp}}$, $\\eta$={r.eta:.2f}")
        self.ax_bb.set_ylabel("$b_b$ (m$^{-1}$)")
        self.ax_bb.set_xlabel("wavelength (nm)")
        self.ax_bb.legend(fontsize=8)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def _error(self, exc):
        messagebox.showerror(type(exc).__name__, f"{exc}\n\n{traceback.format_exc()}")
        self.status.set(f"{type(exc).__name__}: {exc}")


def _num_or_str(value):
    try:
        return float(value)
    except ValueError:
        return value


def main():
    GiopApp().mainloop()


if __name__ == "__main__":
    main()
