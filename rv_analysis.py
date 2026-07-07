#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rv_analysis.py — Radial Velocity (RV) measurement from stellar spectra
======================================================================

Two independent methods in one tool:

1. CCF  (Cross-Correlation Function)
   Between the Lines 2024 workshop (E. Sedaghati) approach: the observed
   spectrum is cross-correlated with a Doppler-shifted template over an RV
   grid; a Gaussian fit to the CCF peak gives the RV. Fast and robust for
   single stars (SB1).

2. BF   (Broadening Function, Rucinski 1992, 2002)
   Method of MAK_Tez Sections 2.2 and 3.2: the observed spectrum is
   modelled as the convolution of the template with a velocity-space
   broadening function,  S(v) = B(v) * T(v),  solved linearly via
   Singular Value Decomposition (SVD). Because the method is linear, the
   components of binary/multiple systems (SB2) appear as independent sharp
   peaks: peak centres give the RVs, widths the rotation (vsini), areas
   the light contributions. Single or double Gaussians are fitted to the
   BF profile.

Modes of operation
------------------
# 1) INTERACTIVE (recommended): run without arguments.
#    A minimal tkinter widget opens: pick the normalized observed spectrum
#    and the synthetic template, choose CCF or BF, press Run. The fit plot
#    is embedded in the window. Without tkinter/display, a terminal wizard
#    with the same steps starts instead.
python rv_analysis.py

# 2) Command line (for scripting / time series):
python rv_analysis.py ccf --spectrum spec.fits --format s2d \
    --teff 6628 --logg 4.251 --feh 0.17 --rv-min -20 --rv-max 100
python rv_analysis.py bf --spectrum spec.obs --template synth.prf \
    --vel-range 500 --components 2 --wave-min 5000 --wave-max 5500

# 3) Self-test on synthetic data (no input files needed):
python rv_analysis.py demo --plot demo.png

Outputs
-------
After every analysis (unless overridden):
  result_CCF.txt / result_BF.txt  — numerical results
  result_CCF.png / result_BF.png  — fit figure

Supported inputs
----------------
- Normalized observed spectrum:
  * any ASCII table (.txt, .dat, .ascii, .obs, ...): first two numeric
    columns are used as wavelength [A] and flux; comment lines (# ; ! %),
    header lines and '-' placeholders are skipped automatically
  * ESPRESSO S2D FITS (multi-order echelle: flux ext=1, wavelength ext=4)
- Synthetic template spectrum:
  * synth3 / SynthV style .prf files, .obs files, or any ASCII table
    (same tolerant reader)
  * or a PHOENIX model downloaded via `expecto` (--teff/--logg/--feh)

Example data (like BinMag): see examples/example_observed.obs and
examples/example_synthetic.prf in this repository.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Speed of light [km/s]
try:
    import astropy.constants as const
    C_KMS = const.c.value / 1000.0
except ImportError:
    C_KMS = 299792.458


# ----------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------

def doppler_shift(wavelength, rv_kms):
    """Shift rest-frame wavelengths to the observed frame for RV [km/s]."""
    return wavelength * (1.0 + rv_kms / C_KMS)


def gauss(x, amp, x0, sigma, offset):
    """Single Gaussian: a*exp(-(x-x0)^2 / 2 sigma^2) + y0"""
    return amp * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2)) + offset


def two_gauss(x, a1, x1, s1, a2, x2, s2, offset):
    """Double Gaussian (two components of an SB2 system)."""
    return (a1 * np.exp(-((x - x1) ** 2) / (2.0 * s1 ** 2))
            + a2 * np.exp(-((x - x2) ** 2) / (2.0 * s2 ** 2))
            + offset)


def read_ascii_spectrum(path):
    """Tolerant ASCII spectrum reader (.txt/.dat/.ascii/.obs/.prf/...).

    Gerçek veri dosyaları çoğu zaman np.loadtxt'nin kaldıramadığı şeyler
    içerir: başlık satırları, '-' gibi yer tutucular, D-üslü Fortran sayıları,
    değişken sütun sayısı. Bu okuyucu:
      - '#', ';', '!', '%' ile başlayan yorum satırlarını atlar,
      - her satırı sayılara çevirir; çevrilemeyen belirteçler ('-' gibi)
        NaN sayılır,
      - en yaygın (modal) sütun sayısına uymayan satırları başlık kabul
        edip eler (SynthV .prf başlık satırı gibi),
      - ilk iki sayısal sütunu (dalgaboyu, akı) alır; NaN'li satırları atar,
      - dalgaboyuna göre sıralar.
    """
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line[0] in "#;!%":
                continue
            vals = []
            for tok in line.replace(",", " ").split():
                try:
                    vals.append(float(tok.replace("D", "E").replace("d", "e")))
                except ValueError:
                    vals.append(np.nan)  # '-' and other placeholders
            rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: no data lines found.")

    ncols = Counter(len(r) for r in rows).most_common(1)[0][0]
    if ncols < 2:
        raise ValueError(f"{path}: need at least two columns "
                         "(wavelength, flux).")
    data = np.array([r[:2] for r in rows if len(r) == ncols], dtype=float)
    data = data[np.isfinite(data).all(axis=1)]
    if data.shape[0] < 5:
        raise ValueError(f"{path}: fewer than 5 usable data rows "
                         "(check the file format).")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def read_spectrum(path, fmt="auto"):
    """Read a spectrum. Returns a list of (wavelength, flux) pairs
    (one pair per echelle order for S2D files).

    fmt: 'auto' | 's2d' | 'text'
    """
    if fmt == "auto":
        fmt = "s2d" if path.lower().endswith((".fits", ".fit", ".fits.gz")) else "text"

    if fmt == "s2d":
        from astropy.io import fits
        with fits.open(path) as hdu:
            flux = np.array(hdu[1].data, dtype=float)
            wvl = np.array(hdu[4].data, dtype=float)
        if flux.ndim == 1:
            return [(wvl, flux)]
        return [(w, f) for w, f in zip(wvl, flux)]

    wl, fx = read_ascii_spectrum(path)
    return [(wl, fx)]


def load_template(args):
    """Load the synthetic/template spectrum from file or via expecto/PHOENIX."""
    if args.template:
        return read_ascii_spectrum(args.template)
    if args.teff is not None:
        try:
            from expecto import get_spectrum
        except ImportError:
            sys.exit("No template file given and 'expecto' is not installed.\n"
                     "Install it with: pip install expecto "
                     "(or use --template FILE)")
        tpl = get_spectrum(T_eff=args.teff, log_g=args.logg, Z=args.feh, cache=True)
        return tpl.wavelength.value, tpl.flux.value
    raise ValueError("A template is required: give a template file "
                     "or T_eff/log g/[Fe/H].")


def barycentric_correction(ra_deg, dec_deg, obstime_isot, site):
    """Barycentric velocity correction [km/s]; ADD it to the measured RV."""
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy.time import Time
    import astropy.units as u

    loc = EarthLocation.of_site(site)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    t = Time(obstime_isot, format="isot", scale="utc")
    return coord.radial_velocity_correction(obstime=t, location=loc).to(u.km / u.s).value


# ----------------------------------------------------------------------
# 1) CCF method (compiled from the workshop notebook)
# ----------------------------------------------------------------------

def calculate_ccf(spec_wl, spec_flux, tpl_wl, tpl_flux, rv_grid):
    """CCF for a single spectrum (or echelle order).

    For each RV the template is Doppler shifted, resampled onto the data
    wavelength grid (np.interp) and dot-multiplied with the data — note the
    workshop warning that the resampling step is essential.

    One change w.r.t. the workshop code: both signals are mean-subtracted
    before the dot product. The continuum contribution is nearly
    shift-independent and, if kept, can swamp the line signal of normalized
    spectra and displace the peak; after mean subtraction the CCF is a true
    cross-covariance peaking where the lines align.
    """
    spec = spec_flux - np.mean(spec_flux)
    ccf = np.empty(rv_grid.size)
    for i, rv in enumerate(rv_grid):
        shifted_wl = doppler_shift(tpl_wl, rv)
        model = np.interp(spec_wl, shifted_wl, tpl_flux)
        ccf[i] = np.dot(spec, model - np.mean(model))
    return ccf


def run_ccf(spectrum_orders, tpl_wl, tpl_flux, rv_min, rv_max, rv_step):
    """Run the CCF over all echelle orders, sum, and fit a Gaussian.

    Returns: dict(rv, rv_err, rv_grid, ccf_total, ccf_orders, popt)
    """
    rv_grid = np.arange(rv_min, rv_max + rv_step, rv_step)
    tpl_norm = tpl_flux / np.nanmax(tpl_flux)

    ccf_orders = []
    for k, (wl, fx) in enumerate(spectrum_orders):
        good = np.isfinite(wl) & np.isfinite(fx)
        wl, fx = wl[good], fx[good]
        if wl.size < 10:
            continue
        # skip orders not covered by the template
        if wl.min() < tpl_wl.min() or wl.max() > tpl_wl.max():
            continue
        peak = np.nanmax(fx)
        if peak <= 0:
            continue
        ccf_orders.append(calculate_ccf(wl, fx / peak, tpl_wl, tpl_norm, rv_grid))
        print(f"  order {k + 1}/{len(spectrum_orders)} done", end="\r")
    print()
    if not ccf_orders:
        raise RuntimeError("No CCF could be computed for any order "
                           "(check the wavelength overlap with the template).")

    ccf_total = np.sum(ccf_orders, axis=0)
    ccf_total = ccf_total / np.nanmax(ccf_total)

    # Gaussian fit, initial guess from the peak position
    x0 = rv_grid[np.argmax(ccf_total)]
    amp0 = ccf_total.max() - np.median(ccf_total)
    p0 = [amp0, x0, 5.0, np.median(ccf_total)]
    popt, pcov = curve_fit(gauss, rv_grid, ccf_total, p0=p0, maxfev=20000)
    rv, rv_err = popt[1], float(np.sqrt(pcov[1, 1]))

    return dict(rv=rv, rv_err=rv_err, rv_grid=rv_grid,
                ccf_total=ccf_total, ccf_orders=np.array(ccf_orders), popt=popt)


# ----------------------------------------------------------------------
# 2) BF method (thesis Sections 2.2/3.2: Rucinski's SVD approach)
# ----------------------------------------------------------------------

def log_wave_grid(wl_min, wl_max, dv_kms):
    """Wavelength grid with constant velocity step (log-lambda spacing).

    In log-lambda space a constant step equals a constant velocity step, so
    a Doppler shift becomes a simple pixel shift — required by the BF
    convolution model.
    """
    step = np.log(1.0 + dv_kms / C_KMS)
    n = int(np.floor(np.log(wl_max / wl_min) / step)) + 1
    return wl_min * np.exp(step * np.arange(n))


def compute_bf(spec_wl, spec_flux, tpl_wl, tpl_flux,
               vel_range=400.0, dv=None, svd_rcond=1e-3, smooth_kms=None):
    """Solve for the Broadening Function via SVD.

    Model:  s = A . b  where the columns of A are pixel-shifted copies of
    the template in velocity space (design/Toeplitz matrix). Small singular
    values are truncated (svd_rcond) to suppress noise — the 'Singular
    Value Decomposition' step described in the thesis.

    Parameters
    ----------
    vel_range : half-width of the BF window [km/s] (scan is +-vel_range)
    dv        : velocity step [km/s]; None -> from the median data pixel
    svd_rcond : singular values with s_i < rcond*s_max are discarded
    smooth_kms: FWHM of the Gaussian smoothing applied to the BF [km/s]
                (None -> 3*dv, the mild smoothing suggested by Rucinski)

    Returns: dict(velocity, bf, bf_smooth, dv, n_kept_sv, n_sv)
    """
    # Convert normalized spectra to line-depth space (1 - flux): continuum
    # ~0 and lines positive, so the BF peaks come out positive.
    good_s = np.isfinite(spec_wl) & np.isfinite(spec_flux)
    good_t = np.isfinite(tpl_wl) & np.isfinite(tpl_flux)
    spec_wl, spec_flux = spec_wl[good_s], spec_flux[good_s]
    tpl_wl, tpl_flux = tpl_wl[good_t], tpl_flux[good_t]

    wl_min = max(spec_wl.min(), tpl_wl.min())
    wl_max = min(spec_wl.max(), tpl_wl.max())
    if wl_max <= wl_min:
        raise ValueError("The spectrum and template wavelength ranges "
                         "do not overlap.")

    if dv is None:
        pix = np.median(np.diff(spec_wl)) / np.median(spec_wl) * C_KMS
        dv = max(pix, 0.5)

    # Leave a margin of one BF window so shifted templates stay in range
    margin = 1.5 * vel_range / C_KMS
    grid = log_wave_grid(wl_min * (1 + margin), wl_max * (1 - margin), dv)

    s = 1.0 - np.interp(grid, spec_wl, spec_flux)   # observed (line depth)
    t = 1.0 - np.interp(grid, tpl_wl, tpl_flux)      # template

    m = grid.size
    half = int(np.ceil(vel_range / dv))
    nbf = 2 * half + 1
    if m <= 2 * nbf:
        raise ValueError("Spectrum segment too short for the BF window; "
                         "widen the wavelength range or reduce vel-range.")

    # Design matrix: row k -> s[k+half] = sum_j b[j] * t[k+half-(j-half)]
    # (positive RV = redshift = shift towards larger pixel index)
    nrow = m - nbf + 1
    idx = (np.arange(nrow)[:, None] + (nbf - 1) - np.arange(nbf)[None, :])
    A = t[idx]
    rhs = s[half:m - half]

    # SVD solution with truncation (regularization)
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    keep = sv > svd_rcond * sv[0]
    inv_sv = np.where(keep, 1.0 / np.where(keep, sv, 1.0), 0.0)
    b = Vt.T @ (inv_sv * (U.T @ rhs))

    velocity = (np.arange(nbf) - half) * dv

    if smooth_kms is None:
        smooth_kms = 3.0 * dv
    sigma_pix = smooth_kms / (2.35482 * dv)
    bf_smooth = gaussian_filter1d(b, sigma_pix)

    return dict(velocity=velocity, bf=b, bf_smooth=bf_smooth, dv=dv,
                n_kept_sv=int(keep.sum()), n_sv=sv.size)


def fit_bf_peaks(velocity, bf, components=1, min_sep=30.0):
    """Fit single/double Gaussians to the BF profile
    (thesis: 'Gaussian fits to the BFs').

    For components=2 the initial guesses are the two highest peaks
    separated by at least min_sep [km/s].

    Returns: list of dict(rv, rv_err, amp, sigma) per component, and popt.
    """
    offset0 = np.median(bf)
    if components == 1:
        i0 = np.argmax(bf)
        p0 = [bf[i0] - offset0, velocity[i0], 20.0, offset0]
        popt, pcov = curve_fit(gauss, velocity, bf, p0=p0, maxfev=20000)
        err = np.sqrt(np.diag(pcov))
        return [dict(rv=popt[1], rv_err=float(err[1]),
                     amp=popt[0], sigma=abs(popt[2]))], popt

    # find two separated peaks
    i1 = int(np.argmax(bf))
    mask = np.abs(velocity - velocity[i1]) > min_sep
    if not mask.any():
        raise RuntimeError("Not enough velocity range for a second peak; "
                           "reduce --min-sep.")
    i2 = int(np.flatnonzero(mask)[np.argmax(bf[mask])])

    p0 = [bf[i1] - offset0, velocity[i1], 20.0,
          bf[i2] - offset0, velocity[i2], 20.0, offset0]
    popt, pcov = curve_fit(two_gauss, velocity, bf, p0=p0, maxfev=40000)
    err = np.sqrt(np.diag(pcov))
    comps = [dict(rv=popt[1], rv_err=float(err[1]), amp=popt[0], sigma=abs(popt[2])),
             dict(rv=popt[4], rv_err=float(err[4]), amp=popt[3], sigma=abs(popt[5]))]
    comps.sort(key=lambda c: c["rv"])          # more negative RV first
    return comps, popt


# ----------------------------------------------------------------------
# Figures (OO Figure API: the same object is saved to disk headlessly and
# embedded into the GUI without backend conflicts)
# ----------------------------------------------------------------------

def make_ccf_figure(result):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 6))
    ax0, ax1 = fig.subplots(2, 1, sharex=True)

    for ccf in result["ccf_orders"]:
        ax0.plot(result["rv_grid"], ccf, lw=0.6, alpha=0.5)
    ax0.set_ylabel("CCF (per order)")

    ax1.plot(result["rv_grid"], result["ccf_total"], "k-", lw=1.2,
             label="Total CCF (normalized)")
    ax1.plot(result["rv_grid"], gauss(result["rv_grid"], *result["popt"]),
             "r-", lw=2, alpha=0.7,
             label=f"Gaussian fit: RV = {result['rv']:.3f} "
                   f"± {result['rv_err']:.3f} km/s")
    ax1.axvline(result["rv"], color="r", ls=":", lw=1)
    ax1.set_xlabel("RV [km/s]")
    ax1.set_ylabel("Normalized CCF")
    ax1.legend()
    fig.tight_layout()
    return fig


def make_bf_figure(bf_result, comps, popt, components):
    from matplotlib.figure import Figure
    v = bf_result["velocity"]
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    ax.plot(v, bf_result["bf"], color="0.7", lw=0.8, label="BF (raw)")
    ax.plot(v, bf_result["bf_smooth"], "b-", lw=1.5, label="BF (smoothed)")
    model = gauss(v, *popt) if components == 1 else two_gauss(v, *popt)
    ax.plot(v, model, "r--", lw=2, alpha=0.8, label="Gaussian fit")
    for i, c in enumerate(comps, 1):
        ax.axvline(c["rv"], color="r", ls=":", lw=1)
        ax.annotate(f"C{i}: {c['rv']:.2f} km/s", (c["rv"], c["amp"]),
                    textcoords="offset points", xytext=(6, 6), color="r")
    ax.set_xlabel("Radial velocity [km/s]")
    ax.set_ylabel("Broadening Function")
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig, outfile):
    fig.savefig(outfile, dpi=150)
    print(f"Figure saved: {outfile}")


# ----------------------------------------------------------------------
# Analysis commands — used by both the CLI and the interactive modes.
# Results are always written to result_CCF/result_BF (txt + png).
# ----------------------------------------------------------------------

def cmd_ccf(args):
    orders = read_spectrum(args.spectrum, args.format)
    tpl_wl, tpl_flux = load_template(args)
    if args.wave_min or args.wave_max:
        lo = args.wave_min or -np.inf
        hi = args.wave_max or np.inf
        orders = [(w[(w >= lo) & (w <= hi)], f[(w >= lo) & (w <= hi)])
                  for w, f in orders]
        orders = [(w, f) for w, f in orders if w.size > 10]

    print(f"Computing the CCF over {len(orders)} order(s)/segment(s)...")
    result = run_ccf(orders, tpl_wl, tpl_flux, args.rv_min, args.rv_max, args.rv_step)

    vbary = 0.0
    if args.ra is not None:
        vbary = barycentric_correction(args.ra, args.dec, args.obstime, args.site)
        print(f"Barycentric correction: {vbary:+.4f} km/s")

    rv = result["rv"] + vbary
    tpl_name = args.template or f"PHOENIX T={args.teff}K"
    summary = ("================ CCF RESULT ================\n"
               f"Normalized spectrum : {args.spectrum}\n"
               f"Synthetic template  : {tpl_name}\n"
               f"RV = {rv:.4f} ± {result['rv_err']:.4f} km/s"
               + ("  (barycentric corrected)\n" if vbary else "\n")
               + "============================================")
    print("\n" + summary)

    outfile = args.output or "result_CCF.txt"
    with open(outfile, "w") as f:
        f.write(f"# Normalized spectrum : {args.spectrum}\n")
        f.write(f"# Synthetic template  : {tpl_name}\n")
        f.write("# method  RV[km/s]  RV_err[km/s]  bary_corr[km/s]\n")
        f.write(f"CCF  {rv:.5f}  {result['rv_err']:.5f}  {vbary:.5f}\n")
    print(f"Results written: {outfile}")

    plotfile = args.plot or "result_CCF.png"
    fig = make_ccf_figure(result)
    save_figure(fig, plotfile)

    return dict(method="CCF", fig=fig, text=summary,
                output=outfile, plot=plotfile)


def cmd_bf(args):
    orders = read_spectrum(args.spectrum, args.format)
    if len(orders) > 1:
        # merge echelle orders into one array (the 'combine' step of the thesis)
        wl = np.concatenate([w for w, _ in orders])
        fx = np.concatenate([f for _, f in orders])
        order_ = np.argsort(wl)
        spec_wl, spec_flux = wl[order_], fx[order_]
    else:
        spec_wl, spec_flux = orders[0]

    if args.wave_min or args.wave_max:
        lo = args.wave_min or -np.inf
        hi = args.wave_max or np.inf
        sel = (spec_wl >= lo) & (spec_wl <= hi)
        spec_wl, spec_flux = spec_wl[sel], spec_flux[sel]

    tpl_wl, tpl_flux = load_template(args)

    print("Solving the BF via SVD...")
    bf_result = compute_bf(spec_wl, spec_flux, tpl_wl, tpl_flux,
                           vel_range=args.vel_range, dv=args.dv,
                           svd_rcond=args.svd_rcond, smooth_kms=args.smooth)
    print(f"  velocity step dv = {bf_result['dv']:.3f} km/s, "
          f"singular values kept: {bf_result['n_kept_sv']}/{bf_result['n_sv']}")

    comps, popt = fit_bf_peaks(bf_result["velocity"], bf_result["bf_smooth"],
                               components=args.components, min_sep=args.min_sep)

    vbary = 0.0
    if args.ra is not None:
        vbary = barycentric_correction(args.ra, args.dec, args.obstime, args.site)
        print(f"Barycentric correction: {vbary:+.4f} km/s")

    tpl_name = args.template or f"PHOENIX T={args.teff}K"
    lines = ["================ BF RESULT =================",
             f"Normalized spectrum : {args.spectrum}",
             f"Synthetic template  : {tpl_name}"]
    for i, c in enumerate(comps, 1):
        lines.append(f"Component {i}: RV = {c['rv'] + vbary:.4f} "
                     f"± {c['rv_err']:.4f} km/s"
                     f"   (amp={c['amp']:.4f}, sigma={c['sigma']:.2f} km/s)")
    if len(comps) == 2 and (comps[0]["amp"] > 0) and (comps[1]["amp"] > 0):
        # BF area ratio ~ light ratio; approximated by amp*sigma
        l2_l1 = (comps[1]["amp"] * comps[1]["sigma"]) / \
                (comps[0]["amp"] * comps[0]["sigma"])
        lines.append(f"Light ratio (C2/C1, from BF areas) ≈ {l2_l1:.3f}")
    lines.append("============================================")
    summary = "\n".join(lines)
    print("\n" + summary)

    outfile = args.output or "result_BF.txt"
    with open(outfile, "w") as f:
        f.write(f"# Normalized spectrum : {args.spectrum}\n")
        f.write(f"# Synthetic template  : {tpl_name}\n")
        f.write("# component  RV[km/s]  RV_err[km/s]  amp  sigma[km/s]  "
                "bary_corr[km/s]\n")
        for i, c in enumerate(comps, 1):
            f.write(f"{i}  {c['rv'] + vbary:.5f}  {c['rv_err']:.5f}  "
                    f"{c['amp']:.5f}  {c['sigma']:.5f}  {vbary:.5f}\n")
    print(f"Results written: {outfile}")

    plotfile = args.plot or "result_BF.png"
    fig = make_bf_figure(bf_result, comps, popt, args.components)
    save_figure(fig, plotfile)

    return dict(method="BF", fig=fig, text=summary,
                output=outfile, plot=plotfile)


def cmd_demo(args):
    """Generate a synthetic SB2 spectrum and solve it with both CCF and BF.

    The true RVs are known, so the behavioural difference of the two
    methods is measured directly: the BF shows sharp separated peaks while
    the CCF peaks are broader and prone to blending.
    """
    rng = np.random.default_rng(42)
    rv1_true, rv2_true = -80.0, 120.0     # component velocities [km/s]
    light_ratio = 0.45                    # C2/C1 light contribution

    # Template: normalized spectrum with random absorption lines, 5000-5500 A
    wl = log_wave_grid(4950.0, 5550.0, 1.0)
    tpl = np.ones_like(wl)
    n_lines = 160
    centers = rng.uniform(4960, 5540, n_lines)
    depths = rng.uniform(0.15, 0.85, n_lines)
    widths = rng.uniform(0.08, 0.25, n_lines)  # A
    for c0, d, w in zip(centers, depths, widths):
        tpl -= d * np.exp(-((wl - c0) ** 2) / (2 * w ** 2))
    tpl = np.clip(tpl, 0.02, None)

    # Observed: two Doppler-shifted, rotationally broadened components
    def rot_broaden(flux, vsini_kms):
        sigma_pix = vsini_kms / 1.0 / 2.35482
        return gaussian_filter1d(flux, sigma_pix)

    f1 = np.interp(wl, doppler_shift(wl, rv1_true), tpl)
    f2 = np.interp(wl, doppler_shift(wl, rv2_true), tpl)
    f1 = rot_broaden(f1, 40.0)
    f2 = rot_broaden(f2, 25.0)
    obs = (f1 + light_ratio * f2) / (1.0 + light_ratio)
    obs += rng.normal(0, 0.004, obs.size)   # S/N ~ 250

    print("Synthetic SB2 system generated:")
    print(f"  True RV1 = {rv1_true} km/s, RV2 = {rv2_true} km/s, "
          f"light ratio = {light_ratio}\n")

    # --- BF ---
    print("--- BF method ---")
    bf_result = compute_bf(wl, obs, wl, tpl, vel_range=300.0, dv=2.0,
                           svd_rcond=5e-4, smooth_kms=10.0)
    comps, popt = fit_bf_peaks(bf_result["velocity"], bf_result["bf_smooth"],
                               components=2, min_sep=50.0)
    for i, c in enumerate(comps, 1):
        print(f"  Component {i}: RV = {c['rv']:8.3f} ± {c['rv_err']:.3f} km/s")

    # --- CCF ---
    print("--- CCF method (same data) ---")
    rv_grid = np.arange(-200, 251, 2.0)
    ccf = calculate_ccf(wl, 1.0 - obs, wl, 1.0 - tpl, rv_grid)
    ccf /= ccf.max()
    ccomp, cpopt = fit_bf_peaks(rv_grid, ccf, components=2, min_sep=50.0)
    for i, c in enumerate(ccomp, 1):
        print(f"  Component {i}: RV = {c['rv']:8.3f} ± {c['rv_err']:.3f} km/s")

    print("\nComparison with the true values (km/s):")
    print(f"  BF : dRV1 = {comps[0]['rv'] - rv1_true:+.3f}, "
          f"dRV2 = {comps[1]['rv'] - rv2_true:+.3f}")
    print(f"  CCF: dRV1 = {ccomp[0]['rv'] - rv1_true:+.3f}, "
          f"dRV2 = {ccomp[1]['rv'] - rv2_true:+.3f}")

    if args.plot:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(11, 11))
        axes = fig.subplots(3, 1)
        sel = (wl > 5195) & (wl < 5235)
        axes[0].plot(wl[sel], obs[sel], "k-", lw=0.8,
                     label="Observed (synthetic SB2)")
        axes[0].plot(wl[sel], tpl[sel], "C0-", lw=0.8, alpha=0.6,
                     label="Template")
        axes[0].set_xlabel("Wavelength [Å]")
        axes[0].set_ylabel("Normalized flux")
        axes[0].legend()
        axes[0].set_title("Synthetic SB2 spectrum (section)")

        v = bf_result["velocity"]
        axes[1].plot(v, bf_result["bf_smooth"], "b-", lw=1.5, label="BF")
        axes[1].plot(v, two_gauss(v, *popt), "r--", lw=1.5,
                     label="Double Gaussian fit")
        for rvt in (rv1_true, rv2_true):
            axes[1].axvline(rvt, color="0.5", ls=":", lw=1)
        axes[1].set_ylabel("BF")
        axes[1].set_title("Broadening Function: sharp, separated components")
        axes[1].legend()

        axes[2].plot(rv_grid, ccf, "k-", lw=1.5, label="CCF")
        axes[2].plot(rv_grid, two_gauss(rv_grid, *cpopt), "r--", lw=1.5,
                     label="Double Gaussian fit")
        for rvt in (rv1_true, rv2_true):
            axes[2].axvline(rvt, color="0.5", ls=":", lw=1)
        axes[2].set_xlabel("Radial velocity [km/s]")
        axes[2].set_ylabel("Normalized CCF")
        axes[2].set_title("CCF: broader peaks, prone to blending")
        axes[2].legend()

        fig.tight_layout()
        save_figure(fig, args.plot)


# ----------------------------------------------------------------------
# Interactive mode — started when run without arguments.
# The tkinter widget is tried first; without tkinter/display the terminal
# wizard runs instead. Both call the same cmd_ccf/cmd_bf core, so the
# outputs (result_*.txt + result_*.png) are identical.
# ----------------------------------------------------------------------

def make_args(**overrides):
    """Namespace with every field expected by cmd_ccf/cmd_bf."""
    base = dict(spectrum=None, format="auto", template=None,
                teff=None, logg=4.5, feh=0.0,
                wave_min=None, wave_max=None,
                ra=None, dec=None, obstime=None, site="paranal",
                plot=None, output=None,
                rv_min=-200.0, rv_max=200.0, rv_step=0.5,
                vel_range=400.0, dv=None, svd_rcond=1e-3, smooth=None,
                components=1, min_sep=30.0)
    base.update(overrides)
    return argparse.Namespace(**base)


def ask(prompt, default=None, cast=str, validate=None, allow_empty=False):
    """Single question for the terminal wizard: default value + validation."""
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            if default is not None:
                return default
            if allow_empty:
                return None
            print("  This field cannot be empty.")
            continue
        try:
            val = cast(raw)
        except (TypeError, ValueError):
            print("  Invalid value, try again.")
            continue
        if validate is not None and not validate(val):
            print("  Invalid choice, try again.")
            continue
        return val


def _ask_common_inputs():
    """Inputs shared by both methods: files and wavelength range."""
    spectrum = ask("Normalized spectrum file (ASCII or S2D FITS)",
                   cast=str, validate=os.path.isfile)
    template = ask("Synthetic template file (.prf/.obs/ASCII; "
                   "empty: download PHOENIX)",
                   allow_empty=True,
                   validate=lambda p: p is None or os.path.isfile(p))
    kw = dict(spectrum=spectrum, template=template)
    if template is None:
        kw["teff"] = ask("  Template T_eff [K]", cast=float)
        kw["logg"] = ask("  Template log g", default=4.5, cast=float)
        kw["feh"] = ask("  Template [Fe/H]", default=0.0, cast=float)
    kw["wave_min"] = ask("Minimum wavelength [A] (empty: all)",
                         allow_empty=True, cast=float)
    kw["wave_max"] = ask("Maximum wavelength [A] (empty: all)",
                         allow_empty=True, cast=float)
    return kw


def _ask_barycentric(kw):
    yanit = ask("Apply barycentric correction? (y/n)", default="n",
                cast=str, validate=lambda s: s.lower() in ("y", "n"))
    if yanit.lower() == "y":
        kw["ra"] = ask("  RA [deg]", cast=float)
        kw["dec"] = ask("  Dec [deg]", cast=float)
        kw["obstime"] = ask("  Observation time (ISOT, e.g. 2024-12-03T02:30:00)")
        kw["site"] = ask("  Observatory (astropy site name: tug, paranal, ...)",
                         default="tug")
    return kw


def run_terminal_wizard():
    """Step-by-step RV analysis in the terminal (when the GUI cannot open)."""
    print("=" * 60)
    print(" RV ANALYSIS — interactive terminal mode")
    print("=" * 60)

    kw = _ask_common_inputs()

    print("\nWhich method do you want to continue with?")
    print("  [1] CCF — cross-correlation (practical for single stars / SB1)")
    print("  [2] BF  — broadening function (recommended for binaries / SB2)")
    secim = ask("Choice", default="2", cast=str,
                validate=lambda s: s in ("1", "2"))

    if secim == "1":
        kw["rv_min"] = ask("RV scan lower limit [km/s]", default=-200.0, cast=float)
        kw["rv_max"] = ask("RV scan upper limit [km/s]", default=200.0, cast=float)
        kw["rv_step"] = ask("RV step [km/s]", default=0.5, cast=float)
        kw = _ask_barycentric(kw)
        print()
        cmd_ccf(make_args(**kw))
    else:
        kw["vel_range"] = ask("BF window half-width [km/s]",
                              default=400.0, cast=float)
        kw["components"] = ask("Number of components (SB1=1, SB2=2)", default=2,
                               cast=int, validate=lambda n: n in (1, 2))
        kw["smooth"] = ask("BF smoothing FWHM [km/s] (empty: auto)",
                           allow_empty=True, cast=float)
        kw["svd_rcond"] = ask("SVD cutoff", default=1e-3, cast=float)
        kw = _ask_barycentric(kw)
        print()
        cmd_bf(make_args(**kw))

    print("\nDone. Result files and the fit figure were saved "
          "to the working directory.")


def run_gui():
    """Minimal tkinter widget: file pickers, CCF/BF choice, embedded fit plot.

    The analysis core is shared with the CLI (cmd_ccf/cmd_bf); the result
    text is shown in the window and the files (result_*.txt, result_*.png)
    are written to disk as usual.
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("RV Analysis")

    main = ttk.Frame(root, padding=10)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)

    spec_var = tk.StringVar()
    tpl_var = tk.StringVar()

    def browse(var, title):
        p = filedialog.askopenfilename(
            title=title,
            filetypes=[("Spectrum files",
                        "*.fits *.fit *.txt *.dat *.ascii *.obs *.prf"),
                       ("All files", "*.*")])
        if p:
            var.set(p)

    # --- input files ---
    ttk.Label(main, text="Normalized spectrum:").grid(row=0, column=0, sticky="w")
    ttk.Entry(main, textvariable=spec_var, width=48).grid(row=0, column=1,
                                                          sticky="ew", padx=4)
    ttk.Button(main, text="Browse...",
               command=lambda: browse(spec_var, "Select normalized spectrum")
               ).grid(row=0, column=2)

    ttk.Label(main, text="Synthetic spectrum:").grid(row=1, column=0, sticky="w")
    ttk.Entry(main, textvariable=tpl_var, width=48).grid(row=1, column=1,
                                                         sticky="ew", padx=4)
    ttk.Button(main, text="Browse...",
               command=lambda: browse(tpl_var, "Select synthetic spectrum")
               ).grid(row=1, column=2)

    # --- wavelength range ---
    wmin_var, wmax_var = tk.StringVar(), tk.StringVar()
    wrow = ttk.Frame(main)
    wrow.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Label(wrow, text="Wavelength range [Å]:").pack(side="left")
    ttk.Entry(wrow, textvariable=wmin_var, width=8).pack(side="left", padx=(4, 2))
    ttk.Label(wrow, text="–").pack(side="left")
    ttk.Entry(wrow, textvariable=wmax_var, width=8).pack(side="left", padx=2)
    ttk.Label(wrow, text="(empty: full range)").pack(side="left", padx=4)

    # --- method ---
    method_var = tk.StringVar(value="BF")
    mrow = ttk.Frame(main)
    mrow.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Label(mrow, text="Method:").pack(side="left")

    ccf_frame = ttk.Frame(main)
    bf_frame = ttk.Frame(main)

    def on_method_change():
        if method_var.get() == "CCF":
            bf_frame.grid_remove()
            ccf_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        else:
            ccf_frame.grid_remove()
            bf_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))

    ttk.Radiobutton(mrow, text="CCF (single star)", variable=method_var,
                    value="CCF", command=on_method_change
                    ).pack(side="left", padx=8)
    ttk.Radiobutton(mrow, text="BF (binary, SB2)", variable=method_var,
                    value="BF", command=on_method_change
                    ).pack(side="left", padx=8)

    rvmin_var = tk.StringVar(value="-200")
    rvmax_var = tk.StringVar(value="200")
    rvstep_var = tk.StringVar(value="0.5")
    for j, (lbl, var) in enumerate([("RV min", rvmin_var),
                                    ("RV max", rvmax_var),
                                    ("step", rvstep_var)]):
        ttk.Label(ccf_frame, text=lbl).grid(row=0, column=2 * j, sticky="w")
        ttk.Entry(ccf_frame, textvariable=var, width=8
                  ).grid(row=0, column=2 * j + 1, padx=(2, 10))
    ttk.Label(ccf_frame, text="[km/s]").grid(row=0, column=6)

    vrange_var = tk.StringVar(value="400")
    comp_var = tk.IntVar(value=2)
    ttk.Label(bf_frame, text="Velocity window ±").grid(row=0, column=0, sticky="w")
    ttk.Entry(bf_frame, textvariable=vrange_var, width=8
              ).grid(row=0, column=1, padx=2)
    ttk.Label(bf_frame, text="km/s").grid(row=0, column=2, padx=(0, 12))
    ttk.Label(bf_frame, text="Components").grid(row=0, column=3)
    ttk.Combobox(bf_frame, textvariable=comp_var, values=[1, 2], width=3,
                 state="readonly").grid(row=0, column=4, padx=4)
    on_method_change()

    # --- run + results + plot ---
    result_text = tk.Text(main, height=7, width=90, state="disabled",
                          font=("Courier", 10))
    result_text.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)

    plot_frame = ttk.Frame(main)
    plot_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
    main.rowconfigure(7, weight=1)
    canvas_holder = {"canvas": None}

    def _f(var, default=None):
        s = var.get().strip()
        return float(s) if s else default

    def run_analysis():
        try:
            if not spec_var.get().strip():
                raise ValueError("No normalized spectrum file selected.")
            if not tpl_var.get().strip():
                raise ValueError("No synthetic spectrum file selected.")
            kw = dict(spectrum=spec_var.get().strip(),
                      template=tpl_var.get().strip(),
                      wave_min=_f(wmin_var), wave_max=_f(wmax_var))
            if method_var.get() == "CCF":
                kw.update(rv_min=_f(rvmin_var, -200.0),
                          rv_max=_f(rvmax_var, 200.0),
                          rv_step=_f(rvstep_var, 0.5))
                payload = cmd_ccf(make_args(**kw))
            else:
                kw.update(vel_range=_f(vrange_var, 400.0),
                          components=int(comp_var.get()))
                payload = cmd_bf(make_args(**kw))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        result_text.configure(state="normal")
        result_text.delete("1.0", "end")
        result_text.insert("1.0", payload["text"] + "\n"
                           f"Saved: {payload['output']}, {payload['plot']}")
        result_text.configure(state="disabled")

        if canvas_holder["canvas"] is not None:
            canvas_holder["canvas"].get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(payload["fig"], master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas_holder["canvas"] = canvas

    ttk.Button(main, text="Run", command=run_analysis
               ).grid(row=5, column=0, columnspan=3, pady=6)

    root.mainloop()


def run_interactive():
    """No-argument entry point: widget first, terminal wizard as fallback."""
    try:
        run_gui()
        return
    except Exception as exc:
        print(f"Could not open the GUI ({exc.__class__.__name__}: {exc})")
        print("Falling back to the terminal wizard...\n")
    run_terminal_wizard()


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------

def add_common_args(p):
    p.add_argument("--spectrum", required=True,
                   help="Normalized observed spectrum file")
    p.add_argument("--format", default="auto", choices=["auto", "s2d", "text"],
                   help="Spectrum format (auto: guess from extension)")
    p.add_argument("--template",
                   help="Synthetic template file (.prf/.obs/ASCII)")
    p.add_argument("--teff", type=float, help="Template T_eff [K] (expecto)")
    p.add_argument("--logg", type=float, default=4.5, help="Template log g")
    p.add_argument("--feh", type=float, default=0.0, help="Template [Fe/H]")
    p.add_argument("--wave-min", type=float, help="Minimum wavelength to use [A]")
    p.add_argument("--wave-max", type=float, help="Maximum wavelength to use [A]")
    p.add_argument("--ra", type=float, help="RA [deg] for barycentric correction")
    p.add_argument("--dec", type=float, help="Dec [deg] for barycentric correction")
    p.add_argument("--obstime", help="Observation time (ISOT, e.g. 2024-12-03T02:30:00)")
    p.add_argument("--site", default="paranal",
                   help="Observatory (astropy site name, e.g. paranal, tug)")
    p.add_argument("--plot", help="Figure PNG file name "
                                  "(default: result_CCF.png / result_BF.png)")
    p.add_argument("--output", help="Result text file name "
                                    "(default: result_CCF.txt / result_BF.txt)")


def main():
    # No arguments -> interactive mode (widget or terminal wizard)
    if len(sys.argv) == 1:
        run_interactive()
        return

    parser = argparse.ArgumentParser(
        description="Radial velocity from spectra: CCF and BF methods. "
                    "Run without arguments for the interactive mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gui = sub.add_parser("gui", help="Open the widget (same as no arguments)")
    p_gui.set_defaults(func=lambda a: run_interactive())

    p_ccf = sub.add_parser("ccf", help="Cross-correlation (workshop method)")
    add_common_args(p_ccf)
    p_ccf.add_argument("--rv-min", type=float, default=-200.0,
                       help="RV scan lower limit [km/s]")
    p_ccf.add_argument("--rv-max", type=float, default=200.0,
                       help="RV scan upper limit [km/s]")
    p_ccf.add_argument("--rv-step", type=float, default=0.5, help="RV step [km/s]")
    p_ccf.set_defaults(func=cmd_ccf)

    p_bf = sub.add_parser("bf", help="Broadening Function (thesis method, SVD)")
    add_common_args(p_bf)
    p_bf.add_argument("--vel-range", type=float, default=400.0,
                      help="BF window half-width [km/s]")
    p_bf.add_argument("--dv", type=float,
                      help="Velocity step [km/s] (default: data pixel)")
    p_bf.add_argument("--svd-rcond", type=float, default=1e-3,
                      help="SVD cutoff (small singular value threshold)")
    p_bf.add_argument("--smooth", type=float,
                      help="BF smoothing FWHM [km/s] (default: 3*dv)")
    p_bf.add_argument("--components", type=int, default=1, choices=[1, 2],
                      help="Number of Gaussians to fit: SB1=1, SB2=2")
    p_bf.add_argument("--min-sep", type=float, default=30.0,
                      help="Minimum separation of the two peaks [km/s]")
    p_bf.set_defaults(func=cmd_bf)

    p_demo = sub.add_parser("demo",
                            help="CCF vs BF comparison on synthetic SB2 data")
    p_demo.add_argument("--plot", help="Comparison figure PNG file name")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
