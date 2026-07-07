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


def read_fits_spectrum(path):
    """FITS spectrum reader for the common layouts of raw/pipeline data.

    Supported:
      - ESPRESSO S2D (flux ext=1, wavelength ext=4, 2D echelle orders)
      - phase-3 style binary tables with WAVE/LAMBDA and FLUX columns
        (FEROS, HARPS, UVES, ...)
      - simple 1D image HDU with a linear wavelength WCS (CRVAL1/CDELT1),
        the classic IRAF product
    Returns a list of (wavelength, flux) pairs (one per echelle order).
    """
    from astropy.io import fits
    with fits.open(path) as hdul:
        # 1) ESPRESSO S2D layout
        try:
            flux = np.array(hdul[1].data, dtype=float)
            wvl = np.array(hdul[4].data, dtype=float)
            if flux.ndim == 2 and flux.shape == wvl.shape:
                return [(w, f) for w, f in zip(wvl, flux)]
        except (IndexError, TypeError, ValueError):
            pass
        # 2) binary table with wave/flux columns (FEROS/HARPS phase-3)
        for hdu in hdul:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            cols = {c.name.upper(): c.name for c in hdu.columns}
            wname = next((cols[k] for k in
                          ("WAVE", "WAVELENGTH", "LAMBDA", "AWAV") if k in cols),
                         None)
            fname = next((cols[k] for k in
                          ("FLUX", "FLUX_REDUCED", "INTENSITY") if k in cols),
                         None)
            if wname and fname:
                wl = np.ravel(np.asarray(hdu.data[wname], dtype=float))
                fx = np.ravel(np.asarray(hdu.data[fname], dtype=float))
                order = np.argsort(wl)
                return [(wl[order], fx[order])]
        # 3) 1D image with linear wavelength WCS
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data, dtype=float)
            h = hdu.header
            if data.ndim == 1 and "CRVAL1" in h:
                cdelt = h.get("CDELT1", h.get("CD1_1"))
                if cdelt:
                    crpix = h.get("CRPIX1", 1.0)
                    wl = h["CRVAL1"] + (np.arange(data.size) + 1 - crpix) * cdelt
                    return [(wl, data)]
    raise ValueError(f"{path}: unrecognized FITS layout (expected S2D, a "
                     "WAVE/FLUX binary table, or a 1D image with CRVAL1/CDELT1).")


def read_spectrum(path, fmt="auto"):
    """Read a spectrum. Returns a list of (wavelength, flux) pairs
    (one pair per echelle order for multi-order files).

    fmt: 'auto' | 's2d' | 'text'  ('s2d' accepts any supported FITS layout)
    """
    if fmt == "auto":
        fmt = "s2d" if path.lower().endswith((".fits", ".fit", ".fits.gz")) else "text"
    if fmt == "s2d":
        return read_fits_spectrum(path)
    wl, fx = read_ascii_spectrum(path)
    return [(wl, fx)]


def fits_header_info(path):
    """Extract OBJECT, DATE-OBS, RA, Dec from a FITS primary header.

    Returns dict with keys object/obstime/ra/dec (values may be None).
    RA/Dec are converted to degrees; sexagesimal strings ('hh:mm:ss') are
    interpreted as hourangle/deg.
    """
    from astropy.io import fits
    info = {"object": None, "obstime": None, "ra": None, "dec": None,
            "exptime": None}
    try:
        with fits.open(path) as hdul:
            h = hdul[0].header
    except Exception:
        return info
    info["object"] = h.get("OBJECT")
    info["obstime"] = h.get("DATE-OBS") or h.get("DATE_OBS")
    info["exptime"] = h.get("EXPTIME")
    ra, dec = h.get("RA"), h.get("DEC")
    try:
        if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
            info["ra"], info["dec"] = float(ra), float(dec)
        elif ra is not None and dec is not None:
            from astropy.coordinates import Angle
            import astropy.units as u
            ra_unit = u.hourangle if ":" in str(ra) else u.deg
            info["ra"] = Angle(str(ra), unit=ra_unit).deg
            info["dec"] = Angle(str(dec), unit=u.deg).deg
    except Exception:
        pass
    return info


def resolve_target(name):
    """Resolve a target name to ICRS coordinates [deg] via SIMBAD (Sesame).

    Raises ValueError when the name cannot be resolved (no match or no
    network); the caller should then fall back to manual coordinate entry.
    """
    from astropy.coordinates import SkyCoord
    try:
        c = SkyCoord.from_name(name)
    except Exception as exc:
        raise ValueError(f"SIMBAD lookup failed for '{name}': {exc}") from exc
    return float(c.ra.deg), float(c.dec.deg)


def normalize_continuum(wl, flux, poly_order=5, iterations=8,
                        low_clip=1.0, high_clip=4.0):
    """Iterative continuum normalization of a raw spectrum (FEROS-style).

    'All FEROS spectra were normalised to the continuum level iteratively':
    a polynomial is fitted to the spectrum, then points lying more than
    low_clip*sigma BELOW the fit (absorption lines) or high_clip*sigma
    above it (cosmics/emission) are rejected and the fit is repeated, so
    the polynomial converges onto the upper envelope — the continuum.
    The interactive comparison with a synthetic spectrum happens in the
    widget / normalize command, where the result is overplotted on the
    template so the user can tune poly_order and re-run.

    Returns (normalized_flux, continuum). Pixels where the continuum is
    not positive come back as NaN.
    """
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(wl) & np.isfinite(flux) & (flux > 0)
    if good.sum() < poly_order + 2:
        raise ValueError("Too few valid pixels for continuum fitting.")
    # scale x to [-1, 1] to keep the polynomial fit well conditioned
    x = 2.0 * (wl - wl[good].min()) / (wl[good].max() - wl[good].min()) - 1.0

    mask = good.copy()
    cont = None
    for _ in range(iterations):
        coeff = np.polyfit(x[mask], flux[mask], poly_order)
        cont = np.polyval(coeff, x)
        resid = flux - cont
        sigma = np.std(resid[mask])
        if sigma <= 0:
            break
        new_mask = good & (resid > -low_clip * sigma) & (resid < high_clip * sigma)
        if new_mask.sum() < poly_order + 2 or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(cont > 0, flux / cont, np.nan)
    return norm, cont


def normalize_spectrum_file(path, fmt="auto", poly_order=5, iterations=8,
                            low_clip=1.0, high_clip=4.0):
    """Normalize a raw spectrum file order by order and merge.

    Returns (wl, norm_flux, orders_raw) where orders_raw is the list of
    (wl, flux, continuum) per order for diagnostic plotting.
    """
    orders = read_spectrum(path, fmt)
    normed, diag = [], []
    for wl, fx in orders:
        good = np.isfinite(wl) & np.isfinite(fx)
        wl, fx = wl[good], fx[good]
        if wl.size < poly_order + 2:
            continue
        nf, cont = normalize_continuum(wl, fx, poly_order, iterations,
                                       low_clip, high_clip)
        keep = np.isfinite(nf)
        normed.append((wl[keep], nf[keep]))
        diag.append((wl, fx, cont))
    if not normed:
        raise ValueError(f"{path}: no order could be normalized.")
    wl = np.concatenate([w for w, _ in normed])
    nf = np.concatenate([f for _, f in normed])
    order_ = np.argsort(wl)
    return wl[order_], nf[order_], diag


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


def get_bary_correction(args):
    """Resolve target info and return the barycentric correction [km/s].

    Coordinate priority: explicit --ra/--dec, then SIMBAD lookup of
    --object, then the FITS header of the spectrum. The observation time
    comes from --obstime or the FITS header (DATE-OBS). Returns 0.0 when
    the correction cannot be computed (with a note on what was missing).
    """
    ra, dec = args.ra, args.dec
    obstime = args.obstime

    if ra is None and getattr(args, "object", None):
        try:
            ra, dec = resolve_target(args.object)
            print(f"SIMBAD: '{args.object}' -> RA = {ra:.5f} deg, "
                  f"Dec = {dec:.5f} deg")
        except ValueError as exc:
            print(f"Warning: {exc}\n  -> give coordinates manually "
                  "with --ra/--dec.")

    if (ra is None or obstime is None) and \
            str(args.spectrum).lower().endswith((".fits", ".fit", ".fits.gz")):
        hdr = fits_header_info(args.spectrum)
        if ra is None and hdr["ra"] is not None:
            ra, dec = hdr["ra"], hdr["dec"]
            print(f"FITS header: RA = {ra:.5f} deg, Dec = {dec:.5f} deg")
        if obstime is None and hdr["obstime"]:
            obstime = hdr["obstime"]
            print(f"FITS header: DATE-OBS = {obstime}")

    if ra is None or dec is None or not obstime:
        if getattr(args, "object", None) or args.ra is not None or obstime:
            print("Note: barycentric correction skipped "
                  "(needs coordinates AND observation time).")
        return 0.0

    v = barycentric_correction(ra, dec, obstime, args.site)
    print(f"Barycentric correction: {v:+.4f} km/s (added to the RV)")
    return v


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

    vbary = get_bary_correction(args)

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

    vbary = get_bary_correction(args)

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


def make_norm_figure(diag, wl, nf, tpl=None):
    """Two-panel normalization figure: raw + continuum fits, and the
    normalized spectrum overplotted on the synthetic template (the
    'interactive comparison with a synthetic spectrum' step)."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 6))
    ax0, ax1 = fig.subplots(2, 1, sharex=True)

    for w, f, cont in diag:
        ax0.plot(w, f, lw=0.5, alpha=0.6)
        ax0.plot(w, cont, "r-", lw=1.0, alpha=0.8)
    ax0.set_ylabel("Raw flux")
    ax0.set_title("Raw spectrum and fitted continuum (red)")

    ax1.plot(wl, nf, "k-", lw=0.6, label="Normalized spectrum")
    if tpl is not None:
        tw, tf = tpl
        sel = (tw >= wl.min()) & (tw <= wl.max())
        if sel.any():
            ax1.plot(tw[sel], tf[sel] / np.nanmax(tf[sel]), "C0-", lw=0.8,
                     alpha=0.7, label="Synthetic template")
    ax1.axhline(1.0, color="0.6", ls=":", lw=1)
    ax1.set_ylim(-0.1, 1.6)
    ax1.set_xlabel("Wavelength [Å]")
    ax1.set_ylabel("Normalized flux")
    ax1.legend()
    fig.tight_layout()
    return fig


def cmd_normalize(args):
    """Normalize a raw spectrum to the continuum and write an ASCII file
    that can be fed directly into the CCF/BF analysis."""
    print(f"Normalizing {args.spectrum} "
          f"(poly order {args.poly_order}, {args.iterations} iterations)...")
    wl, nf, diag = normalize_spectrum_file(
        args.spectrum, args.format, poly_order=args.poly_order,
        iterations=args.iterations, low_clip=args.low_clip,
        high_clip=args.high_clip)

    tpl = None
    if args.template or args.teff is not None:
        tpl = load_template(args)

    outfile = args.output or \
        os.path.splitext(os.path.basename(args.spectrum))[0] + "_norm.txt"
    np.savetxt(outfile, np.column_stack([wl, nf]),
               fmt="%.4f %.5f",
               header=f"normalized from {args.spectrum} "
                      f"(poly order {args.poly_order}, "
                      f"{args.iterations} iterations)")
    print(f"Normalized spectrum written: {outfile}")

    plotfile = args.plot or "result_normalization.png"
    fig = make_norm_figure(diag, wl, nf, tpl)
    save_figure(fig, plotfile)

    summary = ("============ NORMALIZATION DONE ============\n"
               f"Raw spectrum : {args.spectrum}\n"
               f"Output       : {outfile}\n"
               f"Check the continuum fit in {plotfile}; if the normalized\n"
               "spectrum does not match the synthetic template, adjust the\n"
               "polynomial order and run again.\n"
               "============================================")
    print("\n" + summary)
    return dict(method="normalize", fig=fig, text=summary,
                output=outfile, plot=plotfile, wl=wl, flux=nf)


def compute_bjd(obstime_isot, ra_deg, dec_deg, site, exptime=None):
    """Mid-exposure BJD_TDB from a UTC time stamp (thesis: light curves and
    RVs are phased in Barycentric Julian Date)."""
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy.time import Time
    import astropy.units as u

    t = Time(obstime_isot, format="isot", scale="utc")
    if exptime:
        t = t + (float(exptime) / 2.0) * u.s
    loc = EarthLocation.of_site(site)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    ltt = t.light_travel_time(coord, kind="barycentric", location=loc)
    return float((t.tdb + ltt).jd)


def make_rv_curve_figure(rows, ncomp, t0=None, period=None):
    """RV curve figure: RV vs BJD, or vs orbital phase when an ephemeris
    (t0, period) is given."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()

    bjd = np.array([r["bjd"] for r in rows], dtype=float)
    folded = t0 is not None and period is not None and np.isfinite(bjd).all()
    x = ((bjd - t0) / period) % 1.0 if folded else bjd

    colors = ["C0", "C3"]
    labels = ["Component 1", "Component 2"]
    for j in range(ncomp):
        rv = [r["rv"][j] for r in rows]
        err = [r["rv_err"][j] for r in rows]
        ax.errorbar(x, rv, yerr=err, fmt="o", ms=5, capsize=2,
                    color=colors[j], label=labels[j])
    ax.set_xlabel("Orbital phase" if folded else "BJD_TDB")
    ax.set_ylabel("RV [km/s]")
    ax.set_title("Radial velocity curve")
    ax.legend()
    fig.tight_layout()
    return fig


def cmd_batch(args):
    """Process a spectral time series into an RV curve file.

    For every input spectrum: optional continuum normalization (raw FITS
    series), BF or CCF measurement, barycentric correction and BJD_TDB
    computation from the FITS header (DATE-OBS/EXPTIME) or --obstime.
    In SB2 mode the component with the larger BF area (larger light
    contribution) is always reported as component 1, so the labels do not
    swap between epochs.

    Output: result_RV_curve.txt (file, BJD_TDB, RV per component) and
    result_RV_curve.png — ready to be phased and fed to PyWD2015.
    """
    import glob
    files = sorted(set(sum((glob.glob(p) for p in args.spectra), [])))
    if not files:
        sys.exit("No files match the given pattern(s).")
    print(f"{len(files)} spectra to process.\n")

    tpl_wl, tpl_flux = load_template(args)

    # coordinates once (SIMBAD / explicit); per-file obstime from headers
    ra, dec = args.ra, args.dec
    if ra is None and args.object:
        try:
            ra, dec = resolve_target(args.object)
            print(f"SIMBAD: '{args.object}' -> RA = {ra:.5f} deg, "
                  f"Dec = {dec:.5f} deg\n")
        except ValueError as exc:
            print(f"Warning: {exc}\n")

    ncomp = args.components
    rows = []
    for path in files:
        print(f"--- {os.path.basename(path)} ---")
        try:
            if args.normalize:
                wl, fx, _ = normalize_spectrum_file(
                    path, args.format, poly_order=args.poly_order,
                    iterations=args.iterations)
            else:
                orders = read_spectrum(path, args.format)
                wl = np.concatenate([w for w, _ in orders])
                fx = np.concatenate([f for _, f in orders])
                srt = np.argsort(wl)
                wl, fx = wl[srt], fx[srt]
            if args.wave_min or args.wave_max:
                lo = args.wave_min or -np.inf
                hi = args.wave_max or np.inf
                sel = (wl >= lo) & (wl <= hi)
                wl, fx = wl[sel], fx[sel]

            hdr = fits_header_info(path)
            obstime = args.obstime or hdr["obstime"]
            ra_i = ra if ra is not None else hdr["ra"]
            dec_i = dec if dec is not None else hdr["dec"]

            if args.method == "bf":
                bf_result = compute_bf(wl, fx, tpl_wl, tpl_flux,
                                       vel_range=args.vel_range, dv=args.dv,
                                       svd_rcond=args.svd_rcond,
                                       smooth_kms=args.smooth)
                comps, _ = fit_bf_peaks(bf_result["velocity"],
                                        bf_result["bf_smooth"],
                                        components=ncomp,
                                        min_sep=args.min_sep)
                if ncomp == 2:
                    # stable labelling: primary = larger BF area
                    comps.sort(key=lambda c: c["amp"] * c["sigma"],
                               reverse=True)
            else:
                result = run_ccf([(wl, fx)], tpl_wl, tpl_flux,
                                 args.rv_min, args.rv_max, args.rv_step)
                comps = [dict(rv=result["rv"], rv_err=result["rv_err"])]

            vbary, bjd = 0.0, np.nan
            if ra_i is not None and obstime:
                vbary = barycentric_correction(ra_i, dec_i, obstime, args.site)
                bjd = compute_bjd(obstime, ra_i, dec_i, args.site,
                                  exptime=hdr.get("exptime"))
            rows.append(dict(file=os.path.basename(path), bjd=bjd,
                             rv=[c["rv"] + vbary for c in comps],
                             rv_err=[c["rv_err"] for c in comps]))
            msg = ", ".join(f"RV{j + 1} = {c['rv'] + vbary:8.3f} "
                            f"± {c['rv_err']:.3f}"
                            for j, c in enumerate(comps))
            print(f"  BJD = {bjd:.6f}  {msg} km/s")
        except Exception as exc:
            print(f"  SKIPPED ({exc.__class__.__name__}: {exc})")
    if not rows:
        sys.exit("No spectrum could be processed.")

    outfile = args.output or "result_RV_curve.txt"
    with open(outfile, "w") as f:
        f.write(f"# RV curve, method = {args.method.upper()}, "
                f"template = {args.template or f'PHOENIX T={args.teff}K'}\n")
        cols = "  ".join(f"RV{j + 1}[km/s]  RV{j + 1}_err" for j in range(ncomp))
        f.write(f"# file  BJD_TDB  {cols}\n")
        for r in rows:
            vals = "  ".join(f"{r['rv'][j]:.5f}  {r['rv_err'][j]:.5f}"
                             for j in range(len(r["rv"])))
            f.write(f"{r['file']}  {r['bjd']:.6f}  {vals}\n")
    print(f"\nRV curve written: {outfile}")

    plotfile = args.plot or "result_RV_curve.png"
    fig = make_rv_curve_figure(rows, ncomp, t0=args.t0, period=args.period)
    save_figure(fig, plotfile)
    return dict(method="batch", fig=fig, output=outfile, plot=plotfile,
                text=f"{len(rows)} spectra -> {outfile}")


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
                object=None, ra=None, dec=None, obstime=None, site="paranal",
                plot=None, output=None,
                rv_min=-200.0, rv_max=200.0, rv_step=0.5,
                vel_range=400.0, dv=None, svd_rcond=1e-3, smooth=None,
                components=1, min_sep=30.0,
                poly_order=5, iterations=8, low_clip=1.0, high_clip=4.0)
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


def _ask_target(kw):
    """Target identification: SIMBAD name lookup, manual coords as fallback."""
    name = ask("Star name for SIMBAD lookup (empty: skip / manual coords)",
               allow_empty=True)
    if name:
        try:
            ra, dec = resolve_target(name)
            print(f"  SIMBAD: RA = {ra:.5f} deg, Dec = {dec:.5f} deg")
            kw["object"], kw["ra"], kw["dec"] = name, ra, dec
        except ValueError as exc:
            print(f"  {exc}")
            name = None
    if not name:
        manual = ask("Enter coordinates manually? (y/n)", default="n",
                     cast=str, validate=lambda s: s.lower() in ("y", "n"))
        if manual.lower() == "y":
            kw["ra"] = ask("  RA [deg]", cast=float)
            kw["dec"] = ask("  Dec [deg]", cast=float)
    if kw.get("ra") is not None:
        kw["obstime"] = ask("Observation time (ISOT, e.g. 2024-12-03T02:30:00;"
                            " empty: read from FITS header)", allow_empty=True)
        kw["site"] = ask("Observatory (astropy site name: tug, paranal, ...)",
                         default="tug")
    return kw


def _ask_common_inputs(kw):
    """Inputs shared by both methods: files and wavelength range.
    Mutates and returns kw (which may already hold the target info)."""
    have_norm = ask("Do you already have a normalized spectrum? (y/n)",
                    default="y", cast=str,
                    validate=lambda s: s.lower() in ("y", "n"))
    if have_norm.lower() == "y":
        spectrum = ask("Normalized spectrum file (ASCII or FITS)",
                       cast=str, validate=os.path.isfile)
    else:
        raw = ask("Raw spectrum file (FITS or ASCII)",
                  cast=str, validate=os.path.isfile)
        order = ask("  Continuum polynomial order", default=5, cast=int)
        iters = ask("  Clipping iterations", default=8, cast=int)
        payload = cmd_normalize(make_args(spectrum=raw, poly_order=order,
                                          iterations=iters))
        spectrum = payload["output"]
        print(f"  Using normalized spectrum: {spectrum}\n")
        # carry target info over from the raw FITS header (fill gaps only)
        hdr = fits_header_info(raw)
        if not kw.get("obstime") and hdr["obstime"]:
            kw["obstime"] = hdr["obstime"]
            print(f"  FITS header: DATE-OBS = {kw['obstime']}")
        if kw.get("ra") is None and hdr["ra"] is not None:
            kw["ra"], kw["dec"] = hdr["ra"], hdr["dec"]
            print(f"  FITS header: RA = {kw['ra']:.5f} deg, "
                  f"Dec = {kw['dec']:.5f} deg")

    template = ask("Synthetic template file (.prf/.obs/ASCII; "
                   "empty: download PHOENIX)",
                   allow_empty=True,
                   validate=lambda p: p is None or os.path.isfile(p))
    kw.update(spectrum=spectrum, template=template)
    if template is None:
        kw["teff"] = ask("  Template T_eff [K]", cast=float)
        kw["logg"] = ask("  Template log g", default=4.5, cast=float)
        kw["feh"] = ask("  Template [Fe/H]", default=0.0, cast=float)
    kw["wave_min"] = ask("Minimum wavelength [A] (empty: all)",
                         allow_empty=True, cast=float)
    kw["wave_max"] = ask("Maximum wavelength [A] (empty: all)",
                         allow_empty=True, cast=float)
    return kw


def run_terminal_wizard():
    """Step-by-step RV analysis in the terminal (when the GUI cannot open)."""
    print("=" * 60)
    print(" RV ANALYSIS — interactive terminal mode")
    print("=" * 60)

    kw = _ask_target({})
    kw = _ask_common_inputs(kw)

    print("\nWhich method do you want to continue with?")
    print("  [1] CCF — cross-correlation (practical for single stars / SB1)")
    print("  [2] BF  — broadening function (recommended for binaries / SB2)")
    secim = ask("Choice", default="2", cast=str,
                validate=lambda s: s in ("1", "2"))

    if secim == "1":
        kw["rv_min"] = ask("RV scan lower limit [km/s]", default=-200.0, cast=float)
        kw["rv_max"] = ask("RV scan upper limit [km/s]", default=200.0, cast=float)
        kw["rv_step"] = ask("RV step [km/s]", default=0.5, cast=float)
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
        print()
        cmd_bf(make_args(**kw))

    print("\nDone. Result files and the fit figure were saved "
          "to the working directory.")


def run_gui():
    """Minimal tkinter widget.

    Sections, top to bottom: target (SIMBAD lookup with manual-coordinate
    fallback, for the barycentric correction), input files (with a
    'Normalize raw...' dialog for un-normalized spectra), wavelength
    range, CCF/BF method choice, Run, result text and the embedded fit
    plot. The analysis core is shared with the CLI (cmd_ccf/cmd_bf); the
    result files (result_*.txt, result_*.png) are written to disk as usual.
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

    def browse(var, title):
        p = filedialog.askopenfilename(
            title=title,
            filetypes=[("Spectrum files",
                        "*.fits *.fit *.txt *.dat *.ascii *.obs *.prf"),
                       ("All files", "*.*")])
        if p:
            var.set(p)

    # --- target / barycentric correction ---
    target_var = tk.StringVar()
    ra_var, dec_var = tk.StringVar(), tk.StringVar()
    time_var, site_var = tk.StringVar(), tk.StringVar(value="paranal")

    trow = ttk.LabelFrame(main, text="Target (optional, for barycentric "
                                     "correction)", padding=6)
    trow.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))

    ttk.Label(trow, text="Name:").grid(row=0, column=0, sticky="w")
    ttk.Entry(trow, textvariable=target_var, width=18).grid(row=0, column=1,
                                                            padx=2)

    def simbad_lookup():
        name = target_var.get().strip()
        if not name:
            messagebox.showinfo("SIMBAD", "Enter a star name first.")
            return
        try:
            ra, dec = resolve_target(name)
        except ValueError as exc:
            messagebox.showwarning(
                "SIMBAD", f"{exc}\n\nEnter the coordinates manually "
                          "in the RA/Dec fields.")
            return
        ra_var.set(f"{ra:.5f}")
        dec_var.set(f"{dec:.5f}")

    ttk.Button(trow, text="SIMBAD", command=simbad_lookup).grid(row=0, column=2,
                                                                padx=(2, 12))
    ttk.Label(trow, text="RA [deg]:").grid(row=0, column=3)
    ttk.Entry(trow, textvariable=ra_var, width=10).grid(row=0, column=4, padx=2)
    ttk.Label(trow, text="Dec [deg]:").grid(row=0, column=5)
    ttk.Entry(trow, textvariable=dec_var, width=10).grid(row=0, column=6, padx=2)

    ttk.Label(trow, text="Obs time (ISOT):").grid(row=1, column=0, columnspan=2,
                                                  sticky="w", pady=(4, 0))
    ttk.Entry(trow, textvariable=time_var, width=20).grid(row=1, column=2,
                                                          columnspan=2,
                                                          sticky="w",
                                                          pady=(4, 0))
    ttk.Label(trow, text="Site:").grid(row=1, column=4, sticky="e", pady=(4, 0))
    ttk.Entry(trow, textvariable=site_var, width=10).grid(row=1, column=5,
                                                          columnspan=2,
                                                          sticky="w",
                                                          pady=(4, 0))

    # --- input files ---
    spec_var = tk.StringVar()
    tpl_var = tk.StringVar()

    ttk.Label(main, text="Normalized spectrum:").grid(row=1, column=0, sticky="w")
    ttk.Entry(main, textvariable=spec_var, width=48).grid(row=1, column=1,
                                                          sticky="ew", padx=4)
    fbtns = ttk.Frame(main)
    fbtns.grid(row=1, column=2, sticky="w")
    ttk.Button(fbtns, text="Browse...",
               command=lambda: browse(spec_var, "Select normalized spectrum")
               ).pack(side="left")

    ttk.Label(main, text="Synthetic spectrum:").grid(row=2, column=0, sticky="w")
    ttk.Entry(main, textvariable=tpl_var, width=48).grid(row=2, column=1,
                                                         sticky="ew", padx=4)
    ttk.Button(main, text="Browse...",
               command=lambda: browse(tpl_var, "Select synthetic spectrum")
               ).grid(row=2, column=2, sticky="w")

    # --- wavelength range ---
    wmin_var, wmax_var = tk.StringVar(), tk.StringVar()
    wrow = ttk.Frame(main)
    wrow.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Label(wrow, text="Wavelength range [Å]:").pack(side="left")
    ttk.Entry(wrow, textvariable=wmin_var, width=8).pack(side="left", padx=(4, 2))
    ttk.Label(wrow, text="–").pack(side="left")
    ttk.Entry(wrow, textvariable=wmax_var, width=8).pack(side="left", padx=2)
    ttk.Label(wrow, text="(empty: full range)").pack(side="left", padx=4)

    # --- method ---
    method_var = tk.StringVar(value="BF")
    mrow = ttk.Frame(main)
    mrow.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Label(mrow, text="Method:").pack(side="left")

    ccf_frame = ttk.Frame(main)
    bf_frame = ttk.Frame(main)

    def on_method_change():
        if method_var.get() == "CCF":
            bf_frame.grid_remove()
            ccf_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        else:
            ccf_frame.grid_remove()
            bf_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))

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

    # --- results + plot ---
    result_text = tk.Text(main, height=8, width=90, state="disabled",
                          font=("Courier", 10))
    result_text.grid(row=7, column=0, columnspan=3, sticky="ew", pady=6)

    plot_frame = ttk.Frame(main)
    plot_frame.grid(row=8, column=0, columnspan=3, sticky="nsew")
    main.rowconfigure(8, weight=1)
    canvas_holder = {"canvas": None}

    def _f(var, default=None):
        s = var.get().strip()
        return float(s) if s else default

    def show_payload(payload):
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

    # --- normalization dialog for raw spectra ---
    def normalize_dialog():
        dlg = tk.Toplevel(root)
        dlg.title("Normalize raw spectrum")
        dlg.transient(root)
        frm = ttk.Frame(dlg, padding=10)
        frm.grid(row=0, column=0)

        raw_var = tk.StringVar()
        ord_var = tk.StringVar(value="5")
        it_var = tk.StringVar(value="8")

        ttk.Label(frm, text="Raw spectrum (FITS/ASCII):").grid(row=0, column=0,
                                                               sticky="w")
        ttk.Entry(frm, textvariable=raw_var, width=42).grid(row=0, column=1,
                                                            padx=4)
        ttk.Button(frm, text="Browse...",
                   command=lambda: browse(raw_var, "Select raw spectrum")
                   ).grid(row=0, column=2)

        prow = ttk.Frame(frm)
        prow.grid(row=1, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(prow, text="Polynomial order").pack(side="left")
        ttk.Entry(prow, textvariable=ord_var, width=4).pack(side="left", padx=(2, 12))
        ttk.Label(prow, text="Iterations").pack(side="left")
        ttk.Entry(prow, textvariable=it_var, width=4).pack(side="left", padx=2)

        ttk.Label(frm, text="Preview overlays the result on the synthetic\n"
                            "spectrum; adjust the order until they match, "
                            "then press Use.", justify="left"
                  ).grid(row=2, column=0, columnspan=3, sticky="w")

        state = {"payload": None}

        def do_preview():
            try:
                if not raw_var.get().strip():
                    raise ValueError("No raw spectrum file selected.")
                kw = dict(spectrum=raw_var.get().strip(),
                          template=tpl_var.get().strip() or None,
                          poly_order=int(ord_var.get()),
                          iterations=int(it_var.get()))
                state["payload"] = cmd_normalize(make_args(**kw))
            except Exception as exc:
                messagebox.showerror("Error", str(exc), parent=dlg)
                return
            show_payload(state["payload"])
            # auto-fill target info from the FITS header, if present
            hdr = fits_header_info(raw_var.get().strip())
            if hdr["obstime"] and not time_var.get().strip():
                time_var.set(hdr["obstime"])
            if hdr["ra"] is not None and not ra_var.get().strip():
                ra_var.set(f"{hdr['ra']:.5f}")
                dec_var.set(f"{hdr['dec']:.5f}")
            if hdr["object"] and not target_var.get().strip():
                target_var.set(str(hdr["object"]))

        def do_use():
            if state["payload"] is None:
                do_preview()
            if state["payload"] is not None:
                spec_var.set(state["payload"]["output"])
                dlg.destroy()

        brow = ttk.Frame(frm)
        brow.grid(row=3, column=0, columnspan=3, pady=(8, 0))
        ttk.Button(brow, text="Preview", command=do_preview).pack(side="left",
                                                                  padx=4)
        ttk.Button(brow, text="Use", command=do_use).pack(side="left", padx=4)

    ttk.Button(fbtns, text="Normalize raw...", command=normalize_dialog
               ).pack(side="left", padx=(4, 0))

    def run_analysis():
        try:
            if not spec_var.get().strip():
                raise ValueError("No normalized spectrum file selected "
                                 "(use 'Normalize raw...' if you only have "
                                 "a raw spectrum).")
            if not tpl_var.get().strip():
                raise ValueError("No synthetic spectrum file selected.")
            kw = dict(spectrum=spec_var.get().strip(),
                      template=tpl_var.get().strip(),
                      wave_min=_f(wmin_var), wave_max=_f(wmax_var),
                      object=target_var.get().strip() or None,
                      ra=_f(ra_var), dec=_f(dec_var),
                      obstime=time_var.get().strip() or None,
                      site=site_var.get().strip() or "paranal")
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
        show_payload(payload)

    ttk.Button(main, text="Run", command=run_analysis
               ).grid(row=6, column=0, columnspan=3, pady=6)

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
    p.add_argument("--object", help="Star name; coordinates are resolved via "
                                    "SIMBAD for the barycentric correction")
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

    p_batch = sub.add_parser("batch",
                             help="Process a spectral time series into an "
                                  "RV curve (for PyWD2015 etc.)")
    p_batch.add_argument("--spectra", nargs="+", required=True,
                         help="Spectrum files or glob patterns "
                              "(e.g. 'data/*.fits')")
    p_batch.add_argument("--method", default="bf", choices=["bf", "ccf"],
                         help="RV measurement method (default: bf)")
    p_batch.add_argument("--format", default="auto",
                         choices=["auto", "s2d", "text"])
    p_batch.add_argument("--template",
                         help="Synthetic template file (.prf/.obs/ASCII)")
    p_batch.add_argument("--teff", type=float, help="Template T_eff [K] (expecto)")
    p_batch.add_argument("--logg", type=float, default=4.5, help="Template log g")
    p_batch.add_argument("--feh", type=float, default=0.0, help="Template [Fe/H]")
    p_batch.add_argument("--wave-min", type=float,
                         help="Minimum wavelength to use [A]")
    p_batch.add_argument("--wave-max", type=float,
                         help="Maximum wavelength to use [A]")
    p_batch.add_argument("--normalize", action="store_true",
                         help="Continuum-normalize each raw spectrum first")
    p_batch.add_argument("--poly-order", type=int, default=5,
                         help="Continuum polynomial order (with --normalize)")
    p_batch.add_argument("--iterations", type=int, default=8,
                         help="Clipping iterations (with --normalize)")
    p_batch.add_argument("--object", help="Star name for SIMBAD coordinates")
    p_batch.add_argument("--ra", type=float, help="RA [deg]")
    p_batch.add_argument("--dec", type=float, help="Dec [deg]")
    p_batch.add_argument("--obstime",
                         help="Observation time override (default: per-file "
                              "FITS header DATE-OBS)")
    p_batch.add_argument("--site", default="paranal",
                         help="Observatory (astropy site name)")
    p_batch.add_argument("--vel-range", type=float, default=400.0,
                         help="BF window half-width [km/s]")
    p_batch.add_argument("--dv", type=float, help="BF velocity step [km/s]")
    p_batch.add_argument("--svd-rcond", type=float, default=1e-3,
                         help="BF SVD cutoff")
    p_batch.add_argument("--smooth", type=float,
                         help="BF smoothing FWHM [km/s]")
    p_batch.add_argument("--components", type=int, default=2, choices=[1, 2],
                         help="Components to fit (default: 2)")
    p_batch.add_argument("--min-sep", type=float, default=30.0,
                         help="Minimum peak separation [km/s]")
    p_batch.add_argument("--rv-min", type=float, default=-200.0,
                         help="CCF scan lower limit [km/s]")
    p_batch.add_argument("--rv-max", type=float, default=200.0,
                         help="CCF scan upper limit [km/s]")
    p_batch.add_argument("--rv-step", type=float, default=0.5,
                         help="CCF step [km/s]")
    p_batch.add_argument("--t0", type=float,
                         help="Ephemeris T0 [BJD] for phase-folding the plot")
    p_batch.add_argument("--period", type=float,
                         help="Orbital period [days] for phase-folding")
    p_batch.add_argument("--output", help="RV curve file "
                                          "(default: result_RV_curve.txt)")
    p_batch.add_argument("--plot", help="RV curve figure "
                                        "(default: result_RV_curve.png)")
    p_batch.set_defaults(func=cmd_batch)

    p_norm = sub.add_parser("normalize",
                            help="Continuum-normalize a raw spectrum "
                                 "(iterative polynomial fitting)")
    p_norm.add_argument("--spectrum", required=True, help="Raw spectrum file")
    p_norm.add_argument("--format", default="auto",
                        choices=["auto", "s2d", "text"])
    p_norm.add_argument("--poly-order", type=int, default=5,
                        help="Continuum polynomial order")
    p_norm.add_argument("--iterations", type=int, default=8,
                        help="Sigma-clipping iterations")
    p_norm.add_argument("--low-clip", type=float, default=1.0,
                        help="Rejection threshold below the fit [sigma]")
    p_norm.add_argument("--high-clip", type=float, default=4.0,
                        help="Rejection threshold above the fit [sigma]")
    p_norm.add_argument("--template",
                        help="Synthetic spectrum to overlay for comparison")
    p_norm.add_argument("--teff", type=float, help="Template T_eff [K] (expecto)")
    p_norm.add_argument("--logg", type=float, default=4.5, help="Template log g")
    p_norm.add_argument("--feh", type=float, default=0.0, help="Template [Fe/H]")
    p_norm.add_argument("--output", help="Output ASCII file "
                                         "(default: <input>_norm.txt)")
    p_norm.add_argument("--plot", help="Figure PNG file name "
                                       "(default: result_normalization.png)")
    p_norm.set_defaults(func=cmd_normalize)

    p_demo = sub.add_parser("demo",
                            help="CCF vs BF comparison on synthetic SB2 data")
    p_demo.add_argument("--plot", help="Comparison figure PNG file name")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
