#!/usr/bin/env python3
"""
rv_analysis.py — Radial Velocity (RV) measurement from stellar spectra
======================================================================

Two independent methods in one tool:

1. CCF  (Cross-Correlation Function)
   Weighted line-mask CCF (Baranne et al. 1996; Pepe et al. 2002; the
   construction applied by Pino et al. 2018, A&A 619, A3): mask entries
   are the line cores of the template, weighted by line depth, and the
   CCF is the weight-averaged observed line depth over the RV grid. The
   continuum and the pressure-broadened hydrogen wings drop out of the
   correlation — the hydrogen series is excluded from the mask by
   default — which is what makes the method reliable for very hot (OBA)
   stars, where a full-spectrum correlation is dominated by the Balmer
   wings and biased by emission-filled H cores. The peak is measured
   with a Gaussian on a linear baseline over a +-2 FWHM window. The
   previous full-spectrum cross-covariance (Between the Lines 2024
   workshop, E. Sedaghati) remains available as --ccf-mode template.

2. BF   (Broadening Function, Rucinski 1992, 2002)
   The observed spectrum is modelled as the convolution of the template
   with a velocity-space broadening function,  S(v) = B(v) * T(v),
   solved linearly via Singular Value Decomposition (SVD). Because the
   method is linear, the components of binary/multiple systems (SB2)
   appear as independent sharp peaks: peak centres give the RVs, widths
   the rotation (vsini), areas the light contributions. The spectra
   enter the SVD as line depths (1 - flux), so the BF baseline sits at
   zero. Sums of Gaussians are fitted to the smoothed BF; the starting
   points come either from an automatic iterative fit-and-subtract
   search or from per-component [amp, RV, sigma] estimates the user
   reads off the plot (--guess / --guess-amp / --guess-sigma).

   Several practices are adopted from the saphires package (Tofflemire;
   github.com/tofflemire/saphires): multi-order spectra get one BF per
   echelle order combined with 1/std^2 sideband weights
   (bf.weight_combine), the BF is smoothed by the instrumental FWHM c/R
   when the resolution is known (bf.analysis), the noisy BF edges are
   trimmed before profile fitting (fit_trim), unphysical pixels are
   interpolated over (utils.prepare cr_trim), and the barycentric
   correction is applied with the relativistic cross term
   RV + v_bary + RV*v_bary/c (utils.brvc; Wright & Eastman 2014).

The epoch and the observatory
-----------------------------
An RV is only as good as the frame it is quoted in, and the two ways of
losing that silently are the observatory (Earth rotation alone is +-0.46
km/s) and the epoch. Both are therefore taken from the file itself
whenever it says:

  * the observatory comes from the header (Neo-Narval
    LONGITUD/LATITUDE/OBSELEV, ESO 'TEL GEOLON/GEOLAT/GEOELEV', ...)
    unless --site is given explicitly, and the choice is printed;
    a file that records no position at all (SOPHIE, HARPS s1d) falls
    back to the home observatory of the instrument it names, never
    straight to the default site;
  * a Neo-Narval level-2/3 file is FOUR exposures combined but keeps the
    header of the first, so the epoch is rebuilt from the exposure-meter
    table, which spans the whole sequence (flux-weighted mid-time);
  * whatever the pipeline itself computed - BERV, BTLA - is used as a
    cross-check on our own values, and a disagreement is reported.

Modes of operation
------------------
1) INTERACTIVE (recommended): run without arguments.
   A minimal tkinter widget opens: pick the normalized observed spectrum
   and the synthetic template, choose the method (CCF, BF or TODCOR —
   the last one asks for a second template), press Run. The fit
   plot is embedded in the window. Without tkinter/display, a terminal
   wizard with the same steps starts instead.

   python rv_analysis.py

2) Command line (for scripting / time series):

   python rv_analysis.py ccf --spectrum spec.fits --format s2d \
       --teff 6628 --logg 4.251 --feh 0.17 --rv-min -20 --rv-max 100
   python rv_analysis.py bf --spectrum spec.obs --template synth.prf \
       --vel-range 500 --components 2 --wave-min 500 --wave-max 550

3) Self-test on synthetic data (no input files needed):

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
    columns are used as wavelength (Angstrom or nm, auto-detected) and
    flux; comment lines (# ; ! %), header lines and '-' placeholders are
    skipped automatically
  * ESPRESSO S2D FITS (multi-order echelle: flux ext=1, wavelength ext=4)
- Synthetic template spectrum:
  * synth3 / SynthV style .prf files, .obs files, or any ASCII table
    (same tolerant reader)
  * or a PHOENIX model downloaded via `expecto` (--teff/--logg/--feh)

Example data (like BinMag): see examples/example_observed.obs and
examples/example_synthetic.prf in this repository.
"""

import argparse
import datetime
import os
import platform
import sys
import time
import traceback
from collections import Counter

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

try:
    import astropy.constants as const
    C_KMS = const.c.value / 1000.0
except ImportError:
    C_KMS = 299792.458


# ----------------------------------------------------------------------
# Run log
# ----------------------------------------------------------------------
# Every run writes a log file: the full console output of the run, with a
# header recording when it was run, with which command line, in which
# directory and against which package versions. An RV is only as good as
# the settings it was measured with, so the log is what lets a number in
# result_BF.txt be traced back to the run that produced it months later.
#
# The log is written by teeing sys.stdout/sys.stderr, so it catches every
# print, every warning and every traceback without the rest of the code
# having to know about it. Runs append to the same file, newest last.

LOG_DEFAULT_NAME = "rv_analysis.log"
# Separator written between runs so the appended file stays readable.
LOG_RULE = "=" * 70


class _Tee:
    """Write to the console and to the log file at the same time.

    Delegates everything else (isatty, encoding, ...) to the console
    stream, so colour detection and anything else asking the terminal
    for its properties behaves exactly as before.
    """

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def write(self, text):
        self._stream.write(text)
        self._stream.flush()
        try:
            self._log.write(text)
            self._log.flush()
        except (ValueError, OSError):
            pass          # log file already closed or disk full: keep going
        return len(text)

    def flush(self):
        self._stream.flush()
        try:
            self._log.flush()
        except (ValueError, OSError):
            pass

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _package_versions():
    """Version strings of the packages the results depend on."""
    names = ("numpy", "scipy", "astropy", "matplotlib", "PyAstronomy")
    out = []
    for name in names:
        try:
            mod = __import__(name)
            out.append(f"{name} {getattr(mod, '__version__', '?')}")
        except Exception:
            out.append(f"{name} (not installed)")
    return ", ".join(out)


def pop_log_args(argv):
    """Take --log FILE / --log=FILE / --no-log out of argv.

    Scanned and removed before argparse sees the command line, so the
    options work in any position (before or after the subcommand) and the
    log is already open while the arguments themselves are parsed —
    a usage error is then logged too.

    Returns (path, enabled).
    """
    path, enabled = None, True
    rest = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--no-log":
            enabled = False
        elif arg == "--log":
            if i + 1 >= len(argv):
                sys.exit("--log expects a file name")
            path = argv[i + 1]
            i += 1
        elif arg.startswith("--log="):
            path = arg.split("=", 1)[1]
        else:
            rest.append(arg)
        i += 1
    argv[1:] = rest
    return path, enabled


def start_logging(path=None, enabled=True):
    """Open the run log and start teeing stdout/stderr into it.

    Returns the state needed by finish_logging(), or None when logging is
    off or the file cannot be opened (a log that cannot be written is
    never a reason to abandon the analysis).
    """
    if not enabled:
        return None
    path = os.path.abspath(path or LOG_DEFAULT_NAME)
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        log = open(path, "a", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"Warning: cannot write the log file {path}: {exc}",
              file=sys.stderr)
        return None

    started = datetime.datetime.now()
    log.write(f"\n{LOG_RULE}\n")
    log.write(f"rv_analysis.py run started {started:%Y-%m-%d %H:%M:%S}\n")
    log.write(f"{LOG_RULE}\n")
    log.write("command    : {}\n".format(
        " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
        if len(sys.argv) > 1
        else f"{os.path.basename(sys.argv[0])} (interactive mode)"))
    log.write(f"directory  : {os.getcwd()}\n")
    log.write(f"python     : {platform.python_version()} "
              f"({sys.executable})\n")
    log.write(f"packages   : {_package_versions()}\n")
    log.write(f"platform   : {platform.platform()}\n")
    log.write(f"{LOG_RULE}\n\n")
    log.flush()

    state = (log, sys.stdout, sys.stderr, time.time(), path)
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)
    return state


def finish_logging(state, status="finished", exc_info=None):
    """Write the closing line, restore the streams and close the log."""
    if state is None:
        return
    log, out, err, t0, path = state
    if exc_info is not None:
        log.write("\n")
        traceback.print_exception(*exc_info, file=log)
    log.write(f"\n{LOG_RULE}\n")
    log.write(f"run {status} in {time.time() - t0:.1f} s "
              f"({datetime.datetime.now():%Y-%m-%d %H:%M:%S})\n")
    log.write(f"{LOG_RULE}\n")
    sys.stdout, sys.stderr = out, err
    log.close()
    print(f"Log written to {path}")

# BF velocity step of the reference implementation (BF_main stepV). The
# per-bin BF amplitude scales with the step, so this default keeps amp
# values on the reference scale; data with coarser pixels use the pixel.
BF_STEPV = 5.0

# Sampling guards for the Gaussian component fit.
#
# The Gaussians are fitted to the BF sampled on a grid of step dv, so a
# component whose sigma spans only a bin or two is not constrained by the
# data: curve_fit still returns a centre, but the covariance is nearly
# singular and the reported rv_err becomes meaningless (values of 10^2 to
# 10^4 km/s in practice). The flat 5 km/s reference step does exactly that
# to the sharp-lined stars of a high-resolution spectrograph, so when the
# resolution is known the default step is instead tied to the instrumental
# FWHM: BF_STEP_PER_FWHM samples across it.
BF_STEP_PER_FWHM = 4.0
# A fitted sigma below this many dv bins counts as undersampled.
BF_MIN_SIGMA_BINS = 2.0
# Refinement passes allowed when an undersampled component is found.
BF_MAX_REFINE = 2
# Never refine below this fraction of the data pixel: the BF carries no
# information on a finer grid, and oversampling a correlated profile only
# shrinks the formal errors without improving the velocities.
BF_REFINE_PIXEL_FRAC = 1.0
# rv_err above this [km/s] means the component is not measured at all.
BF_DEGENERATE_ERR = 5.0


def bf_default_step(inst_fwhm=None, pixel_kms=None):
    """Default BF velocity step dv [km/s].

    With a known instrumental FWHM (from --spectrograph/--resolution) the
    step samples the narrowest profile the instrument can produce,
    BF_STEP_PER_FWHM times across its FWHM, so the Gaussian fit is always
    constrained by several bins. Without one the reference BF_STEPV is
    kept. Never finer than the data pixel, which sets the real limit.
    """
    step = (float(inst_fwhm) / BF_STEP_PER_FWHM if inst_fwhm
            else BF_STEPV)
    if pixel_kms is not None:
        step = max(step, float(pixel_kms))
    return step


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


def gauss_no(x, amp, mu, sig):
    """Single Gaussian without offset (reference BF_main 'gaussian')."""
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def two_gauss_no(x, amp1, mu1, sig1, amp2, mu2, sig2):
    """Double Gaussian without offset — the exact fit model of the
    reference document (BF_main.txt)."""
    gauss1 = amp1 * np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.)))
    gauss2 = amp2 * np.exp(-np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))
    return gauss1 + gauss2


def gauss_lin(x, amp, mu, sig, slope, offset):
    """Gaussian on a linear baseline — the CCF peak model of Pino et al.
    (2018), f(v) = C + B v + A exp(-(v-v0)^2 / 2 sigma^2): the linear
    term absorbs the local slope left by broad features so the fitted
    centre is not pulled."""
    return (amp * np.exp(-((x - mu) ** 2) / (2.0 * sig ** 2))
            + slope * x + offset)


def multi_gauss_no(x, *params):
    """Sum of N Gaussians; params = (amp, mu, sigma) per component, so
    len(params) = 3N. A trailing (3N+1)-th parameter, when present, is a
    shared constant baseline offset — the saphires d_gaussian_off /
    t_gaussian_off model: BFs computed against an observed template sit
    on a non-zero pedestal that the fit must carry."""
    params = list(params)
    offset = params.pop() if len(params) % 3 == 1 else 0.0
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for k in range(0, len(params), 3):
        amp, mu, sig = params[k:k + 3]
        y = y + amp * np.exp(-((x - mu) ** 2) / (2.0 * sig ** 2))
    return y + offset


def eval_gauss_model(x, popt):
    """Evaluate the fitted Gaussian model from the popt length
    (multiples of 3: offset-free N-Gaussian models; 4/7: models with a
    constant offset term)."""
    if len(popt) == 4:
        return gauss(x, *popt)
    if len(popt) == 7:
        return two_gauss(x, *popt)
    return multi_gauss_no(x, *popt)


def multi_rot_profile(x, params, epsilon=0.6, inst_sigma_pix=None):
    """Sum of N Gray (1992) rotational profiles, params = (amp, v0,
    vsini) per component; amp is the profile height at the centre. A
    trailing (3N+1)-th parameter, when present, is a shared constant
    baseline offset (see multi_gauss_no). When inst_sigma_pix is given
    the summed profile is convolved with the instrumental Gaussian
    (x must then be a uniform velocity grid). The DDO-series / saphires
    practice for the broadened components of close binaries."""
    params = list(params)
    offset = params.pop() if len(params) % 3 == 1 else 0.0
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    peak = 2.0 * (1.0 - epsilon) + 0.5 * np.pi * epsilon
    for k in range(0, len(params), 3):
        amp, v0, vsini = params[k:k + 3]
        u = (x - v0) / max(abs(vsini), 1e-6)
        inside = np.abs(u) < 1.0
        prof = np.zeros_like(x)
        prof[inside] = (2.0 * (1.0 - epsilon) * np.sqrt(1.0 - u[inside] ** 2)
                        + 0.5 * np.pi * epsilon * (1.0 - u[inside] ** 2))
        y = y + amp * prof / peak
    if inst_sigma_pix:
        y = gaussian_filter1d(y, inst_sigma_pix)
    return y + offset


def model_offset(popt):
    """Shared constant baseline of a fitted model; 0 when the model has
    no offset term. Offset models carry it as the last parameter and
    have len(popt) % 3 == 1 (4, 7, 3N+1)."""
    popt = np.ravel(np.asarray(popt, dtype=float))
    return float(popt[-1]) if popt.size % 3 == 1 else 0.0


def component_area(comp, epsilon=0.6):
    """Profile integral of a fitted component — the 'Amp' quoted in the
    saphires bf plots (fit_int): amp*sigma*sqrt(2 pi) for a Gaussian;
    for the unit-peak Gray rotational profile amp*vsini*
    (pi(1-eps) + 2 pi eps/3) / (2(1-eps) + pi eps/2). Proportional to
    the component's light contribution."""
    if comp.get("kind") == "rot":
        num = np.pi * (1.0 - epsilon) + 2.0 * np.pi * epsilon / 3.0
        den = 2.0 * (1.0 - epsilon) + 0.5 * np.pi * epsilon
        return abs(comp["amp"] * comp["sigma"]) * num / den
    return abs(comp["amp"] * comp["sigma"]) * np.sqrt(2.0 * np.pi)


def eval_bf_model(x, model):
    """Evaluate a fitted profile model descriptor as returned by
    fit_bf_peaks / fit_peak_with_base / fit_ccf_peak: dict(kind, popt,
    epsilon, inst_sigma_pix). kind 'rot' uses rotational profiles,
    'gauss_lin' the Gaussian-on-linear-baseline CCF model, anything
    else the Gaussian models."""
    if model.get("kind") == "rot":
        return multi_rot_profile(x, model["popt"],
                                 epsilon=model.get("epsilon", 0.6),
                                 inst_sigma_pix=model.get("inst_sigma_pix"))
    if model.get("kind") == "gauss_lin":
        return gauss_lin(x, *model["popt"])
    return eval_gauss_model(x, model["popt"])


def read_two_columns(path):
    """Tolerant two-column ASCII reader, no unit interpretation.

    Used by read_ascii_spectrum (wavelength, flux), which adds the
    nm->Angstrom rule on top of it. See read_ascii_spectrum for what the
    parser tolerates.
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
                    vals.append(np.nan)
            rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: no data lines found.")

    ncols = Counter(len(r) for r in rows).most_common(1)[0][0]
    if ncols < 2:
        raise ValueError(f"{path}: need at least two columns.")
    data = np.array([r[:2] for r in rows if len(r) == ncols], dtype=float)
    data = data[np.isfinite(data).all(axis=1)]
    if data.shape[0] < 5:
        raise ValueError(f"{path}: fewer than 5 usable data rows "
                         "(check the file format).")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def read_ascii_spectrum(path):
    """Tolerant ASCII spectrum reader (.txt/.dat/.ascii/.obs/.prf/...).

    Real data files often contain things np.loadtxt cannot handle: header
    lines, placeholders such as '-', Fortran D-exponent numbers, a varying
    column count. This reader:
      - skips comment lines starting with '#', ';', '!' or '%',
      - converts every line to numbers; tokens that cannot be converted
        (such as '-') count as NaN,
      - drops lines whose column count differs from the most common
        (modal) one, treating them as headers (SynthV .prf style),
      - takes the first two numeric columns (wavelength, flux) and drops
        rows containing NaN,
      - sorts by wavelength.

    Wavelengths may be given in Angstrom or nm: a median value below 1500
    is taken to mean nm and converted to Angstrom, the internal unit of
    the analysis.
    """
    wl, fx = read_two_columns(path)
    if np.median(wl) < 1500.0:
        wl = wl * 10.0
    return wl, fx


def read_iraf_multispec(hdu):
    """IRAF echelle 'multispec' reader: a 2D (orders x pixels) image
    whose wavelength solution lives in the WAT2_* header cards
    (wtype=multispec, one spec<N> attribute per order). Supports the
    dispersion types written by dispcor/ecreidentify: dtype 0 (linear),
    1 (log-linear) and 2 (nonlinear) with Chebyshev (ftype 1) and
    Legendre (ftype 2) function blocks; the Doppler factor z of each
    order is applied as w/(1+z), so rv/dop-corrected products come out
    in their corrected frame. Returns a list of (wavelength, flux)
    orders, or None when the HDU is not a multispec image.
    """
    import re
    h = hdu.header
    if "multispec" not in str(h.get("WAT0_001", "")).lower() and \
            str(h.get("CTYPE1", "")).strip().upper() != "MULTISPE":
        return None
    if hdu.data is None:
        return None
    data = np.asarray(hdu.data, dtype=float)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        return None

    keys = sorted((k for k in h if str(k).startswith("WAT2_")),
                  key=lambda s: int(str(s).split("_")[1]))
    wat2 = "".join(str(h[k]).ljust(68) for k in keys)
    specs = re.findall(r'spec(\d+)\s*=\s*"([^"]+)"', wat2)
    if not specs:
        raise ValueError("multispec FITS without spec<N> attributes in "
                         "the WAT2 cards.")

    orders = []
    npix = data.shape[1]
    p = np.arange(1.0, npix + 1.0)
    for num, s in sorted(specs, key=lambda t: int(t[0])):
        row = int(num) - 1
        if row >= data.shape[0]:
            continue
        vals = s.split()
        dtype = int(float(vals[2]))
        w1, dw = float(vals[3]), float(vals[4])
        z = float(vals[6])
        if dtype in (0, -1):
            wl = w1 + dw * (p - 1.0)
        elif dtype == 1:
            wl = 10.0 ** (w1 + dw * (p - 1.0))
        elif dtype == 2:
            wl = np.zeros(npix)
            i = 9
            while i + 2 < len(vals):
                wt, w0 = float(vals[i]), float(vals[i + 1])
                ftype = int(float(vals[i + 2]))
                if ftype not in (1, 2):
                    raise ValueError(
                        f"multispec order {num}: function type {ftype} "
                        "is not supported (only Chebyshev=1 and "
                        "Legendre=2); re-run IRAF dispcor with a "
                        "chebyshev/legendre solution or linearize the "
                        "spectrum.")
                order_n = int(float(vals[i + 3]))
                pmin, pmax = float(vals[i + 4]), float(vals[i + 5])
                coeffs = [float(v) for v in vals[i + 6:i + 6 + order_n]]
                n = (2.0 * p - (pmax + pmin)) / (pmax - pmin)
                if ftype == 1:
                    wfun = np.polynomial.chebyshev.chebval(n, coeffs)
                else:
                    wfun = np.polynomial.legendre.legval(n, coeffs)
                wl = wl + wt * (w0 + wfun)
                i += 6 + order_n
        else:
            raise ValueError(f"multispec order {num}: dispersion type "
                             f"{dtype} is not supported.")
        wl = wl / (1.0 + z)
        flux = data[row]
        steps = np.flatnonzero(np.diff(wl) <= 0)
        if steps.size:
            edges = np.concatenate(([0], steps + 1, [wl.size]))
            a, b = max(zip(edges[:-1], edges[1:]), key=lambda r: r[1] - r[0])
            wl, flux = wl[a:b], flux[a:b]
        if wl.size > 10:
            orders.append((wl, flux))
    return orders or None


def extract_flux_band(data, header):
    """Reduce a 2D/3D/4D image HDU to the 1D flux vector, when possible.

    Wavelength always runs along the last axis (NAXIS1). Degenerate
    leading axes (length 1, e.g. a (1, 1, N) UVES/MIDAS product or a
    (1, 1, 1, N) cube) are squeezed away. A remaining leading axis is
    treated as the IRAF 'band' axis (onedspec/apall products: band 1 =
    spectrum, 2 = raw, 3 = background, 4 = sigma): the band whose
    BANDID<n> card contains 'spectrum' is chosen, band 0 otherwise.
    Returns the 1D flux array, or None when the data cannot be reduced
    to one spectrum this way (e.g. a genuine 2D CCD image).
    """
    data = np.squeeze(np.asarray(data, dtype=float))
    while data.ndim > 1:
        nband = data.shape[0]
        band = 0
        ids = [str(header.get(f"BANDID{i + 1}", "")).lower()
               for i in range(nband)]
        if any(ids):
            # the IRAF band kind is the first word of the BANDID card
            # ('spectrum - background fit, ...', 'raw - ...',
            #  'background - ...', 'sigma - ...')
            for i, bid in enumerate(ids):
                if bid.split()[:1] == ["spectrum"]:
                    band = i
                    break
        elif nband > 8:
            # no BANDID cards and many rows: this is not a band axis
            # (more likely raw CCD rows or unlabeled echelle orders)
            return None
        data = np.squeeze(data[band])
    return data if data.ndim == 1 and data.size > 10 else None


def describe_raw_frame(path, header):
    """Return an explanatory error message when the header identifies a
    RAW (unreduced) observatory frame, else None.

    ESO raw science frames carry HIERARCH ESO DPR CATG/TYPE cards (the
    template keywords of the acquisition system), while reduced products
    carry HIERARCH ESO PRO CATG instead - a raw echelle CCD exposure can
    therefore be told apart from a reduced spectrum that merely failed
    to parse. The message points to the 'fetch-adp' subcommand, which
    downloads the reduced ADP counterpart from the ESO archive.
    """
    dpr = str(header.get("ESO DPR TYPE", header.get("ESO DPR CATG", "")))
    if not dpr or header.get("ESO PRO CATG"):
        return None
    inst = header.get("INSTRUME", "unknown instrument")
    mjd = header.get("MJD-OBS", "?")
    return (
        f"{path}: this is a RAW {inst} frame (ESO DPR TYPE = {dpr}, "
        f"MJD-OBS = {mjd}) - an unextracted CCD exposure, not a spectrum. "
        "No RV can be measured from it directly. Download its reduced "
        "(ADP phase-3) counterpart from the ESO archive with:\n"
        f"  rv_analysis.py fetch-adp --spectra '{path}'\n"
        "and run the analysis on the ADP file instead.")


def read_libre_esprit(hdul):
    """Libre-ESpRIT reader (NARVAL / Neo-Narval / ESPaDOnS 'st3' files).

    The spectrum sits in one binary table whose columns are
    'Wavelength1', 'Intensity', 'Polarization', 'Error', 'Null',
    'Continuum', 'Null 2' (a non-polarimetric product carries only the
    first columns). All echelle orders are concatenated into that single
    table, red order first, so the wavelength column is NOT monotonic:
    it climbs inside an order and drops back at every order boundary.
    The boundaries are the row indices of the 'Orderlimit' column of a
    later table (zero-padded at the end); when that table is missing
    they are recovered from the drops in the wavelength column.

    'Intensity' is the continuum-normalized Stokes I - the 'Continuum'
    column holds the ADU continuum it was divided by - which is what the
    BF/CCF steps want. The Stokes V / null columns are not used.

    Orders the pipeline could not solve a continuum for are dropped:
    there the normalized intensity scatters around zero and piles up on
    its clip bounds (-1 and 10), which is noise, not spectrum. In a
    low-S/N exposure this typically removes the blue and far-red ends of
    the coverage.
    Returns the orders blue-to-red as (wavelength, flux) pairs, or None
    when the file does not have this layout.
    """
    from astropy.io import fits
    wl = fx = cont = None
    limits = None
    for hdu in hdul:
        if not isinstance(hdu, fits.BinTableHDU):
            continue
        cols = {c.name.upper().replace(" ", ""): c.name for c in hdu.columns}
        if limits is None and "ORDERLIMIT" in cols:
            limits = np.ravel(np.asarray(hdu.data[cols["ORDERLIMIT"]], dtype=int))
        if wl is not None:
            continue
        # 'Wavelength1' in the polarimetric product, 'Wavelength' otherwise
        wname = next((cols[k] for k in ("WAVELENGTH1", "WAVELENGTH") if k in cols),
                     None)
        if wname is None or "INTENSITY" not in cols:
            continue
        # the null/polarization/continuum columns are the Libre-ESpRIT
        # signature: without one of them this is some other WAVE/FLUX table
        if not any(k in cols for k in
                   ("POLARIZATION", "CONTINUUM", "NULL", "NULL2")):
            continue
        wl = np.ravel(np.asarray(hdu.data[wname], dtype=float))
        fx = np.ravel(np.asarray(hdu.data[cols["INTENSITY"]], dtype=float))
        if "CONTINUUM" in cols:            # kept for the diagnosis below
            cont = np.ravel(np.asarray(hdu.data[cols["CONTINUUM"]],
                                       dtype=float))
    if wl is None or wl.size < 10:
        return None

    edges = []
    if limits is not None and limits.size:
        # keep the strictly increasing head of the column (the tail is
        # zero padding) and make sure the last order is closed off
        edges = [0]
        for v in limits:
            if edges[-1] < v <= wl.size:
                edges.append(int(v))
        if edges[-1] != wl.size:
            edges.append(wl.size)
    if len(edges) < 3:
        # no usable Orderlimit table: cut where the wavelength drops back
        drops = np.flatnonzero(np.diff(wl) <= 0) + 1
        edges = [0, *drops.tolist(), wl.size]

    orders, dropped = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        w, f = wl[a:b], fx[a:b]
        good = np.isfinite(w) & np.isfinite(f)
        w, f = w[good], f[good]
        if w.size < 10:
            continue
        # a normalized stellar spectrum is non-negative: a quarter of the
        # order below zero (or parked on the -1 / 10 clip bounds) means
        # the continuum was fitted to noise, not to a stellar continuum
        if np.mean((f < 0.0) | (f >= 10.0)) > 0.25:
            dropped.append((w.min(), w.max()))
            continue
        srt = np.argsort(w)
        orders.append((w[srt], f[srt]))
    if not orders:
        # The layout WAS recognized - saying "unrecognized FITS layout"
        # here would send the user looking for a reader bug instead of at
        # the exposure. Report what is actually wrong with the data.
        clipped = float(np.mean((fx < 0.0) | (fx >= 10.0)))
        floor = ""
        if cont is not None and cont.size == fx.size:
            med = float(np.nanmedian(cont))
            if med <= 2.0 + 1e-9:
                floor = (f" and the 'Continuum' column sits at its floor "
                         f"(median {med:g} ADU), so the extracted flux "
                         "cannot be recovered from it either")
        raise ValueError(
            "recognized as a Libre-ESpRIT product (NARVAL / Neo-Narval / "
            f"ESPaDOnS), but none of its {len(dropped)} echelle orders "
            "has a usable continuum solution: "
            f"{100 * clipped:.0f}% of the normalized-intensity pixels sit "
            f"on the -1 / 10 clip bounds{floor}. The DRS found no stellar "
            "continuum to divide by, so there is no spectrum in this file "
            "to normalize, correlate or measure - this is an exposure to "
            "drop, not a format problem. Check the other epochs of the "
            "same target (this one is unusable as delivered), or have the "
            "level-1/level-2 data re-reduced.")
    if dropped:
        lo = min(d[0] for d in dropped)
        hi = max(d[1] for d in dropped)
        print(f"Note: {len(dropped)} of {len(dropped) + len(orders)} "
              f"Libre-ESpRIT orders ({lo:.0f}-{hi:.0f} A) have no continuum "
              "solution (unusable S/N) and were skipped.")
    orders.sort(key=lambda o: o[0][0])          # blue to red
    return stitch_order_overlaps(orders)


def stitch_order_overlaps(orders, min_edge=0.03):
    """Cut each echelle order back to the half of its overlap with the
    neighbouring order, so consecutive orders meet without overlapping.

    Adjacent Libre-ESpRIT orders overlap by ~35% of their span, and the
    normalized intensity is at its worst exactly there: the blaze runs
    out at an order end, so the continuum division blows up and the
    normalized flux swings to the -1 / 10 clip bounds (in the observed
    files, ~25% of the pixels in the outer decile of an order against ~1%
    mid-order). Merging such orders by wavelength sorting interleaves
    those spikes into the clean data of the neighbouring order, which is
    where they are most damaging: the SVD sees them as sharp lines, and
    the line check compares the model against garbage pixels.

    Splitting each overlap at its midpoint keeps, for every wavelength,
    the order that samples it furthest from its own edge, and leaves no
    duplicated pixels behind. An order end with no overlapping
    neighbour (the two ends of the coverage, or a gap left by a dropped
    order) is trimmed by `min_edge` of the order span instead; the same
    minimum is enforced on the stitched side, for the rare narrow
    overlap. Orders are assumed sorted blue-to-red.
    """
    out = []
    for k, (w, f) in enumerate(orders):
        span = w[-1] - w[0]
        lo = w[0] + min_edge * span
        hi = w[-1] - min_edge * span
        if k > 0 and orders[k - 1][0][-1] > w[0]:
            lo = max(lo, 0.5 * (w[0] + orders[k - 1][0][-1]))
        if k + 1 < len(orders) and orders[k + 1][0][0] < w[-1]:
            hi = min(hi, 0.5 * (w[-1] + orders[k + 1][0][0]))
        sel = (w >= lo) & (w <= hi)
        if sel.sum() > 10:
            out.append((w[sel], f[sel]))
    return out or orders


def linear_wcs_axis(header, npix):
    """Axis values of a linear 1-D FITS WCS (CRVAL1 + CDELT1/CD1_1,
    CRPIX1), or None when the header carries no such solution."""
    if "CRVAL1" not in header:
        return None
    cdelt = header.get("CDELT1", header.get("CD1_1"))
    if not cdelt:
        return None
    crpix = float(header.get("CRPIX1", 1.0))
    return (float(header["CRVAL1"])
            + (np.arange(npix) + 1 - crpix) * float(cdelt))


def velocity_axis_note(axis, header):
    """Describe a linear WCS axis that CANNOT be a wavelength axis.

    A wavelength axis is strictly positive in every unit (A, nm, m);
    an axis running through zero is a VELOCITY axis. The case that
    reaches this code in practice is the ESO-MIDAS xcorr product
    ('..._ln_corr_v.fits'): the cross-correlation function itself,
    sampled on a velocity grid in km/s that is symmetric about zero,
    e.g. CRVAL1 = -31515.9, CDELT1 = 1.92358 over 32768 = 2^15 pixels.
    Read as a wavelength array it gives negative 'wavelengths' and the
    BF/CCF that follow are meaningless, so the reader stops here.

    Returns None when the axis is a plausible wavelength axis.
    """
    if axis is None or axis.size == 0 or float(np.nanmin(axis)) > 0.0:
        return None
    lo, hi = float(axis[0]), float(axis[-1])
    step = float(axis[1] - axis[0]) if axis.size > 1 else 0.0
    ctype = str(header.get("CTYPE1", "")).strip()
    symmetric = abs(lo + hi) <= 2.0 * abs(step)
    what = ("a pre-computed cross-correlation function"
            if symmetric else "a velocity axis")
    note = (f"the CRVAL1/CDELT1 axis runs from {lo:.4g} to {hi:.4g} "
            f"(step {step:.6g}) over {axis.size} pixels, so it is not a "
            f"wavelength axis (wavelengths are positive) but "
            f"{what} in velocity units")
    if ctype:
        note += f" [CTYPE1 = '{ctype}']"
    if symmetric:
        note += (".\nThis is the ESO-MIDAS xcorr product: the file already "
                 "IS the correlation profile, and there is no spectrum in "
                 "it to correlate again.\nStart from the spectrum that went "
                 "INTO the correlation (the merged, normalized spectrum on "
                 "a wavelength scale), not from its output: every method "
                 "here - 'ccf', 'bf' and 'todcor' - works on the spectrum.")
    else:
        note += (". Supply the spectrum on a wavelength scale instead.")
    return note


def correlation_profile_note(path):
    """Return velocity_axis_note() for `path` if it holds a correlation
    profile rather than a spectrum, else None.

    Cheap enough to run when a file is picked in the widget, so the user
    is told what the file is BEFORE being asked for a template that a
    correlation profile does not need. Never raises: anything it cannot
    read is simply not a profile as far as this check is concerned.
    """
    if not path or not os.path.isfile(path):
        return None
    if not str(path).lower().endswith((".fits", ".fit", ".fts")):
        return None
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is None or isinstance(hdu, fits.BinTableHDU):
                    continue
                data = np.asarray(hdu.data, dtype=float)
                if data.ndim > 1:
                    data = extract_flux_band(data, hdu.header)
                if data is None or data.ndim != 1:
                    continue
                axis = linear_wcs_axis(hdu.header, data.size)
                if axis is None:
                    continue
                note = velocity_axis_note(axis, hdu.header)
                if note:
                    return note
    except Exception:
        return None
    return None


def read_fits_spectrum(path):
    """FITS spectrum reader for the common layouts of raw/pipeline data.

    Supported:
      - ESPRESSO S2D (flux ext=1, wavelength ext=4, 2D echelle orders)
      - phase-3 style binary tables with WAVE/LAMBDA and FLUX columns
        (FEROS, HARPS, UVES, ...)
      - Libre-ESpRIT products (NARVAL / Neo-Narval / ESPaDOnS 'st3'):
        one table with all orders concatenated (see read_libre_esprit)
      - IRAF echelle 'multispec' images (CTYPE1=MULTISPE, WAT2 cards;
        linear, log and Chebyshev/Legendre dispersion solutions)
      - 1D image HDUs with a linear wavelength WCS (CRVAL1/CDELT1): the
        classic IRAF product, and multi-extension echelle products that
        carry one order per extension (e.g. SALT HRS pipeline files) —
        every such HDU becomes one order
      - 2D/3D/4D image HDUs with a linear wavelength WCS along NAXIS1:
        degenerate axes are squeezed and the IRAF band axis (BANDID
        cards: spectrum / raw / background / sigma) is reduced to the
        flux band (see extract_flux_band)
    Raw (unreduced) ESO frames are recognized by their DPR cards and
    rejected with a pointer to the fetch-adp subcommand.
    Returns a list of (wavelength, flux) pairs (one per echelle order).
    """
    from astropy.io import fits
    with fits.open(path) as hdul:
        try:
            flux = np.array(hdul[1].data, dtype=float)
            wvl = np.array(hdul[4].data, dtype=float)
            if flux.ndim == 2 and flux.shape == wvl.shape:
                return [(w, f) for w, f in zip(wvl, flux)]
        except (IndexError, TypeError, ValueError):
            pass
        try:
            esprit = read_libre_esprit(hdul)
        except ValueError as exc:               # recognized but unusable
            raise ValueError(f"{path}: {exc}") from None
        if esprit:
            return esprit
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
        for hdu in hdul:
            ms = read_iraf_multispec(hdu)
            if ms:
                return ms
        wcs_orders = []
        vel_note = None
        for hdu in hdul:
            if hdu.data is None or isinstance(hdu, fits.BinTableHDU):
                continue                # a table has no wavelength WCS
            data = np.asarray(hdu.data, dtype=float)
            h = hdu.header
            if data.ndim > 1:
                data = extract_flux_band(data, h)
            if data is not None and data.ndim == 1:
                wl = linear_wcs_axis(h, data.size)
                if wl is None:
                    continue
                # A wavelength axis is positive; anything running through
                # zero is a velocity axis (MIDAS xcorr output) and must
                # not be silently used as wavelengths.
                note = velocity_axis_note(wl, h)
                if note:
                    vel_note = vel_note or note
                    continue
                wcs_orders.append((wl, data))
        if wcs_orders:
            return wcs_orders
        if vel_note:
            raise ValueError(f"{path}: {vel_note}")
        for hdu in hdul:
            err = describe_raw_frame(path, hdu.header)
            if err:
                raise ValueError(err)
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data)
            h = hdu.header
            ctypes = " ".join(str(h.get(k, "")) for k in
                              ("CTYPE1", "CTYPE2", "CTYPE3")).strip()
            celestial = any(t in ctypes for t in ("RA--", "DEC-", "-TAN"))
            if data.ndim >= 3 or (data.ndim == 2 and celestial):
                inst = h.get("INSTRUME", "unknown instrument")
                tel = h.get("TELESCOP", "unknown telescope")
                raise ValueError(
                    f"{path}: this FITS contains IMAGING data, not a "
                    f"spectrum: {data.shape} pixels from {inst} on {tel}"
                    + (f", celestial WCS '{ctypes}'" if ctypes else "")
                    + ". No spectrum can be extracted from it; RVCalPy "
                    "needs a spectral layout (ESPRESSO S2D, a WAVE/FLUX "
                    "binary table, or a 1D image with CRVAL1/CDELT1).")
        layout = "; ".join(
            f"HDU{i} {type(h).__name__}"
            + (f" {np.asarray(h.data).shape}" if h.data is not None
               else " (no data)")
            for i, h in enumerate(hdul))
    raise ValueError(f"{path}: unrecognized FITS layout ({layout}); "
                     "expected S2D, a WAVE/FLUX binary table, an IRAF "
                     "multispec image (WAT2 cards), or a 1D-4D image with "
                     "CRVAL1/CDELT1 (band/degenerate axes are reduced "
                     "automatically).")


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
    interpreted as hourangle/deg. Any radial-velocity related cards
    (RV, VRAD, VHELIO, BERV, CCF RV, ... with a numeric value) are
    collected in 'rv_cards' as (keyword, value, comment) tuples.
    """
    from astropy.io import fits
    info = {"object": None, "obstime": None, "ra": None, "dec": None,
            "exptime": None, "rv_cards": [], "epoch_note": None,
            "neonarval": None, "ohp": None}
    try:
        with fits.open(path) as hdul:
            h = hdul[0].header
    except Exception:
        return info
    rv_tokens = {"RV", "RVC", "RVS", "VRAD", "RADVEL", "VHELIO", "VLSR",
                 "BERV", "BARYCORR", "HRV", "HJDRV"}
    for card in h.cards:
        key = str(card.keyword)
        if not key:
            continue
        tokens = set(key.replace("-", " ").replace("_", " ").split())
        if tokens & rv_tokens and isinstance(card.value, (int, float)) \
                and not isinstance(card.value, bool):
            info["rv_cards"].append((key, float(card.value),
                                     str(card.comment or "")))
        if len(info["rv_cards"]) >= 12:
            break
    info["object"] = h.get("OBJECT")
    info["obstime"] = h.get("DATE-OBS") or h.get("DATE_OBS")
    if info["obstime"] and ":" not in str(info["obstime"]) \
            and "T" not in str(info["obstime"]):
        # date-only DATE-OBS (Libre-ESpRIT writes '2025-06-15'): a full
        # ISOT timestamp elsewhere in the header replaces it outright
        for key in ("DATE-FIT", "DATE-BEG", "DATE-STR"):
            full = h.get(key)
            if full and "T" in str(full):
                info["obstime"] = str(full)
                break
    if info["obstime"] and ":" not in str(info["obstime"]) \
            and "T" not in str(info["obstime"]):
        for key in ("UTMIDDLE", "UT", "UTC-OBS", "UT-OBS", "TIME-OBS",
                    "UTSHUT"):
            ut = h.get(key)
            if ut in (None, ""):
                continue
            if isinstance(ut, (int, float)):
                hh = int(ut) % 24
                mm = int((ut % 1) * 60)
                ss = ((ut % 1) * 60 % 1) * 60
                ut = f"{hh:02d}:{mm:02d}:{ss:06.3f}"
            if ":" in str(ut):
                info["obstime"] = f"{info['obstime']} {ut}"
                if key == "UTMIDDLE":
                    info["exptime"] = 0.0
                break
    info["exptime"] = h.get("EXPTIME") if info["exptime"] is None \
        else info["exptime"]

    # Neo-Narval level-2/3: the primary header describes sub-exposure 1
    # of a 4-exposure sequence, so its times are NOT the epoch of the
    # product. Replace them with the mid-time of the whole sequence and
    # zero the exposure time, since that mid-time is already the middle.
    neo = neonarval_header_info(path)
    if neo and neo.get("mid_jd"):
        from astropy.time import Time
        # what the header cards alone would have given: start of
        # sub-exposure 1 plus half ITS exposure time
        old_jd = _isot_to_jd(info["obstime"])
        old_mid = (old_jd + float(info["exptime"] or 0.0) / 2.0 / 86400.0
                   if old_jd is not None else None)
        info["obstime"] = Time(neo["mid_jd"], format="jd",
                               scale="utc").isot
        info["exptime"] = 0.0
        info["neonarval"] = neo
        shift = ((neo["mid_jd"] - old_mid) * 1440.0
                 if old_mid is not None else None)
        info["epoch_note"] = (
            f"Neo-Narval: epoch = mid of the {neo['nsub']}-exposure "
            f"sequence ({neo['epoch_source']}"
            + (f", span {neo['span_s']:.0f} s" if neo["span_s"] else "")
            + f") = {info['obstime']}"
            + (f", {shift:+.1f} min from the first sub-exposure"
               if shift is not None and abs(shift) >= 0.05 else ""))

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

    # SOPHIE (OHP DRS): none of the keywords above exist in these files,
    # so everything still missing is filled from the 'OHP ...' cards
    # (see ohp_header_info). Last, so a standard card always wins.
    ohp = ohp_header_info(path)
    if ohp:
        info["ohp"] = ohp
        if info["object"] is None:
            info["object"] = ohp["object"]
        if info["ra"] is None and ohp["ra"] is not None:
            info["ra"], info["dec"] = ohp["ra"], ohp["dec"]
        if not info["obstime"] and ohp["obstime"]:
            info["obstime"] = ohp["obstime"]
            if info["exptime"] is None:
                info["exptime"] = ohp["exptime"]
            info["epoch_note"] = (
                "SOPHIE: DATE-OBS is the exposure START (OHP OBS DATE "
                "START)"
                + (f"; the mid-exposure epoch adds DKTM/2 = "
                   f"{ohp['exptime'] / 2.0:.0f} s" if ohp["exptime"]
                   else "; no DKTM card, so no mid-exposure shift")
                + (f". The DRS's own BJD is {ohp['drs_bjd']:.6f} (UTC-based)"
                   if ohp["drs_bjd"] else ""))
    return info


# Header cards giving the observatory position, as
# (longitude_East, latitude, altitude) triples. Longitudes are degrees
# East throughout (negative in Chile), the convention of OBSERVATORIES
# and of the MIDAS comp/bary command.
SITE_CARD_SETS = (
    ("LONGITUD", "LATITUDE", "OBSELEV"),        # Neo-Narval / NARVAL (TBL)
    ("HIERARCH ESO TEL GEOLON", "HIERARCH ESO TEL GEOLAT",
     "HIERARCH ESO TEL GEOELEV"),               # ESO La Silla / Paranal
    ("GEOLON", "GEOLAT", "GEOELEV"),
    # FITS WCS Paper III: -L is the LONGITUDE and -B the LATITUDE. Reading
    # them the other way round survives the plausibility guard below for
    # every site with |lon| < 90 deg, so it fails silently.
    ("OBSGEO-L", "OBSGEO-B", "OBSGEO-H"),
    ("LONG-OBS", "LAT-OBS", "ALT-OBS"),
    ("SITELON", "SITELAT", "SITEALT"),          # written by 'normalize'
    ("OBS-LONG", "OBS-LAT", "OBS-ELEV"),
)


def header_site(path):
    """(lon_East, lat, alt, cards) of the observatory as recorded in the
    file's own header, or None.

    Reading the site from the file removes the commonest silent error in
    a barycentric correction: leaving the default observatory in place
    for a spectrum taken somewhere else. Earth rotation alone is +-0.46
    km/s, so a wrong site costs more than everything the RV pipeline
    tries to get right - and nothing in the numbers gives it away.

    FITS headers are searched HDU by HDU, the ASCII '# KEY = VALUE'
    header written by the normalize step the same way. Only fully
    numeric, physically plausible triples are accepted.
    """
    def take(get):
        for lon_k, lat_k, alt_k in SITE_CARD_SETS:
            lon, lat = get(lon_k), get(lat_k)
            if lon is None or lat is None:
                continue
            try:
                lon, lat = float(lon), float(lat)
                alt = float(get(alt_k) or 0.0)
            except (TypeError, ValueError):
                continue
            if abs(lon) > 360.0 or abs(lat) > 90.0 or not -500 < alt < 9000:
                continue
            if lon > 180.0:
                lon -= 360.0
            return lon, lat, alt, f"{lon_k}/{lat_k}/{alt_k}"
        return None

    if not path or not os.path.isfile(path):
        return None
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        try:
            from astropy.io import fits
            with fits.open(path) as hdul:
                for hdu in hdul:
                    found = take(hdu.header.get)
                    if found:
                        return found
        except Exception:
            return None
        return None
    try:
        cards = {}
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if "=" in line:
                    k, v = line.lstrip("# ").split("=", 1)
                    cards[k.strip().upper()] = v.strip()
        return take(cards.get)
    except OSError:
        return None


def neonarval_header_info(path):
    """Epoch and pipeline values of a Neo-Narval (Libre-ESpRIT) file.

    A level-2/3 product is FOUR level-1 exposures combined, and it keeps
    the primary header of the FIRST of them (NeoNarval data format, A.
    Lopez Ariste, sect. 1): DATE-OBS is a bare date, TIME-ISO/TIMEEND and
    EXPTIME describe sub-exposure 1 alone, and DATE-FIT is the file
    CREATION time, not an exposure time. Reading the epoch off those
    cards therefore places the measurement half a sequence too early -
    14 min on a 4x616 s sequence, 3.5 min on a 4x130 s one - which goes
    straight into the orbital phase and, for a short-period binary, into
    the RV itself.

    The exposure-meter table ('Time Photometry' / 'Photometry') does span
    the whole sequence, so it gives both the true time span and a
    flux-weighted mid-time. When it is missing, the FILE1..FILE4 cards
    (+ EXPTIME) bracket the sequence instead, and TIME-ISO/TIMEEND of the
    first sub-exposure is the last resort.

    Returns None unless the file is a NARVAL/Neo-Narval product, else a
    dict with mid_jd, start_jd, end_jd, span_s, nsub, epoch_source, and
    the DRS's own berv [km/s] and btla [JD] (both None when absent).
    """
    if not path or not str(path).lower().endswith((".fits", ".fit",
                                                   ".fits.gz", ".fit.gz")):
        return None
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            inst = str(hdr.get("INSTRUM") or hdr.get("INSTRUME") or "")
            if "NARVAL" not in inst.upper():
                return None
            out = dict(instrument=inst.strip(),
                       berv=_as_float(hdr.get("BERV")),
                       btla=_as_float(hdr.get("BTLA")),
                       exptime=_as_float(hdr.get("EXPTIME")),
                       nsub=sum(1 for k in ("FILE1", "FILE2", "FILE3",
                                            "FILE4") if hdr.get(k)) or 1,
                       start_jd=None, end_jd=None, mid_jd=None,
                       span_s=None, epoch_source=None)

            # 1) the exposure meter: the only table covering the sequence
            for hdu in hdul:
                if not isinstance(hdu, fits.BinTableHDU):
                    continue
                names = {c.name.upper().replace(" ", ""): c.name
                         for c in hdu.columns}
                tname = names.get("TIMEPHOTOMETRY") or names.get("TIME")
                fname = names.get("PHOTOMETRY")
                if tname is None or fname is None:
                    continue
                t = np.asarray(hdu.data[tname], dtype=float).ravel()
                f = np.asarray(hdu.data[fname], dtype=float).ravel()
                good = np.isfinite(t) & (t > 2.4e6) & np.isfinite(f)
                if good.sum() < 4:
                    continue
                t, f = t[good], f[good]
                w = f - f.min()
                out["start_jd"], out["end_jd"] = float(t.min()), float(t.max())
                out["mid_jd"] = (float(np.sum(t * w) / np.sum(w))
                                 if np.sum(w) > 0 else
                                 0.5 * (out["start_jd"] + out["end_jd"]))
                out["epoch_source"] = ("exposure meter, flux-weighted"
                                       if np.sum(w) > 0 else
                                       "exposure meter, mid-point")
                break

            # 2) FILE1..FILE4 (their stamps are creation times, but they
            #    bracket the sequence to about a second)
            if out["mid_jd"] is None:
                start = _as_float(hdr.get("DATE_JUL"))
                if start is None:
                    start = _isot_to_jd(_neonarval_start_isot(hdr))
                last = None
                for k in ("FILE4", "FILE3", "FILE2", "FILE1"):
                    last = _neonarval_name_jd(hdr.get(k))
                    if last is not None:
                        break
                if start is not None and last is not None \
                        and out["exptime"]:
                    out["start_jd"] = start
                    out["end_jd"] = last + out["exptime"] / 86400.0
                    out["mid_jd"] = 0.5 * (out["start_jd"] + out["end_jd"])
                    out["epoch_source"] = "FILE1..FILE4 timestamps + EXPTIME"

            # 3) the first sub-exposure alone (what the old code did)
            if out["mid_jd"] is None:
                start = _isot_to_jd(_neonarval_start_isot(hdr))
                if start is not None and out["exptime"]:
                    out["start_jd"] = start
                    out["end_jd"] = start + out["exptime"] / 86400.0
                    out["mid_jd"] = 0.5 * (out["start_jd"] + out["end_jd"])
                    out["epoch_source"] = ("TIME-ISO + EXPTIME/2 (first "
                                           "sub-exposure only)")
            if out["start_jd"] is not None and out["end_jd"] is not None:
                out["span_s"] = (out["end_jd"] - out["start_jd"]) * 86400.0
            return out
    except Exception:
        return None


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _packed_sexagesimal(value, is_ra):
    """Unpack the OHP/ESO packed coordinate convention to degrees:
    'HHMMSS.ss' for RA (91022.64 -> 09h10m22.64s) and '(+-)DDMMSS.ss'
    for Dec (525210.74 -> +52d52'10.74"). Returns None when the number
    cannot be read that way."""
    v = _as_float(value)
    if v is None:
        return None
    sign = -1.0 if v < 0 else 1.0
    v = abs(v)
    d = int(v // 10000)
    m = int((v - 10000 * d) // 100)
    s = v - 10000 * d - 100 * m
    if not (0 <= m < 60 and 0 <= s < 60):
        return None
    deg = sign * (d + m / 60.0 + s / 3600.0)
    return deg * 15.0 if is_ra else deg


def ohp_header_info(path):
    """Target, epoch and pipeline values of a SOPHIE (OHP DRS) file.

    A SOPHIE product records none of the keywords the generic reader
    looks for: there is no INSTRUME, no DATE-OBS, no RA/DEC and no
    EXPTIME. Everything is under the DRS's own 'OHP ...' prefix, so a
    SOPHIE spectrum otherwise arrives with no coordinates and no epoch
    at all - which silently means v_bary = 0, no BJD and no orbital
    phase.

      OHP TARG ALPHA / DELTA  packed HHMMSS.ss / (+-)DDMMSS.ss
      OHP TARG NAME           target
      OHP OBS DATE START      exposure START (not the mid-time)
      OHP CCD DKTM            exposure time [s]
      OHP DRS BJD             the DRS's own mid-exposure BJD
      OHP DRS BERV            the DRS's own barycentric velocity [km/s]

    Verified against the DRS on six SOPHIE spectra: start + DKTM/2
    reproduces 'OHP DRS BJD' to a constant 70.0 s, which is TDB-UTC
    (69.2 s) - so DKTM is the exposure time, our mid-exposure epoch is
    right, and the DRS's BJD is UTC-based while ours is BJD_TDB.

    Returns None unless the file carries the OHP keywords, else a dict
    with instrument/object/ra/dec/obstime/exptime/drs_bjd/berv.
    """
    if not path or not str(path).lower().endswith((".fits", ".fit",
                                                   ".fits.gz", ".fit.gz")):
        return None
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            if not (hdr.get("OHP DRS VERSION") or hdr.get("OHP TARG NAME")):
                return None
            name = hdr.get("OHP TARG NAME")
            start = hdr.get("OHP OBS DATE START")
            return dict(
                instrument="SOPHIE",
                object=str(name).strip() if name else None,
                ra=_packed_sexagesimal(hdr.get("OHP TARG ALPHA"), True),
                dec=_packed_sexagesimal(hdr.get("OHP TARG DELTA"), False),
                obstime=str(start).strip() if start else None,
                exptime=_as_float(hdr.get("OHP CCD DKTM")),
                drs_bjd=_as_float(hdr.get("OHP DRS BJD")),
                berv=_as_float(hdr.get("OHP DRS BERV")))
    except Exception:
        return None


def _neonarval_start_isot(hdr):
    """'YYYY-MM-DDThh:mm:ss' of the exposure start from the documented
    cards: DATE-OBS (a bare date) + TIME-ISO ('Begin Exposure'), falling
    back to DATE-FIT (the file creation time)."""
    date = str(hdr.get("DATE-OBS") or "").strip()
    ut = str(hdr.get("TIME-ISO") or "").strip()
    if date and "T" in date:
        return date
    if date and ut:
        return f"{date}T{ut}"
    fit = str(hdr.get("DATE-FIT") or "").strip()
    return fit or None


def _neonarval_name_jd(name):
    """JD of the 'NEO_YYYYMMDD_hhmmss' stamp in a Libre-ESpRIT file name."""
    import re
    m = re.search(r"(\d{8})[_-](\d{6})", str(name or ""))
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    return _isot_to_jd(f"{d[:4]}-{d[4:6]}-{d[6:]}T"
                       f"{t[:2]}:{t[2:4]}:{t[4:]}")


def _isot_to_jd(stamp):
    if not stamp:
        return None
    try:
        from astropy.time import Time
        return float(Time(str(stamp), format="isot", scale="utc").jd)
    except Exception:
        return None


def parse_coord(ra, dec):
    """RA/Dec (each) to degrees. Accepts decimal degrees (number or
    string) or sexagesimal 'hh:mm:ss' / 'dd:mm:ss' with ':', space or
    comma separators — the MIDAS comp/bary style '12,50,39.7' (RA in
    hours) and '-03,07,49.8' (Dec in degrees). Returns (ra_deg, dec_deg);
    a None input stays None. astropy is imported only when a sexagesimal
    string actually needs parsing (decimal/None inputs need no astropy)."""
    def one(val, is_ra):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if s == "":
            return None
        try:
            return float(s)          # already decimal degrees
        except ValueError:
            pass
        from astropy.coordinates import Angle
        import astropy.units as u
        s = s.replace(",", ":").replace(" ", ":")
        while "::" in s:
            s = s.replace("::", ":")
        return Angle(s, unit=(u.hourangle if is_ra else u.deg)).deg

    return one(ra, True), one(dec, False)


def ascii_header_info(path):
    """Read OBJECT / DATE-OBS / RA / DEC / EXPTIME from the comment header
    of an ASCII spectrum — lines such as '# RA = 192.66' or
    '# RA = 12:50:39.7' as written by the normalize step. RA/Dec are
    parsed to degrees (sexagesimal accepted). Returns the same dict shape
    as fits_header_info (values None when absent)."""
    info = {"object": None, "obstime": None, "ra": None, "dec": None,
            "exptime": None, "rv_cards": [], "epoch_note": None,
            "neonarval": None, "ohp": None}
    ra_s = dec_s = None
    try:
        with open(path, "r", errors="ignore") as fh:
            for raw in fh:
                s = raw.strip()
                if not s:
                    continue
                if s[0] not in "#;!%":
                    break                    # header ends at the first data row
                body = s.lstrip("#;!% \t")
                if "=" not in body:
                    continue
                key, _, val = body.partition("=")
                key = "".join(ch for ch in key.lower() if ch.isalnum())
                val = val.strip()
                if key == "ra":
                    ra_s = val
                elif key in ("dec", "de"):
                    dec_s = val
                elif key in ("dateobs", "obstime", "date", "mjd", "jd", "bjd"):
                    info["obstime"] = val
                elif key == "object":
                    info["object"] = val
                elif key == "exptime":
                    try:
                        info["exptime"] = float(val)
                    except ValueError:
                        pass
    except OSError:
        return info
    if ra_s is not None or dec_s is not None:
        try:
            info["ra"], info["dec"] = parse_coord(ra_s, dec_s)
        except Exception:
            pass
    return info


def header_info(path):
    """OBJECT/DATE-OBS/RA/DEC/EXPTIME from a spectrum file, dispatching on
    the extension: FITS headers for .fits/.fit(.gz), else the ASCII
    comment header written by the normalize step."""
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz")):
        return fits_header_info(path)
    return ascii_header_info(path)


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


def query_simbad(name):
    """Coordinates + spectral type + V magnitude from SIMBAD.

    Uses the sim-script endpoint with the format
    "%COO(d;A)|%COO(d;D)|%SP(S)|%FLUXLIST(V;F)"; when that fails, falls
    back to Sesame for the coordinates only. Returns
    dict(ra, dec, sptype, vmag) — sptype/vmag may be None.
    Raises ValueError when the name cannot be resolved at all.
    """
    import urllib.parse
    import urllib.request

    info = dict(ra=None, dec=None, sptype=None, vmag=None)
    script = ('output console=off script=off\n'
              'format object "%COO(d;A)|%COO(d;D)|%SP(S)|%FLUXLIST(V;F)"\n'
              f'query id {name}\n')
    url = ("https://simbad.u-strasbg.fr/simbad/sim-script?script="
           + urllib.parse.quote(script))
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode(errors="ignore")
        if "::error" not in text:
            for line in reversed(text.strip().splitlines()):
                parts = line.split("|")
                if len(parts) == 4:
                    info["ra"] = float(parts[0])
                    info["dec"] = float(parts[1])
                    sp = parts[2].strip()
                    info["sptype"] = sp if sp and sp != "~" else None
                    try:
                        info["vmag"] = float(parts[3])
                    except ValueError:
                        pass
                    break
    except Exception:
        pass
    if info["ra"] is None:
        info["ra"], info["dec"] = resolve_target(name)
    return info


def name_variants(name):
    """Plausible spellings of a variable-star designation to try against
    a name-matched database. Mainly the GCVS zero-padding: 'V563 Lyr' is
    also written 'V0563 Lyr' (VarAstro's form) — the number is padded to
    four digits and, conversely, de-padded; a compact 'V563Lyr' also gets
    a spaced form. The original name always comes first; duplicates are
    dropped, order preserved."""
    import re
    name = str(name).strip()
    out = [name]
    m = re.match(r"^([Vv])\s*0*(\d{1,4})\s*[- ]?\s*(.+)$", name)
    if m:
        n, rest = int(m.group(2)), m.group(3).strip()
        out.append(f"V{n:04d} {rest}")     # V0563 Lyr (GCVS zero-padded)
        out.append(f"V{n} {rest}")         # V563 Lyr
    seen, uniq = set(), []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def simbad_identifiers(name):
    """All SIMBAD identifiers (cross-names) of a target — 'V* V563 Lyr',
    'GSC ...', 'TYC ...', 'HD ...', etc. — so a name-matched database that
    does not recognize the entered name can be retried with its aliases.
    Returns a list with the leading 'V* '/'NAME ' catalogue tags stripped,
    most-useful (variable-star / bright-catalogue) names first; empty and
    never raising on failure."""
    import re
    import urllib.parse
    import urllib.request

    script = ('output console=off script=off\n'
              'format object "%IDLIST[%*\\n]"\n'
              f'query id {name}\n')
    url = ("https://simbad.u-strasbg.fr/simbad/sim-script?script="
           + urllib.parse.quote(script))
    ids = []
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode(errors="ignore")
        if "::error" not in text:
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("::") or s.startswith("~"):
                    continue
                s = re.sub(r"^(V\*|SV\*|NAME|\*)\s+", "", s).strip()
                if s and s not in ids and s.lower() != name.strip().lower():
                    ids.append(s)
    except Exception:
        pass

    def priority(idn):
        u = idn.upper()
        if re.match(r"^V\d", u):
            return 0                       # GCVS variable designation
        if u.startswith(("HD ", "HIP ", "BD", "TYC ", "GSC ")):
            return 1                       # bright/astrometric catalogues
        return 2

    return sorted(ids, key=priority)


def query_varastro(name):
    """Ephemeris (T0, P) of an eclipsing binary from the VarAstro database
    (https://var.astro.cz).

    The public pages are used: the site search resolves the star id and
    the star page carries the primary-minimum epoch (CustEpoch) and the
    period (CustPeriod). VarAstro matches by name, and it stores variable
    stars in the GCVS zero-padded form ('V0563 Lyr', not 'V563 Lyr'), so
    the search is retried over name_variants(name) and, if still not
    found, over the target's SIMBAD identifiers (simbad_identifiers) —
    the entered name need not be VarAstro's exact spelling. The values
    are returned in VarAstro's native units: T0 in HJD, period in days;
    truncated epochs are expanded to full HJD (+2400000). Returns
    dict(t0, period, t0_unit, star, url); raises ValueError when nothing
    resolves — the caller then falls back to manual entry.
    """
    import re
    import urllib.parse
    import urllib.request

    base = "https://var.astro.cz"

    def fetch(url):
        req = urllib.request.Request(url, headers={"User-Agent":
                                                   "rv_analysis/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode(errors="ignore")

    def norm_star(s):
        s = re.sub(r"\s+", " ", str(s).upper().strip())
        return re.sub(r"^V0*(\d{1,4})\b",
                      lambda m: f"V{int(m.group(1)):04d}", s)

    def search_star(term):
        """Star id from the VarAstro search results. Prefers the result
        whose label matches the search term (after GCVS zero-padding),
        so a fuzzy hit like 'V0593 Lyr' for 'V563 Lyr' is rejected rather
        than accepted; for a non-variable-star term the first result is
        used, since VarAstro resolved a specific catalogue id."""
        html = fetch(f"{base}/en/Home/Search?term={urllib.parse.quote(term)}")
        pairs = re.findall(r'/en/Stars/(\d+)"[^>]*>\s*([^<]{1,40})', html)
        if not pairs:
            return None
        tn = norm_star(term)
        for sid, label in pairs:
            if norm_star(label) == tn:
                return sid
        if re.match(r"^V\d", tn):          # a V-designation must match exactly
            return None
        return pairs[0][0]

    star_id, tried = None, []
    for term in name_variants(name):
        tried.append(term.lower())
        try:
            star_id = search_star(term)
        except Exception as exc:
            if not tried[:-1]:
                raise ValueError(f"VarAstro search failed: {exc}") from exc
            star_id = None
        if star_id:
            break
    if star_id is None:                    # widen with SIMBAD cross-names
        for ident in simbad_identifiers(name)[:12]:
            for term in name_variants(ident):
                if term.lower() in tried:
                    continue
                tried.append(term.lower())
                try:
                    star_id = search_star(term)
                except Exception:
                    star_id = None
                if star_id:
                    break
            if star_id:
                break
    if star_id is None:
        raise ValueError(f"VarAstro: no star found for '{name}' "
                         f"(tried {len(tried)} name variant(s)/alias(es)).")

    star_url = f"{base}/en/Stars/{star_id}"
    try:
        page = fetch(star_url)
    except Exception as exc:
        raise ValueError(f"VarAstro star page failed: {exc}") from exc

    pm = re.search(r'id="CustPeriod"[^>]*value="([\d.]+)"', page)
    em = re.search(r'id="CustEpoch"[^>]*value="([\d.]+)"', page)
    if not (pm and em):
        raise ValueError(f"VarAstro: no ephemeris elements on {star_url}.")
    t0 = float(em.group(1))
    if t0 < 1000000.0:
        t0 += 2400000.0
    tm = re.search(r"<title>\s*([^<|]+)", page)
    star = tm.group(1).strip() if tm else name
    star = re.sub(r"\s*-\s*VarAstro\s*$", "", star)
    return dict(t0=t0, period=float(pm.group(1)), t0_unit="HJD",
                star=star, url=star_url)


def hjd_to_bjd(hjd, ra_deg, dec_deg, scale="utc"):
    """Convert a heliocentric Julian date (HJD) to BJD_TDB.

    HJD in VarAstro-style minima databases is assumed to be UTC-based; if
    your source uses TT/TDB, change it with the `scale` argument. A wrong
    scale costs at most TDB-UTC (about 69 s today). The epoch is referred
    to the geocenter — the standard convention for an HJD with no
    observatory attached (Eastman, Siverd & Gaudi 2010); the site term is
    below 21 ms. The heliocentric light-travel time is removed to recover
    the geocentric time, then the barycentric light-travel time is added
    and the result is expressed on the TDB scale. Target coordinates are
    required.
    """
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    loc = EarthLocation.from_geocentric(0.0 * u.m, 0.0 * u.m, 0.0 * u.m)
    t = Time(float(hjd), format="jd", scale=scale, location=loc)
    ltt_helio = t.light_travel_time(coord, kind="heliocentric")
    t_geo = t - ltt_helio
    ltt_bary = t_geo.light_travel_time(coord, kind="barycentric")
    return float((t_geo.tdb + ltt_bary).jd)


def _fit_continuum_single(wl, flux, poly_order, iterations,
                          low_clip, high_clip):
    """One iterative-clipping polynomial continuum fit over a segment;
    returns the continuum evaluated on every pixel."""
    good = np.isfinite(wl) & np.isfinite(flux) & (flux > 0)
    if good.sum() < poly_order + 2:
        raise ValueError("Too few valid pixels for continuum fitting.")
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
    return cont


def _fit_continuum_bspline(wl, flux, iterations, low_clip, high_clip,
                           window):
    """Iterative-clipping cubic B-spline continuum fit over a whole
    segment, interior knots every `window` Angstrom — the saphires
    utils.cont_norm approach (bkspace = w_width). Same asymmetric
    clipping as the polynomial fit, so the spline converges onto the
    continuum regions; returns the continuum on every pixel."""
    from scipy.interpolate import LSQUnivariateSpline

    good = np.isfinite(wl) & np.isfinite(flux) & (flux > 0)
    if good.sum() < 8:
        raise ValueError("Too few valid pixels for continuum fitting.")
    mask = good.copy()
    cont = None
    for _ in range(iterations):
        x, keep_first = np.unique(wl[mask], return_index=True)
        y = flux[mask][keep_first]
        knots = np.arange(x.min() + window, x.max() - 0.5 * window, window)
        knots = [t for t in knots
                 if ((x > t - window) & (x <= t)).sum() > 4
                 and ((x > t) & (x <= t + window)).sum() > 4]
        spl = LSQUnivariateSpline(x, y, t=knots, k=3)
        cont = spl(wl)
        resid = flux - cont
        sigma = np.std(resid[mask])
        if sigma <= 0:
            break
        new_mask = good & (resid > -low_clip * sigma) \
            & (resid < high_clip * sigma)
        if new_mask.sum() < 8 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return cont


def normalize_continuum(wl, flux, poly_order=3, iterations=20,
                        low_clip=1.0, high_clip=4.0, window=200.0,
                        method="poly"):
    """Iterative continuum normalization of a raw spectrum (FEROS-style).

    The defaults (polynomial order 3, 20 clipping iterations) are tuned
    for the 500-550 nm region typically used for RV work: over such a
    narrow window the continuum is smooth, so a low-order polynomial
    avoids over-fitting broad features while the extra iterations let the
    clipping converge. Both values remain user-adjustable.

    'All FEROS spectra were normalised to the continuum level iteratively':
    a polynomial is fitted to the spectrum, then points lying more than
    low_clip*sigma BELOW the fit (absorption lines) or high_clip*sigma
    above it (cosmics/emission) are rejected and the fit is repeated, so
    the polynomial converges onto the upper envelope — the continuum.
    The interactive comparison with a synthetic spectrum happens in the
    widget / normalize command, where the result is overplotted on the
    template so the user can tune poly_order and re-run.

    window : continuum window width [Angstrom]. A single low-order
        polynomial can only follow the continuum over a limited range,
        so segments wider than 1.5x this value are normalized piecewise:
        the same iterative fit runs in 50%-overlapping windows and the
        continua are blended with triangular weights. This matches the
        common practice for wide wavelength ranges (saphires
        utils.cont_norm uses a spline with w_width = 200 A; IRAF's
        continuum task fits piecewise splines). Echelle orders narrower
        than ~300 A keep the classic single fit. None/0 forces a single
        fit regardless of width.

    method : continuum model. 'poly' (default) - the windowed
        polynomial described above; 'bspline' - a cubic B-spline with
        interior knots every `window` Angstrom, the saphires
        utils.cont_norm model. Both use the same iterative asymmetric
        clipping, so the fit always references the continuum regions.

    Returns (normalized_flux, continuum). Pixels where the continuum is
    not positive come back as NaN.
    """
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if method == "bspline":
        cont = _fit_continuum_bspline(wl, flux, iterations,
                                      low_clip, high_clip,
                                      window or 200.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(cont > 0, flux / cont, np.nan)
        return norm, cont

    span = np.nanmax(wl) - np.nanmin(wl)
    if window and span > 1.5 * window:
        half = window / 2.0
        cont_sum = np.zeros(wl.size)
        wgt_sum = np.zeros(wl.size)
        for lo in np.arange(np.nanmin(wl), np.nanmax(wl), half):
            m = (wl >= lo) & (wl <= lo + window)
            if m.sum() < max(poly_order + 2, 20):
                continue
            try:
                c = _fit_continuum_single(wl[m], flux[m], poly_order,
                                          iterations, low_clip, high_clip)
            except ValueError:
                continue
            center = lo + half
            wgt = np.maximum(1.0 - np.abs(wl[m] - center) / half, 1e-3)
            cont_sum[m] += wgt * c
            wgt_sum[m] += wgt
        if not np.any(wgt_sum > 0):
            raise ValueError("Too few valid pixels for continuum fitting.")
        cont = np.full(wl.size, np.nan)
        covered = wgt_sum > 0
        cont[covered] = cont_sum[covered] / wgt_sum[covered]
    else:
        cont = _fit_continuum_single(wl, flux, poly_order, iterations,
                                     low_clip, high_clip)

    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(cont > 0, flux / cont, np.nan)
    return norm, cont


def normalize_spectrum_file(path, fmt="auto", poly_order=3, iterations=20,
                            low_clip=1.0, high_clip=4.0, window=200.0,
                            method="poly"):
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
                                       low_clip, high_clip, window=window,
                                       method=method)
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
    """Load the synthetic/template spectrum from file or via expecto/PHOENIX.

    ASCII files (.prf/.obs/...) and FITS templates (e.g. an observed
    standard used as template, saphires-style) are both accepted;
    multi-order FITS templates are merged. The template must be
    continuum-normalized, like the observed spectrum.
    """
    if args.template:
        if str(args.template).lower().endswith((".fits", ".fit", ".fits.gz")):
            orders = read_fits_spectrum(args.template)
            wl = np.concatenate([w for w, _ in orders])
            fx = np.concatenate([f for _, f in orders])
            srt = np.argsort(wl)
            return wl[srt], fx[srt]
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


def parse_time_input(value):
    """Observation time as an astropy Time: accepts an ISOT string
    ('2024-12-03T02:30:00'), an ISO date with or without time
    ('1998-10-16', '1998-10-16 20:07:42'), an old-standard FITS date
    'DD/MM/YY' ('16/10/98', optionally followed by a UT time; years
    >= 50 are taken as 19xx) or a JD/BJD number ('2453254.847090365').

    A numeric value is treated as a BJD and, by definition of the BJD,
    carried on the TDB scale: Time(bjd, scale='tdb', format='jd'). A
    calendar stamp is a clock reading and stays on the UTC scale.
    """
    import re
    from astropy.time import Time
    s = str(value).strip()
    try:
        jd = float(s)
    except ValueError:
        pass
    else:
        return Time(jd, format="jd", scale="tdb"), True
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})(?:[T\s]+(.+))?$", s)
    if m:
        day, mon, year = (int(m.group(g)) for g in (1, 2, 3))
        if year < 100:
            year += 1900 if year >= 50 else 2000
        s = f"{year:04d}-{mon:02d}-{day:02d}"
        if m.group(4):
            s += f" {m.group(4).strip()}"
    try:
        return Time(s.replace("T", " "), format="iso", scale="utc"), False
    except ValueError:
        pass
    try:
        return Time(s, format="isot", scale="utc"), False
    except ValueError as exc:
        raise ValueError(
            f"Unrecognized observation time '{value}'. Accepted: ISOT "
            "(2024-12-03T02:30:00), ISO date [+ time] (1998-10-16 "
            "20:07:42), old FITS date DD/MM/YY (16/10/98 [+ UT time]), "
            "or a JD/BJD number.") from exc


def orbital_phase(bjd, t0, period):
    """Orbital phase from the ephemeris BJD_MinI = t0 + period * E.
    Primary eclipse is at phase 0.0, secondary near 0.5."""
    return float(((bjd - t0) / period) % 1.0)


def barycentric_correction(ra_deg, dec_deg, obstime, site):
    """Barycentric velocity correction [km/s]; combine it with the
    measured RV via apply_barycentric. `obstime` may be an ISOT string
    or a JD/BJD number."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    loc = earth_location(site)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    t, _ = parse_time_input(obstime)
    return coord.radial_velocity_correction(obstime=t, location=loc).to(u.km / u.s).value


def apply_barycentric(rv_measured, vbary):
    """Barycentric-correct a measured RV [km/s]:

        RV = RV_measured + v_bary + RV_measured * v_bary / c

    (Wright & Eastman 2014; the saphires brvc practice and the form
    recommended by astropy's radial_velocity_correction). The
    relativistic cross term matters at the ~10 m/s level for
    |RV| ~ 100 km/s and simple addition is inconsistent for large RVs.
    """
    return rv_measured + vbary + rv_measured * vbary / C_KMS


BARY_CROSSCHECK_TOL = 0.05      # km/s: louder than any real difference


def crosscheck_pipeline_bary(path, vbary, bjd, berv_hdr=None):
    """Compare our barycentric velocity and BJD with the values the
    instrument pipeline wrote into the file, when it wrote any.

    A pipeline BERV is an independent computation of the same quantity:
    if the two agree to a few m/s, the coordinates, the epoch and the
    observatory are all right at once. If they do not, something is
    wrong that no other output would reveal - a wrong site is worth up
    to 0.46 km/s, a wrong epoch a few m/s per minute, wrong coordinates
    up to 0.03 km/s per arcmin. This check is the reason the site and
    epoch fixes above can be trusted.

    Some pipelines store the correction with the opposite sign (the
    velocity to ADD vs the velocity of the observer): a match after a
    sign flip is reported as such rather than as a disagreement.

    Returns dict(messages=[...], summary=str or None).
    """
    msgs, bits = [], []
    if berv_hdr is not None and np.isfinite(vbary):
        d, dflip = vbary - berv_hdr, vbary + berv_hdr
        if abs(d) <= BARY_CROSSCHECK_TOL:
            msgs.append(f"  cross-check: pipeline BERV = {berv_hdr:+.4f} "
                        f"km/s, ours {vbary:+.4f} ({d * 1000:+.0f} m/s) - OK")
            bits.append(f"BERV agrees to {abs(d) * 1000:.0f} m/s")
        elif abs(dflip) <= BARY_CROSSCHECK_TOL:
            msgs.append(f"  cross-check: pipeline BERV = {berv_hdr:+.4f} "
                        f"km/s matches ours ({vbary:+.4f}) with the "
                        "OPPOSITE sign convention - values agree, "
                        "conventions differ")
            bits.append("BERV agrees up to the sign convention")
        else:
            msgs.append(
                f"WARNING: our barycentric velocity ({vbary:+.4f} km/s) "
                f"differs from the pipeline's own BERV ({berv_hdr:+.4f} "
                f"km/s) by {d:+.4f} km/s. Check, in this order: the "
                "observatory (--site; a wrong one costs up to 0.46 "
                "km/s), the coordinates (--ra/--dec vs the header) and "
                "the epoch (--obstime).")
            bits.append(f"BERV differs by {d:+.3f} km/s")

    neo = neonarval_header_info(path)
    if neo and neo.get("btla") and bjd is not None:
        dt = (bjd - neo["btla"]) * 1440.0            # minutes
        span = (neo["span_s"] or 0.0) / 60.0
        # BTLA is the barycentric time of the START of the sequence, so
        # our mid-exposure BJD must fall inside the sequence.
        if -0.5 <= dt <= span + 0.5:
            msgs.append(f"  cross-check: BTLA (header, sequence start) = "
                        f"{neo['btla']:.6f}; our mid-exposure BJD is "
                        f"{dt:+.1f} min later, inside the {span:.1f} min "
                        "sequence - OK")
            bits.append(f"BJD is {dt:+.1f} min after BTLA")
        else:
            msgs.append(
                f"WARNING: our BJD ({bjd:.6f}) is {dt:+.1f} min from the "
                f"header BTLA ({neo['btla']:.6f}), which is outside the "
                f"{span:.1f} min exposure sequence. One of the two times "
                "is wrong.")
            bits.append(f"BJD is {dt:+.1f} min from BTLA")

    ohp = ohp_header_info(path)
    if ohp and ohp.get("drs_bjd") and bjd is not None:
        # 'OHP DRS BJD' is the DRS's own mid-exposure BJD, but on the UTC
        # scale; ours is BJD_TDB. The two therefore differ by TDB-UTC
        # (69.2 s in 2025) BY CONSTRUCTION, and it is the residual after
        # removing that offset which tests the epoch.
        try:
            from astropy.time import Time
            t = Time(float(ohp["drs_bjd"]), format="jd", scale="utc")
            scale_s = float((t.tdb.jd - t.jd) * 86400.0)
        except Exception:
            scale_s = 69.2
        dt = (bjd - ohp["drs_bjd"]) * 86400.0
        resid = dt - scale_s
        if abs(resid) <= 5.0:
            msgs.append(f"  cross-check: DRS BJD (UTC-based) = "
                        f"{ohp['drs_bjd']:.6f}; ours (TDB) is {dt:+.1f} s "
                        f"later, of which {scale_s:.1f} s is TDB-UTC - "
                        f"agrees to {resid:+.1f} s")
            bits.append(f"BJD agrees with the DRS to {abs(resid):.1f} s")
        else:
            msgs.append(
                f"WARNING: our BJD_TDB ({bjd:.6f}) is {resid:+.1f} s from "
                f"the DRS's own BJD ({ohp['drs_bjd']:.6f}) after removing "
                f"the {scale_s:.1f} s TDB-UTC scale difference. Check the "
                "exposure time (OHP CCD DKTM) and the epoch.")
            bits.append(f"BJD differs from the DRS by {resid:+.1f} s")
    return dict(messages=msgs, summary="; ".join(bits) or None)


def get_target_context(args):
    """Resolve target info; return dict(vbary, bjd, phase).

    Coordinate priority: explicit --ra/--dec, then SIMBAD lookup of
    --object, then the FITS header of the spectrum. The observation time
    comes from --obstime or the FITS header (DATE-OBS) and may be given
    directly as a BJD number — all analysis times are reported in BJD, not
    local/UTC calendar time. A numeric time is taken to be the BJD itself
    (as in the reference document); an ISOT stamp is converted to
    mid-exposure BJD_TDB. The orbital phase is computed when an ephemeris
    (--t0, --period) is available.
    """
    ra, dec = parse_coord(args.ra, args.dec)
    obstime = args.obstime
    exptime = None
    site, site_note = observing_site(args)

    # Frame guard: block the run when the spectrum's own header says
    # its wavelengths are already barycentric but the correction is
    # about to be applied a second time (--force-bary overrides).
    try:
        frame_chk = barycentric_frame_check(args.spectrum)
    except Exception:
        frame_chk = None
    frame_note = None
    if frame_chk is not None and (frame_chk["frame"] is not None
                                  or frame_chk["verdict"] != "unknown"
                                  or frame_chk["berv"] is not None):
        frame_note = f"header check: {frame_chk['verdict']}"
        if frame_chk["frame"]:
            frame_note += f" ({frame_chk['frame']})"
        if frame_chk["berv"] is not None:
            frame_note += (f", header v_bary = "
                           f"{frame_chk['berv']:+.4f} km/s")
    if (frame_chk is not None and frame_chk["verdict"] == "corrected"
            and not getattr(args, "no_bary", False)):
        if getattr(args, "force_bary", False):
            print("WARNING: the spectrum header says the wavelength "
                  "scale is already barycentric - applying the "
                  "correction again would double it. Proceeding anyway "
                  "(--force-bary).")
        else:
            raise ValueError(
                "the spectrum header says its wavelength scale is "
                "ALREADY barycentric - applying the correction again "
                "would double it. Disable 'Apply barycentric "
                "correction' / use --no-bary (v_bary is still computed "
                "and written to the result file as a note), or override "
                "with --force-bary. Evidence: 'barycheck' command.")

    if ra is None and getattr(args, "object", None):
        try:
            sim = query_simbad(args.object)
            ra, dec = sim["ra"], sim["dec"]
            extra = ""
            if sim["sptype"]:
                extra += f", SpT = {sim['sptype']}"
            if sim["vmag"] is not None:
                extra += f", V = {sim['vmag']:.2f}"
            print(f"SIMBAD: '{args.object}' -> RA = {ra:.5f} deg, "
                  f"Dec = {dec:.5f} deg{extra}")
        except ValueError as exc:
            print(f"Warning: {exc}\n  -> give coordinates manually "
                  "with --ra/--dec.")

    # The header is read even when --ra/--dec AND --obstime were all given
    # explicitly: EXPTIME is recorded nowhere else, and without it the BJD
    # would be the START of the exposure instead of its middle (half of a
    # 616 s exposure is 5 minutes of orbital phase).
    hdr = None
    if getattr(args, "spectrum", None):
        try:
            hdr = header_info(args.spectrum)
        except Exception:
            hdr = None
    if hdr is not None:
        src = ("FITS header" if str(args.spectrum).lower().endswith(
            (".fits", ".fit", ".fits.gz")) else "file header")
        exptime = hdr["exptime"]
        if ra is None and hdr["ra"] is not None:
            ra, dec = hdr["ra"], hdr["dec"]
            print(f"{src}: RA = {ra:.5f} deg, Dec = {dec:.5f} deg")
        if obstime is None and hdr["obstime"]:
            obstime = hdr["obstime"]
            print(f"{src}: DATE-OBS = {obstime}")
            if hdr.get("epoch_note"):
                print(f"  {hdr['epoch_note']}")

    # The BJD and barycentric computations need astropy (and, for an
    # astropy site name, its online registry). A failure there must not
    # abort the RV measurement itself, which is pure numpy/scipy: warn
    # and continue with no correction.
    bjd = None
    if obstime is not None:
        try:
            _, is_jd = parse_time_input(obstime)
            if is_jd:
                bjd = float(str(obstime).strip())
            elif ra is not None and dec is not None:
                if site_note:
                    print(site_note)
                    site_note = None
                bjd = compute_bjd(obstime, ra, dec, site,
                                  exptime=exptime)
            if bjd is not None:
                print(f"BJD = {bjd:.6f}")
        except Exception as exc:
            print(f"Warning: BJD computation skipped ({exc}).")

    vbary = 0.0
    if ra is not None and dec is not None and obstime is not None:
        try:
            if site_note:
                print(site_note)
                site_note = None
            vbary = barycentric_correction(ra, dec, obstime, site)
            if getattr(args, "no_bary", False):
                print(f"Barycentric correction: {vbary:+.4f} km/s "
                      "(computed, NOT applied to the reported RVs)")
            else:
                print(f"Barycentric correction: {vbary:+.4f} km/s "
                      "(added to the RV)")
        except Exception as exc:
            print(f"Warning: barycentric correction skipped ({exc}).")
            vbary = 0.0
        cross = crosscheck_pipeline_bary(
            args.spectrum, vbary, bjd,
            berv_hdr=(frame_chk or {}).get("berv"))
        for line in cross["messages"]:
            print(line)
        if frame_note and cross["summary"]:
            frame_note += f"; {cross['summary']}"
    elif getattr(args, "object", None) or args.ra is not None or obstime:
        print("Note: barycentric correction skipped "
              "(needs coordinates AND observation time).")

    phase = None
    t0 = getattr(args, "t0", None)
    period = getattr(args, "period", None)
    t0_is_hjd = False
    if t0 is None and getattr(args, "varastro", False) \
            and getattr(args, "object", None):
        try:
            va = query_varastro(args.object)
            t0, period = va["t0"], va["period"]
            t0_is_hjd = True
            print(f"VarAstro: {va['star']} -> T0 = {t0:.5f} HJD, "
                  f"P = {period:.7f} d  ({va['url']})")
        except ValueError as exc:
            print(f"Warning: {exc}\n  -> give the ephemeris manually "
                  "with --t0/--period.")
    if t0 is not None and getattr(args, "t0_to_bjd", False):
        if ra is not None and dec is not None:
            t0_bjd = hjd_to_bjd(t0, ra, dec)
            print(f"Ephemeris T0 converted: HJD {t0:.6f} -> "
                  f"BJD {t0_bjd:.6f} (used for the analysis)")
            t0 = t0_bjd
        else:
            print("Note: HJD -> BJD conversion of T0 skipped "
                  "(needs target coordinates).")
    elif t0_is_hjd:
        print("Note: T0 is kept in HJD (VarAstro native unit); "
              "use --t0-to-bjd to convert it to BJD.")
    if bjd is not None and t0 is not None and period:
        phase = orbital_phase(bjd, t0, period)
        which = ("primary eclipse" if phase < 0.05 or phase > 0.95 else
                 "secondary eclipse" if abs(phase - 0.5) < 0.05 else
                 "out of eclipse")
        print(f"Orbital phase = {phase:.4f} ({which})")

    return dict(vbary=vbary, bjd=bjd, phase=phase,
                frame_note=frame_note)


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


def calculate_ccf_fft(orders, tpl_wl, tpl_flux, dv_kms, max_shift_kms,
                      apodize=0.1, highpass_kms=2000.0, lowpass_kms=None):
    """Cross-correlation of spectrum against template by FFT — the
    Tonry & Davis (1979) construction, and what ESO-MIDAS `xcorr` does.

    On a log-lambda grid a Doppler shift is an integer pixel shift, so
    the correlation at *every* velocity is one transform pair instead of
    one resampling per trial velocity: O(N log N) for the whole profile
    against O(N_rv x N) for the direct sum. The full +-max_shift range
    comes out in a single pass, which is why an FFT correlation is
    normally computed over the entire velocity span at once.

    The three steps that make it correct rather than merely fast:

      * **apodization** — a cosine bell over the outer `apodize`
        fraction of each end. The DFT treats the spectrum as periodic;
        without a taper the discontinuity between the two ends injects
        power at every wavenumber and rings across the whole profile.
      * **zero padding** to at least 2N, so the circular correlation of
        the DFT equals the linear correlation that is wanted.
      * **bandpass** — a zero-phase cosine bell removing wavenumbers
        below the continuum scale (`highpass_kms`, residual
        normalization waves) and above the resolution element
        (`lowpass_kms`, pixel noise). Being real and even it cannot
        move the peak.

    Returns (velocity, ccf_total, ccf_per_order); the correlation is
    normalized so the peak is a correlation coefficient.
    """
    wl_lo = max(min(w.min() for w, _ in orders), tpl_wl.min())
    wl_hi = min(max(w.max() for w, _ in orders), tpl_wl.max())
    if not wl_hi > wl_lo:
        raise RuntimeError("The spectrum and the template do not overlap "
                           "in wavelength; no CCF can be computed.")
    grid = log_wave_grid(wl_lo, wl_hi, dv_kms)
    n = grid.size
    if n < 64:
        raise RuntimeError(f"Only {n} pixels of overlap at dv = "
                           f"{dv_kms:g} km/s — too few for an FFT CCF.")

    taper = np.ones(n)
    edge = max(int(apodize * n), 1)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
    taper[:edge] = ramp
    taper[n - edge:] = ramp[::-1]

    span = n * dv_kms
    if lowpass_kms is None:
        lowpass_kms = 2.0 * dv_kms
    k1, k2 = span / (4.0 * highpass_kms), span / highpass_kms
    k3, k4 = span / (2.0 * lowpass_kms), span / lowpass_kms

    try:
        from scipy.fft import next_fast_len
        nfft = next_fast_len(2 * n)
    except ImportError:
        nfft = 1 << int(np.ceil(np.log2(2 * n)))

    def prepared(values):
        v = np.asarray(values, dtype=float)
        v = (v - v.mean()) * taper
        spec = np.fft.rfft(v, nfft)
        spec *= cosine_bell(spec.size, k1 * nfft / n, k2 * nfft / n,
                            k3 * nfft / n, k4 * nfft / n)
        return spec

    tpl = np.interp(grid, tpl_wl, tpl_flux)
    ftpl = prepared(tpl)
    tpl_power = float(np.sum(np.abs(ftpl) ** 2))

    lag = np.arange(nfft)
    lag = np.where(lag > nfft // 2, lag - nfft, lag)
    velocity = lag * dv_kms
    keep = np.abs(velocity) <= max_shift_kms
    order_sort = np.argsort(velocity[keep])

    per_order, total = [], np.zeros(int(keep.sum()))
    for k, (wl, fx) in enumerate(orders):
        good = np.isfinite(wl) & np.isfinite(fx)
        if good.sum() < 10:
            continue
        w, f = wl[good], fx[good]
        inside = (grid >= w.min()) & (grid <= w.max())
        if inside.sum() < 64:
            continue
        sig = np.full(n, np.nan)
        sig[inside] = np.interp(grid[inside], w, f)
        sig[~inside] = np.nanmean(sig[inside])
        fsig = prepared(sig)
        norm = np.sqrt(float(np.sum(np.abs(fsig) ** 2)) * tpl_power)
        cc = np.fft.irfft(fsig * np.conj(ftpl), nfft) * nfft / max(norm, 1e-30)
        per_order.append(cc[keep][order_sort])
        total += per_order[-1]
        print(f"  order {k + 1}/{len(orders)} correlated by FFT",
              end="\r")
    print()
    if not per_order:
        raise RuntimeError("No order overlaps the template well enough "
                           "for an FFT CCF.")
    return velocity[keep][order_sort], total / len(per_order), per_order


# Hydrogen Balmer and strong Paschen lines [A]. In hot (OBA) stars these
# pressure-broadened wings dominate a full-spectrum correlation while
# their cores are often filled by emission, so the CCF line mask skips
# them by default and the RV comes from the He/metal line cores.
HYDROGEN_WL = [6562.797, 4861.325, 4340.472, 4101.734, 3970.075,
               3889.064, 3835.397, 3797.909, 3770.633, 3750.151,
               8598.392, 8750.472, 8862.783, 9014.910, 9229.014,
               9545.969]


def build_ccf_mask(tpl_wl, tpl_flux, min_depth=0.05, exclude_balmer=True,
                   balmer_half=15.0):
    """Weighted CCF line mask derived from the template (Baranne et al.
    1996; Pepe et al. 2002 — the CCF construction followed by Pino et
    al. 2018, A&A 619, A3): the mask entries are the local minima of the
    continuum-normalized template deeper than `min_depth`, each refined
    to sub-pixel precision with a parabola through the three lowest
    points and weighted by its line depth, so deep lines contribute more
    RV information.

    exclude_balmer drops every mask line within `balmer_half` [A] of the
    hydrogen series (HYDROGEN_WL) — the hot-star practice: RVs come from
    the He/metal lines, not from the broad H wings or their frequently
    emission-filled cores. A clearly un-normalized template (e.g. a raw
    PHOENIX flux spectrum) is normalized first with a coarse
    upper-envelope continuum so the depths stay meaningful.

    Returns (line_wl, line_weight) arrays sorted in wavelength.
    """
    good = np.isfinite(tpl_wl) & np.isfinite(tpl_flux)
    wl, fx = np.asarray(tpl_wl)[good], np.asarray(tpl_flux)[good]

    med = float(np.median(fx))
    if not 0.5 <= med <= 1.5:
        nbin = max(wl.size // 200, 10)
        edges = np.linspace(wl.min(), wl.max(), nbin + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        env = np.array([np.percentile(fx[(wl >= a) & (wl < b)], 95)
                        if ((wl >= a) & (wl < b)).sum() > 5 else np.nan
                        for a, b in zip(edges[:-1], edges[1:])])
        ok = np.isfinite(env) & (env > 0)
        fx = fx / np.interp(wl, centers[ok], env[ok])

    imin = np.flatnonzero((fx[1:-1] < fx[:-2]) & (fx[1:-1] <= fx[2:])) + 1
    depth = 1.0 - fx[imin]
    keep = depth >= min_depth
    imin, depth = imin[keep], depth[keep]

    line_wl = wl[imin].astype(float)
    for j, i in enumerate(imin):
        y0, y1, y2 = fx[i - 1], fx[i], fx[i + 1]
        den = y0 - 2.0 * y1 + y2
        if den > 0:
            off = 0.5 * (y0 - y2) / den
            if abs(off) < 1.0:
                line_wl[j] = wl[i] + off * 0.5 * (wl[i + 1] - wl[i - 1])

    if exclude_balmer and line_wl.size:
        far = np.ones(line_wl.size, dtype=bool)
        for h in HYDROGEN_WL:
            far &= np.abs(line_wl - h) > balmer_half
        line_wl, depth = line_wl[far], depth[far]

    if line_wl.size == 0:
        raise ValueError("No usable mask line in the template (check the "
                         "normalization, or lower --mask-depth).")
    return line_wl, depth


def calculate_ccf_mask(spec_wl, spec_flux, line_wl, line_w, rv_grid):
    """Weighted line-mask CCF contributions of one spectrum/order:
    CCF(v) = sum_i w_i d(lambda_i (1+v/c)) with d = 1 - flux, so the
    profile peaks where the observed line cores align with the mask.
    Only mask lines whose shifted position stays inside the data for
    every velocity of the grid are used — the line set is then identical
    across the grid and the CCF has no edge steps.

    Returns (weighted_depth_sum[v], weight_sum) for combining orders.
    """
    lo = spec_wl.min() / (1.0 + rv_grid.min() / C_KMS)
    hi = spec_wl.max() / (1.0 + rv_grid.max() / C_KMS)
    use = (line_wl >= lo) & (line_wl <= hi)
    if not use.any():
        return None, 0.0
    lwl, w = line_wl[use], line_w[use]
    depth = 1.0 - spec_flux
    num = np.empty(rv_grid.size)
    for i, rv in enumerate(rv_grid):
        num[i] = np.dot(w, np.interp(doppler_shift(lwl, rv),
                                     spec_wl, depth))
    return num, float(w.sum())


def fit_ccf_peak(rv_grid, ccf):
    """CCF peak measurement following Pino et al. (2018): a Gaussian on
    a linear baseline, f(v) = C + B v + A exp(-(v-v0)^2/2 sigma^2),
    fitted over a window of +-2 FWHM around the maximum. The local
    linear term and the restricted window keep slopes from residual
    broad features (hot-star H wings) from pulling the centre.

    Returns ([component dict], model descriptor) like fit_peak_with_base.
    """
    rv_grid = np.asarray(rv_grid, dtype=float)
    ccf = np.asarray(ccf, dtype=float)
    step = float(np.median(np.diff(rv_grid)))
    i0 = int(np.nanargmax(ccf))
    base = float(np.nanmedian(ccf))
    half = base + 0.5 * (ccf[i0] - base)
    j = i0
    while j + 1 < ccf.size and ccf[j + 1] > half:
        j += 1
    k = i0
    while k - 1 >= 0 and ccf[k - 1] > half:
        k -= 1
    w50 = max(rv_grid[j] - rv_grid[k], 3.0 * step)

    sel = np.abs(rv_grid - rv_grid[i0]) <= max(2.0 * w50, 7.5 * step)
    p0 = [ccf[i0] - base, rv_grid[i0], max(w50 / 2.35482, step), 0.0, base]
    popt, pcov = curve_fit(gauss_lin, rv_grid[sel], ccf[sel], p0=p0,
                           maxfev=20000)
    err = np.sqrt(np.diag(pcov))
    comp = dict(rv=float(popt[1]), rv_err=float(err[1]),
                amp=float(popt[0]), sigma=abs(float(popt[2])),
                kind="gauss_lin")
    return [comp], dict(kind="gauss_lin", popt=popt)


def profile_peak_snr(x, profile, comps):
    """Significance of a fitted peak: component amplitude over the robust
    (MAD) scatter of the profile away from every fitted component. A low
    value (< ~4) means the 'peak' is not distinguishable from the noise
    floor — the fit found something, but not a meaningful RV."""
    x = np.asarray(x, dtype=float)
    profile = np.asarray(profile, dtype=float)
    away = np.ones(x.size, dtype=bool)
    for c in comps:
        away &= np.abs(x - c["rv"]) > 3.0 * max(c.get("sigma", 0.0), 1.0)
    if away.sum() < 10:
        return np.inf
    wing = profile[away]
    noise = 1.4826 * np.median(np.abs(wing - np.median(wing)))
    peak = max(abs(float(c["amp"])) for c in comps)
    return peak / max(noise, 1e-12)


LOW_SNR_WARNING = (
    "WARNING: the fitted peak is only {snr:.1f}x the profile noise - "
    "this result is probably NOT meaningful. Check that the template "
    "wavelength range overlaps a CLEAN part of the observed spectrum "
    "(real lines visible, no merge/normalization artifacts) and matches "
    "the star's spectral type; restrict --wave-min/--wave-max to such a "
    "region if needed.")

LOW_DATA_SNR_NOTE = (
    "Note: continuum S/N ~ {snr:.0f} is low. Katz et al. (2025) find "
    "low-S/N spectra to be the dominant source of spurious RVs - treat "
    "this result with care.")


def estimate_snr(flux):
    """Robust per-pixel continuum S/N of a normalized spectrum. The MAD
    of second differences over runs of near-continuum pixels
    (|flux - 1| < 0.05) measures only the high-frequency noise — lines
    and slow slopes cancel out of a second difference — and sqrt(6)
    converts it to a per-pixel sigma. Reported with the results because
    low S/N is the dominant source of spurious RV measurements (Katz et
    al. 2025, A&A 704, A294). Returns inf for noise-free input."""
    f = np.asarray(flux, dtype=float)
    f = f[np.isfinite(f)]
    if f.size < 30:
        return np.inf
    near = np.abs(f - 1.0) < 0.05
    ok = near[:-2] & near[1:-1] & near[2:]
    d2 = np.diff(f, 2)
    d2 = d2[ok] if ok.sum() >= 30 else d2
    sigma = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0)
    return float(1.0 / sigma) if sigma > 1e-9 else np.inf


# Coarse stage of the automatic RV search (Katz et al. 2025, A&A 704,
# A294, SOPHIE procedure): scan +-900 km/s in 10 km/s steps to locate
# the peak, then lay the fine grid around it. A fixed narrow default
# window would silently miss genuinely fast-moving stars.
CCF_SEARCH_HALF = 900.0
CCF_SEARCH_STEP = 10.0


def run_ccf(spectrum_orders, tpl_wl, tpl_flux, rv_min=None, rv_max=None,
            rv_step=0.5, mode="mask", mask_depth=0.05, keep_balmer=False,
            fft_dv=None, fft_lowpass=None):
    """Run the CCF over all echelle orders, combine, and fit the peak.

    RV search: when rv_min/rv_max are not given, the search is
    two-stage, following the SOPHIE off-line procedure of Katz et al.
    (2025, A&A 704, A294): a coarse scan over +-900 km/s in 10 km/s
    steps locates the peak — so genuinely fast-moving stars are never
    outside the window — and the fine rv_step grid is then laid over
    peak +- 2.5 FWHM (at least +-25 km/s) for the actual fit. Explicit
    limits are used as-is in a single fine pass.

    mode 'mask' (default): weighted line-mask CCF — the construction of
    Baranne et al. (1996) / Pepe et al. (2002) as applied by Pino et al.
    (2018, A&A 619, A3). The template only provides the line positions
    and depth weights; the correlation samples the observed line cores,
    so the continuum and the broad hydrogen wings drop out of the
    problem. This is what makes the CCF reliable for very hot (OBA)
    stars, where the full-spectrum correlation is dominated by the
    pressure-broadened Balmer wings and biased by emission-filled H
    cores; the hydrogen series is excluded from the mask unless
    keep_balmer=True. The combined CCF is the weight-averaged line
    depth, and the peak is measured with a Gaussian on a linear
    baseline over a +-2 FWHM window (fit_ccf_peak).

    mode 'template': the previous full-spectrum cross-covariance (the
    workshop approach), kept for line-rich cool stars and comparison.

    Unphysical pixels (normalized flux > 1.2 or < 0: cosmics, emission
    spikes) are interpolated over first, as before the BF SVD.

    Returns: dict(rv, rv_err, amp, sigma, rv_grid, ccf_total,
    ccf_orders, popt, mode, peak_snr, search[, n_lines])
    """
    auto = rv_min is None or rv_max is None
    span = CCF_SEARCH_HALF if auto else max(abs(rv_min), abs(rv_max))

    if mode == "mask":
        line_wl, line_w = build_ccf_mask(tpl_wl, tpl_flux,
                                         min_depth=mask_depth,
                                         exclude_balmer=not keep_balmer)
        print(f"CCF line mask: {line_wl.size} template lines "
              f"(depth >= {mask_depth:g}"
              + (", hydrogen series excluded" if not keep_balmer
                 else ", hydrogen series kept") + ")")
        orders_clean = []
        for wl, fx in spectrum_orders:
            good = np.isfinite(wl) & np.isfinite(fx)
            wl, fx = wl[good], fx[good]
            if wl.size < 10:
                continue
            bad = (fx > 1.2) | (fx < 0.0)
            if 0 < bad.sum() < 0.5 * fx.size:
                fx = fx.copy()
                fx[bad] = np.interp(wl[bad], wl[~bad], fx[~bad])
            orders_clean.append((wl, fx))

        def ccf_on(grid):
            per_order, num_total, w_total = [], np.zeros(grid.size), 0.0
            for k, (wl, fx) in enumerate(orders_clean):
                num, wsum = calculate_ccf_mask(wl, fx, line_wl, line_w,
                                               grid)
                if num is None or wsum <= 0:
                    continue
                per_order.append(num / wsum)
                num_total += num
                w_total += wsum
                print(f"  order {k + 1}/{len(orders_clean)} done",
                      end="\r")
            print()
            if w_total <= 0:
                raise RuntimeError(
                    "No mask line falls inside the data for the whole "
                    "RV grid; narrow --rv-min/--rv-max or check the "
                    "wavelength overlap with the template.")
            return num_total / w_total, per_order
    elif mode == "fft":
        # One transform pair gives the correlation at every velocity, so
        # the profile is computed once over the full search span and the
        # two-stage search only ever reads slices of it.
        dv_fft = fft_dv if fft_dv else max(rv_step, 0.5)
        print(f"FFT cross-correlation on a log-lambda grid, "
              f"dv = {dv_fft:g} km/s over ±{span:g} km/s")
        v_fft, cc_fft, cc_fft_orders = calculate_ccf_fft(
            spectrum_orders, tpl_wl, tpl_flux, dv_fft, span,
            lowpass_kms=fft_lowpass)

        def ccf_on(grid):
            total = np.interp(grid, v_fft, cc_fft)
            per_order = [np.interp(grid, v_fft, c) for c in cc_fft_orders]
            peak = np.nanmax(total)
            if peak > 0:
                total = total / peak
                per_order = [c / peak for c in per_order]
            return total, per_order
    else:
        tpl_norm = tpl_flux / np.nanmax(tpl_flux)
        max_shift = span / C_KMS
        lo_wl = tpl_wl.min() * (1.0 + max_shift)
        hi_wl = tpl_wl.max() * (1.0 - max_shift)
        orders_clean = []
        for wl, fx in spectrum_orders:
            good = np.isfinite(wl) & np.isfinite(fx)
            wl, fx = wl[good], fx[good]
            sel = (wl >= lo_wl) & (wl <= hi_wl)
            wl, fx = wl[sel], fx[sel]
            if wl.size < 10 or np.nanmax(fx) <= 0:
                continue
            orders_clean.append((wl, fx / np.nanmax(fx)))

        def ccf_on(grid):
            per_order = []
            for k, (wl, fx) in enumerate(orders_clean):
                per_order.append(calculate_ccf(wl, fx, tpl_wl, tpl_norm,
                                               grid))
                print(f"  order {k + 1}/{len(orders_clean)} done",
                      end="\r")
            print()
            if not per_order:
                raise RuntimeError(
                    "No CCF could be computed for any order (check the "
                    "wavelength overlap with the template).")
            total = np.sum(per_order, axis=0)
            return total / np.nanmax(total), per_order

    search = "single pass over the given RV window"
    if auto:
        coarse = None
        for half in (CCF_SEARCH_HALF, 450.0, 200.0):
            grid_c = np.arange(-half, half + CCF_SEARCH_STEP,
                               CCF_SEARCH_STEP)
            try:
                cc, _ = ccf_on(grid_c)
            except RuntimeError:
                continue
            coarse = (grid_c, cc, half)
            break
        if coarse is None:
            raise RuntimeError(
                "The coarse RV scan failed for every search width; "
                "check the wavelength overlap with the template.")
        grid_c, cc, half = coarse
        i0 = int(np.nanargmax(cc))
        base = float(np.nanmedian(cc))
        level = base + 0.5 * (cc[i0] - base)
        j = i0
        while j + 1 < cc.size and cc[j + 1] > level:
            j += 1
        k = i0
        while k - 1 >= 0 and cc[k - 1] > level:
            k -= 1
        fwhm = max(grid_c[j] - grid_c[k], 2.0 * CCF_SEARCH_STEP)
        v0 = float(grid_c[i0])
        fine_half = min(max(2.5 * fwhm, 25.0), 450.0)
        lo = max(v0 - fine_half, -half)
        hi = min(v0 + fine_half, half)
        search = (f"two-stage (Katz et al. 2025): coarse +-{half:g} km/s "
                  f"@ {CCF_SEARCH_STEP:g} km/s -> peak near "
                  f"{v0:+.0f} km/s, refined over [{lo:.0f}, {hi:.0f}] "
                  f"@ {rv_step:g} km/s")
        print("RV search: " + search)
        rv_grid = np.arange(lo, hi + rv_step, rv_step)
    else:
        rv_grid = np.arange(rv_min, rv_max + rv_step, rv_step)

    ccf_total, ccf_orders = ccf_on(rv_grid)

    if mode in ("mask", "fft"):
        # Both are already bandpassed/baseline-free enough for the
        # Pino et al. (2018) Gaussian-on-a-linear-baseline fit.
        comps, popt = fit_ccf_peak(rv_grid, ccf_total)
    else:
        comps, popt = fit_peak_with_base(rv_grid, ccf_total,
                                         with_offset=True)
    result = dict(rv=comps[0]["rv"], rv_err=comps[0]["rv_err"],
                  amp=comps[0]["amp"], sigma=comps[0]["sigma"],
                  rv_grid=rv_grid, ccf_total=ccf_total,
                  ccf_orders=np.array(ccf_orders), popt=popt, mode=mode,
                  search=search,
                  peak_snr=float(profile_peak_snr(rv_grid, ccf_total,
                                                  comps)))
    if mode == "mask":
        result["n_lines"] = int(line_wl.size)
    return result


def log_wave_grid(wl_min, wl_max, dv_kms):
    """Wavelength grid with constant velocity step (log-lambda spacing).

    In log-lambda space a constant step equals a constant velocity step, so
    a Doppler shift becomes a simple pixel shift — required by the BF
    convolution model.
    """
    step = np.log(1.0 + dv_kms / C_KMS)
    n = int(np.floor(np.log(wl_max / wl_min) / step)) + 1
    return wl_min * np.exp(step * np.arange(n))


# Representative resolving powers R = lambda/dlambda. Many instruments
# have several modes; the value is the one most useful for RV work
# (highest common resolution), with the range noted. R only sets the
# instrumental broadening c/R used to smooth/degrade, so a representative
# value is sufficient.
SPECTROGRAPHS = [
    # --- high-resolution optical echelle ---
    ("HARPS",     115000, "ESO 3.6 m, La Silla"),
    ("HARPS-N",   115000, "TNG 3.58 m, La Palma"),
    ("ESPRESSO",  140000, "ESO VLT 8.2 m, Paranal (HR; up to 190000 UHR)"),
    ("UVES",       80000, "ESO VLT 8.2 m, Paranal (up to 110000)"),
    ("FEROS",      48000, "MPG/ESO 2.2 m, La Silla"),
    ("SOPHIE",     75000, "OHP 1.93 m (HR mode)"),
    ("CORALIE",    60000, "Euler 1.2 m, La Silla"),
    ("PEPSI",     120000, "LBT 2x8.4 m, Mount Graham"),
    ("NARVAL",     65000, "TBL 2 m, Pic du Midi"),
    ("ESPaDOnS",   68000, "CFHT 3.6 m, Mauna Kea"),
    ("MIKE",       40000, "Magellan 6.5 m, Las Campanas (up to 83000)"),
    ("PFS",        80000, "Magellan 6.5 m (Planet Finder; up to 130000)"),
    ("GHOST",      56000, "Gemini South 8.1 m (standard; 76000 high-res)"),
    ("STELES",     50000, "SOAR 4.1 m, Cerro Pachon"),
    ("CHIRON",     80000, "CTIO/SMARTS 1.5 m (slicer; up to 136000)"),
    ("DAO",        10000, "Dominion Astrophysical Obs. 1.8 m (Plaskett) / "
                          "1.2 m, Victoria BC"),
    # --- near-IR high-resolution ---
    ("CRIRES+",   100000, "ESO VLT 8.2 m, Paranal (0.2\" slit)"),
    ("NIRPS",      90000, "ESO 3.6 m, La Silla (HA 100000, HE 80000)"),
    ("WINERED",    28000, "Magellan-class (WIDE; up to 100000 HIRES)"),
    # --- mid-resolution ---
    ("FLAMES-GIRAFFE", 25000, "ESO VLT 8.2 m, Paranal (R 6000-30000)"),
    ("X-shooter",   8900, "ESO VLT 8.2 m, Paranal (VIS; UVB 5400, NIR 5600)"),
    ("MagE",        4100, "Magellan 6.5 m (Echellette, 1\" slit)"),
    ("M2FS",       18000, "Magellan 6.5 m (fiber; R 1000-40000)"),
    ("IFUM",        4000, "Magellan 6.5 m IFUs feeding M2FS"),
    ("GMOS",        4400, "Gemini 8.1 m (R831 grating; R 600-4400)"),
    ("SOXS",        4500, "ESO NTT 3.58 m, La Silla (UV-VIS 4500, NIR 5000)"),
    ("FIRE",        6000, "Magellan 6.5 m (echelle; up to 8000)"),
    # --- low-resolution ---
    ("FORS2",       2600, "ESO VLT 8.2 m, Paranal (highest-res grism)"),
    ("MUSE",        3000, "ESO VLT 8.2 m, Paranal (R 1770-3590)"),
    ("LDSS3",       1800, "Magellan 6.5 m, Las Campanas"),
    ("EFOSC2",      2200, "ESO NTT 3.58 m, La Silla (grism-dependent)"),
    ("KMOS",        4000, "ESO VLT 8.2 m, Paranal (R 2000-4200)"),
    ("Flamingos-2", 3000, "Gemini South 8.1 m (R up to ~3000)"),
    ("TripleSpec",  3500, "4 m-class NIR (TSpec; R 2700-3500)"),
    # --- space ---
    ("Gaia-RVS",   11500, "Gaia satellite, Radial Velocity Spectrometer "
                          "(Ca II triplet 845-872 nm)"),
    # --- amateur / education echelle ---
    ("Whoppshel-50",  30000, "Whoppshel echelle (Shelyak Instruments), "
                             "50 micron fiber + FIGU unit"),
    ("Whoppshel-105", 15000, "Whoppshel echelle (Shelyak Instruments), "
                             "105 micron fiber + FIGU unit"),
]


# Observatory geodetic coordinates (longitude East-positive [deg],
# latitude [deg], altitude [m]) plus lookup aliases. Kept in-tool so the
# barycentric correction — which needs the observer's position on Earth,
# including the diurnal term (MIDAS comp/bary takes the observatory
# longitude and latitude explicitly) — works without astropy's online
# site registry. Users can add their own site via the manual lon/lat/alt
# fields (GUI) or a "lon,lat,alt" string (--site).
OBSERVATORIES = [
    ("ESO VLT (Paranal)", -70.4045, -24.6272, 2635.0,
     ("paranal", "vlt", "esoparanal"),
     "UVES, ESPRESSO, X-shooter, FLAMES, FORS2, MUSE, KMOS, CRIRES+"),
    ("ESO La Silla", -70.7377, -29.2543, 2400.0,
     ("lasilla", "esolasilla"),
     "HARPS, FEROS, NIRPS, CORALIE (Euler)"),
    ("ESO NTT (La Silla)", -70.7317, -29.2584, 2375.0,
     ("ntt", "esontt"),
     "EFOSC2, SOXS"),
    ("ESO ALMA (Chajnantor)", -67.7549, -23.0229, 5058.7,
     ("alma", "chajnantor"),
     "ALMA array, Llano de Chajnantor"),
    ("ESO ELT (Cerro Armazones)", -70.1922, -24.5892, 3046.0,
     ("elt", "cerroarmazones", "armazones"),
     "Extremely Large Telescope (under construction)"),
    ("OHP (Saint-Michel)", 5.7133, 43.9308, 650.0,
     ("ohp", "hauteprovence", "saintmichel"),
     "Observatoire de Haute-Provence — SOPHIE"),
    ("TUG (Bakirlitepe)", 30.3353, 36.8250, 2500.0,
     ("tug", "tubitak", "rtt150", "bakirlitepe"),
     "TUBITAK National Observatory, Turkey"),
    ("Kreiken (Ankara)", 32.7772, 39.8436, 1256.0,
     ("kreiken", "ankara", "ahlatlibel"),
     "Ankara University Kreiken Observatory"),
    ("CTIO (Cerro Tololo)", -70.8065, -30.1652, 2207.0,
     ("ctio", "cerrotololo", "smarts"),
     "CHIRON (SMARTS 1.5 m)"),
    ("Las Campanas (Magellan)", -70.6926, -29.0146, 2380.0,
     ("lco", "lascampanas", "magellan", "clay", "baade"),
     "MIKE, PFS, MagE, M2FS, IFUM, FIRE, LDSS3, WINERED"),
    ("SOAR (Cerro Pachon)", -70.7337, -30.2380, 2713.0,
     ("soar", "cerropachon"),
     "STELES"),
    ("Gemini South (Cerro Pachon)", -70.7367, -30.2408, 2722.0,
     ("geminisouth", "geminis"),
     "GHOST, GMOS-S, Flamingos-2"),
    ("Gemini North (Mauna Kea)", -155.4690, 19.8238, 4213.0,
     ("gemininorth", "geminin"),
     "GMOS-N"),
    ("CFHT (Mauna Kea)", -155.4685, 19.8253, 4204.0,
     ("cfht", "canadafrancehawaii"),
     "ESPaDOnS"),
    ("Keck (Mauna Kea)", -155.4747, 19.8260, 4145.0,
     ("keck", "wmkeck"),
     "HIRES"),
    ("LBT (Mount Graham)", -109.8892, 32.7016, 3221.0,
     ("lbt", "mountgraham", "largebinocular"),
     "PEPSI"),
    ("DAO (Victoria)", -123.4167, 48.5192, 229.0,
     ("dao", "dominion", "victoria", "plaskett"),
     "Dominion Astrophysical Observatory 1.8 m / 1.2 m"),
    ("TBL (Pic du Midi)", 0.1425, 42.9367, 2877.0,
     ("tbl", "picdumidi", "bernardlyot"),
     "NARVAL"),
    ("TNG (La Palma)", -17.8892, 28.7542, 2387.0,
     ("tng", "galileo"),
     "HARPS-N"),
    ("Roque de los Muchachos (La Palma)", -17.8850, 28.7606, 2326.0,
     ("orm", "roque", "lapalma"),
     "La Palma observatory"),
]


def observatory_coords(name):
    """(lon_East_deg, lat_deg, alt_m) of a built-in observatory by
    (case-insensitive, punctuation-insensitive) name or alias; None when
    unknown."""
    if not name:
        return None
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    for oname, lon, lat, alt, aliases, _note in OBSERVATORIES:
        keys = ["".join(ch for ch in s.lower() if ch.isalnum())
                for s in (oname,) + tuple(aliases)]
        if key in keys:
            return lon, lat, alt
    return None


DEFAULT_SITE = "paranal"
AUTO_SITE_LABEL = "Automatic (from the file header)"
AUTO_SITE_HELP = (
    "Default: the observatory recorded in the file's own header "
    "(Neo-Narval LONGITUD/LATITUDE/OBSELEV, ESO TEL GEOLON/..., "
    "...), then the home observatory of the instrument the header "
    "names, and only then paranal")


# Home observatory of an instrument, for files that record WHAT took the
# spectrum but not WHERE from - a SOPHIE s1d carries neither a telescope
# position nor even an INSTRUME card, only its 'OHP ...' keywords. An
# instrument does not travel, so this is a safe last resort, and a far
# better one than DEFAULT_SITE: reducing a SOPHIE spectrum as if it came
# from Paranal costs 0.24 km/s of barycentric velocity (measured), which
# is 20x the agreement the pipeline cross-check is meant to detect.
# Keys are normalized (letters and digits only, upper case).
INSTRUMENT_SITES = {
    "SOPHIE": "ohp",
    "HARPS": "lasilla", "FEROS": "lasilla", "CORALIE": "lasilla",
    "NIRPS": "lasilla",
    "EFOSC2": "ntt", "SOXS": "ntt",
    "HARPN": "tng", "HARPSN": "tng",
    "UVES": "paranal", "ESPRESSO": "paranal", "XSHOOTER": "paranal",
    "CRIRES": "paranal", "GIRAFFE": "paranal", "FLAMES": "paranal",
    "FORS2": "paranal", "MUSE": "paranal", "KMOS": "paranal",
    "NARVAL": "tbl", "NEONARVAL": "tbl",
    "ESPADONS": "cfht", "HIRES": "keck", "PEPSI": "lbt",
    "CHIRON": "ctio",
    "MIKE": "lco", "PFS": "lco", "MAGE": "lco", "M2FS": "lco",
    "FIRE": "lco", "LDSS3": "lco", "WINERED": "lco",
    "STELES": "soar", "GHOST": "geminisouth", "GMOSS": "geminisouth",
    "GMOSN": "gemininorth",
}


def _instrument_key(name):
    return "".join(ch for ch in str(name or "").upper() if ch.isalnum())


def header_instrument(path):
    """Instrument name recorded in a spectrum file, or None.

    Reads INSTRUME / INSTRUM from any FITS HDU, or the '# INSTRUME ='
    card of a normalized ASCII spectrum. A SOPHIE file names its
    instrument nowhere - it is recognized by its 'OHP ...' keyword
    prefix instead, which is the OHP DRS signature.
    """
    if not path or not os.path.isfile(path):
        return None
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        try:
            from astropy.io import fits
            with fits.open(path) as hdul:
                for hdu in hdul:
                    hdr = hdu.header
                    inst = hdr.get("INSTRUME") or hdr.get("INSTRUM")
                    if inst:
                        return str(inst).strip()
                    if hdr.get("OHP DRS VERSION") or hdr.get("OHP TARG NAME"):
                        return "SOPHIE"
        except Exception:
            return None
        return None
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if "=" in line:
                    key, _, val = line.lstrip("# ").partition("=")
                    if key.strip().upper() in ("INSTRUME", "INSTRUM",
                                               "INSTRUMENT"):
                        return val.strip() or None
    except OSError:
        return None
    return None


def instrument_site(path):
    """(site_name, instrument) of the instrument's home observatory, or
    (None, instrument) when the instrument is unknown to
    INSTRUMENT_SITES."""
    inst = header_instrument(path)
    if not inst:
        return None, None
    key = _instrument_key(inst)
    site = INSTRUMENT_SITES.get(key)
    if site is None:                    # CRIRES+ -> CRIRES, HARPS_A -> HARPS
        for k, v in INSTRUMENT_SITES.items():
            if key.startswith(k):
                site = v
                break
    return site, inst


def observing_site(args, path=None):
    """(site, note): the observatory to use, and one line saying where
    it came from - to be printed only if a barycentric computation
    actually happens, so files with no coordinates stay quiet.

    Order: --site when the user gave one, else the position recorded in
    the file's own header, else the home observatory of the instrument
    the header names (INSTRUMENT_SITES), else DEFAULT_SITE. '--site'
    defaults to None precisely so that this order can be honoured - an
    explicit choice always wins, and a file that knows where it was
    taken never silently inherits somebody else's observatory.
    """
    site = getattr(args, "site", None)
    if site:
        return site, None
    path = path or getattr(args, "spectrum", None)
    found = header_site(path)
    if found:
        lon, lat, alt, cards = found
        name = next((o[0] for o in OBSERVATORIES
                     if abs(o[1] - lon) < 0.05 and abs(o[2] - lat) < 0.05),
                    None)
        return (f"{lon},{lat},{alt}",
                f"Observatory from the file header ({cards}): "
                f"lon {lon:+.4f} E, lat {lat:+.4f}, {alt:.0f} m"
                + (f"  [{name}]" if name else ""))
    inst_site, inst = instrument_site(path)
    if inst_site:
        coords = observatory_coords(inst_site)
        where = (f"lon {coords[0]:+.4f} E, lat {coords[1]:+.4f}, "
                 f"{coords[2]:.0f} m" if coords else inst_site)
        return inst_site, (
            f"Observatory from the instrument ({inst}): {inst_site} "
            f"({where}). The file records no telescope position; give "
            "--site if this is not where the spectrum was taken.")
    return DEFAULT_SITE, (
        f"Note: no observatory in the file header; using the default "
        f"site '{DEFAULT_SITE}'. Give --site if the spectrum was taken "
        "elsewhere - a wrong site is worth up to 0.46 km/s of "
        "barycentric velocity.")


def earth_location(site):
    """astropy EarthLocation for a barycentric/BJD computation.

    `site` may be (1) a built-in OBSERVATORIES name or alias, (2) a
    "lon,lat[,alt]" string in decimal degrees / metres (longitude
    East-positive — the observatory longitude and latitude the MIDAS
    comp/bary command takes explicitly), or (3) any astropy of_site()
    name (needs the online registry).
    """
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    coords = observatory_coords(site)
    if coords is None and isinstance(site, str) and "," in site:
        try:
            vals = [float(p) for p in site.split(",") if p.strip() != ""]
            if len(vals) >= 2:
                coords = (vals[0], vals[1], vals[2] if len(vals) > 2 else 0.0)
        except ValueError:
            coords = None
    if coords is not None:
        lon, lat, alt = coords
        return EarthLocation.from_geodetic(lon * u.deg, lat * u.deg,
                                           alt * u.m)
    return EarthLocation.of_site(site)


def spectrograph_resolution(name):
    """R of a built-in spectrograph by (case-insensitive) name."""
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    for n, r, _ in SPECTROGRAPHS:
        if "".join(ch for ch in n.lower() if ch.isalnum()) == key:
            return float(r)
    known = ", ".join(n for n, _, _ in SPECTROGRAPHS)
    raise ValueError(f"Unknown spectrograph '{name}'. Built-in: {known}. "
                     "Use --resolution to enter R manually.")


def get_resolution(args):
    """Effective R: explicit --resolution wins, otherwise the built-in
    value of --spectrograph; None when neither is given."""
    if getattr(args, "resolution", None):
        return args.resolution
    sp = getattr(args, "spectrograph", None)
    if sp:
        r = spectrograph_resolution(sp)
        print(f"Spectrograph {sp}: R = {r:g} "
              f"(instrumental broadening c/R = {C_KMS / r:.2f} km/s)")
        return r
    return None


def min_vsini(resolution):
    """Smallest meaningful vsini at a given R: the instrumental profile
    itself is Δv_inst ≈ c/R wide, so rotation below that cannot be
    resolved."""
    return C_KMS / float(resolution)


def cr_trim(spec_wl, spec_flux):
    """Interpolate over unphysical normalized-flux pixels (> 1.2 or < 0)
    following the cr_trim step of saphires.utils.prepare: cosmic rays,
    emission spikes and the continuum failures at echelle order edges.

    Returns (flux, n_fixed) — the flux unchanged with n_fixed = 0 when
    nothing is affected, or when more than half the pixels are (which
    means the spectrum is not normalized rather than spike-ridden).
    """
    bad = (spec_flux > 1.2) | (spec_flux < 0.0)
    n_bad = int(bad.sum())
    if not 0 < n_bad < 0.5 * spec_flux.size:
        return spec_flux, 0
    spec_flux = spec_flux.copy()
    spec_flux[bad] = np.interp(spec_wl[bad], spec_wl[~bad], spec_flux[~bad])
    return spec_flux, n_bad


def prepare_template(tpl_wl, tpl_flux, resolution=None, vsini=None,
                     epsilon=0.6, verbose=True):
    """Degrade the synthetic template BEFORE the BF/CCF measurement.

    Synthetic spectra are effectively of 'infinite' resolution and
    rotationless; two effects must be applied before comparing them with
    an observed spectrum (a user choice — they differ per
    telescope/spectrograph):

    resolution : spectrograph resolving power R = lambda/dlambda.
        The template is convolved with a Gaussian of FWHM = c/R [km/s]
        (e.g. FEROS R=48000 -> 6.2 km/s, ESPRESSO R=140000 -> 2.1 km/s).
        Note: broadening the template makes the SVD design matrix more
        ill-conditioned; if the BF becomes unstable, the cutoff is
        auto-escalated or can be set with svd_rcond (see compute_bf).
    vsini : projected rotational velocity [km/s].
        Rotational broadening with the Gray (1992) profile using the
        linear limb-darkening coefficient `epsilon` (default 0.6).

    Both convolutions are carried out on a constant-velocity (log-lambda)
    grid, so a single kernel is exact across the whole wavelength range.
    Returns the (possibly resampled) template (wavelength, flux).
    """
    if not resolution and not vsini:
        return tpl_wl, tpl_flux

    good = np.isfinite(tpl_wl) & np.isfinite(tpl_flux)
    tpl_wl, tpl_flux = tpl_wl[good], tpl_flux[good]
    pix = np.median(np.diff(tpl_wl)) / np.median(tpl_wl) * C_KMS
    dv = max(min(pix, 2.0), 0.2)
    grid = log_wave_grid(tpl_wl.min(), tpl_wl.max(), dv)
    flux = np.interp(grid, tpl_wl, tpl_flux)

    if vsini and vsini > 0:
        if resolution and resolution > 0 and vsini < min_vsini(resolution):
            print(f"Warning: vsini = {vsini:g} km/s is below the "
                  f"instrumental broadening c/R = "
                  f"{min_vsini(resolution):.2f} km/s at R = {resolution:g} "
                  "- unresolvable; rotational broadening skipped.")
        else:
            nk = max(int(np.ceil(vsini / dv)), 1)
            v = np.arange(-nk, nk + 1) * dv
            x = v / float(vsini)
            kern = np.zeros_like(x)
            inside = np.abs(x) < 1.0
            kern[inside] = (2.0 * (1.0 - epsilon)
                            * np.sqrt(1.0 - x[inside] ** 2)
                            + 0.5 * np.pi * epsilon * (1.0 - x[inside] ** 2))
            kern /= kern.sum()
            flux = np.convolve(flux, kern, mode="same")
            if verbose:
                print(f"Template: rotational broadening vsini = "
                      f"{vsini:g} km/s (epsilon = {epsilon:g}) applied")

    if resolution and resolution > 0:
        fwhm_kms = C_KMS / float(resolution)
        sigma_pix = fwhm_kms / 2.35482 / dv
        flux = gaussian_filter1d(flux, sigma_pix)
        if verbose:
            print(f"Template: degraded to R = {resolution:g} "
                  f"(instrumental FWHM = {fwhm_kms:.2f} km/s)")

    return grid, flux


def compute_bf(spec_wl, spec_flux, tpl_wl, tpl_flux,
               vel_range=500.0, dv=None, svd_rcond=None, smooth_kms=None,
               inst_fwhm=None, verbose=True):
    """Broadening Function following the reference implementation
    (BF_main.txt; Rucinski 1999 via PyAstronomy's pyasl.SVD).

    Reference steps:
      1. log-wavelength grid  w1 = w00 * (1+r)^k  with  r = stepV/c
         (w00/stepV in the reference: 4800 A / 5 km/s; here w00 comes from
         the data overlap and stepV from `dv`)
      2. template and observed spectra are interpolated onto w1 and
         inverted to line depths (1 - flux). Rucinski's BF is defined on
         continuum-subtracted line profiles; the inversion removes the
         continuum from the SVD problem, so the BF baseline sits at zero
         and the wings decay cleanly. (Feeding continuum-normalized flux
         as-is makes the SVD reproduce the continuum as a constant
         pedestal ~ continuum/m under the whole BF, with rolled-off,
         even negative edges — the profile then rides on a false base.)
      3. svd.decompose(template, m);  bf = svd.getBroadeningFunction(obs)
         (m odd, reference m=501; here m = 2*ceil(vel_range/dv)+1)
      4. velocity axis = svd.getRVAxis(r, 1)
      5. Gaussian smoothing of the BF, sigma = 2 bins in the reference
         (used when smooth_kms is not given)

    pyasl.SVD is used when PyAstronomy is installed; otherwise an exact
    NumPy replication of the same algorithm runs (verified identical).

    Parameters
    ----------
    vel_range : half-width of the BF window [km/s]
    dv        : velocity step stepV [km/s]; None -> bf_default_step():
        the instrumental FWHM divided by BF_STEP_PER_FWHM when the
        resolution is known, else the reference BF_STEPV (5 km/s), and
        never finer than the median velocity pixel of the data. The step
        must resolve the narrowest component the instrument can deliver,
        otherwise the Gaussian fit is unconstrained and its formal error
        is meaningless. The per-bin BF amplitude scales linearly with
        dv (the profile integral is what is conserved), so amplitudes
        are only comparable between analyses run with the same step;
        the default keeps them on the reference-implementation scale.
    svd_rcond : relative singular-value cutoff. None (default) selects it
        automatically: the smallest cutoff from a ladder starting at 0
        (no truncation — the choice of both the reference notebook and
        BF-rvplotter, Rucinski's smooth-instead-of-truncate route) whose
        smoothed BF keeps the wing noise below a tenth of the peak; wide
        continuum-dominated ranges and broadened templates then get just
        enough truncation to stay clean. The selection is printed. An
        explicit number is used as-is (0 = never truncate).
    smooth_kms: smoothing FWHM [km/s]; None -> the instrumental FWHM
        c/R when `inst_fwhm` is given (the saphires bf.analysis
        practice: smoothing below the spectrograph resolution makes no
        sense), else sigma = 2 bins (reference notebook)
    inst_fwhm : instrumental FWHM c/R [km/s]; used as the default
        smoothing width (see smooth_kms)
    verbose   : print notes/selections (set False for per-order runs)

    Observed pixels with unphysical normalized flux (> 1.2 or < 0:
    cosmic rays, emission spikes) are interpolated over before the SVD,
    following the cr_trim step of saphires.utils.prepare.

    Returns: dict(velocity, bf, bf_smooth, dv, n_kept_sv, n_sv)
    """
    good_s = np.isfinite(spec_wl) & np.isfinite(spec_flux)
    good_t = np.isfinite(tpl_wl) & np.isfinite(tpl_flux)
    spec_wl, spec_flux = spec_wl[good_s], spec_flux[good_s]
    tpl_wl, tpl_flux = tpl_wl[good_t], tpl_flux[good_t]

    spec_flux, n_fixed = cr_trim(spec_wl, spec_flux)
    if n_fixed and verbose:
        print(f"Note: {n_fixed} unphysical pixel(s) (flux > 1.2 "
              "or < 0: cosmics/emission) interpolated over before "
              "the SVD.")

    wl_min = max(spec_wl.min(), tpl_wl.min())
    wl_max = min(spec_wl.max(), tpl_wl.max())
    if wl_max <= wl_min:
        raise ValueError("The spectrum and template wavelength ranges "
                         "do not overlap.")
    if wl_max - wl_min > 800.0 and verbose:
        print(f"Note: BF computed over a {wl_max - wl_min:.0f} A wide "
              "range; a narrower line-rich window (e.g. --wave-min 500 "
              "--wave-max 550) usually gives a cleaner BF.")

    pix = np.median(np.diff(spec_wl)) / np.median(spec_wl) * C_KMS
    if dv is None:
        dv = bf_default_step(inst_fwhm, pix)
        if verbose and dv > pix:
            src = ("instrumental FWHM / "
                   f"{BF_STEP_PER_FWHM:g}, FWHM = {inst_fwhm:.2f} km/s"
                   if inst_fwhm else "reference default")
            print(f"BF velocity step: stepV = {dv:g} km/s ({src}; "
                  f"data pixel = {pix:.2f} km/s; --dv overrides)")

    r = dv / C_KMS
    n = int(np.floor(np.log(wl_max / wl_min) / np.log(1.0 + r))) + 1
    w1 = wl_min * np.power(1.0 + r, np.arange(float(n)))

    tpl = 1.0 - np.interp(w1, tpl_wl, tpl_flux)
    obs = 1.0 - np.interp(w1, spec_wl, spec_flux)

    half = int(np.ceil(vel_range / dv))
    m = 2 * half + 1
    if n <= 2 * m:
        raise ValueError("Spectrum segment too short for the BF window; "
                         "widen the wavelength range or reduce vel-range.")

    try:
        from PyAstronomy import pyasl
        svd = pyasl.SVD()
        svd.decompose(tpl, m)
        w = np.ravel(np.asarray(svd.getSingularValues()))
        solve = lambda lim: svd.getBroadeningFunction(obs, wlimit=lim,
                                                      asarray=True)
        velocity = svd.getRVAxis(r, 1)
    except ImportError:
        nn = n - m + 1
        des = np.empty((nn, m))
        for i in range(m):
            des[:, i] = tpl[i:i + nn]
        U, w, Vt = np.linalg.svd(des, full_matrices=False)
        rhs = obs[m // 2: len(obs) - (m // 2)]

        def solve(lim):
            winv = np.where(w > lim, 1.0 / w, 0.0)
            return Vt.T @ (winv * (U.T @ rhs))
        velocity = (-np.arange(m) + m // 2) * r * C_KMS

    if smooth_kms:
        sigma_pix = smooth_kms / (2.35482 * dv)
    elif inst_fwhm:
        sigma_pix = inst_fwhm / (2.35482 * dv)
        if verbose:
            print(f"BF smoothing: instrumental FWHM c/R = "
                  f"{inst_fwhm:.2f} km/s (saphires practice; "
                  "use --smooth to override)")
    else:
        sigma_pix = 2.0

    def bf_quality(b):
        """Noise-to-peak ratio of a candidate BF: smoothed wing noise
        plus high-frequency wing roughness of the raw solution, relative
        to the peak amplitude (lower is cleaner). The peak region itself
        is excluded so sharp real peaks are not penalized."""
        sm = gaussian_filter1d(b, sigma_pix)
        ipk = int(np.argmax(np.abs(sm)))
        peak = abs(float(sm[ipk]))
        if peak <= 0 or not np.isfinite(peak):
            return np.inf
        span = float(velocity.max() - velocity.min())
        away = np.abs(velocity - velocity[ipk]) > 0.25 * span
        if not away.any():
            return np.inf
        wing = sm[away]
        noise = 1.4826 * np.median(np.abs(wing - np.median(wing)))
        rough = float(np.std((b - sm)[away]))
        return (float(noise) + rough) / peak

    if svd_rcond is None:
        cand = []
        for rc in (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            lim = rc * w.max() if rc > 0 else 0.0
            b = np.ravel(np.asarray(solve(lim)))
            if not np.all(np.isfinite(b)):
                continue
            ratio = bf_quality(b)
            if np.isfinite(ratio):
                cand.append((rc, b, ratio))
        if not cand:
            raise RuntimeError("The BF could not be stabilized at any "
                               "SVD cutoff; check the input spectra.")
        best_ratio = min(c[2] for c in cand)
        tol = max(0.10, 1.3 * best_ratio)
        best = next(c for c in cand if c[2] <= tol)
        svd_rcond, bf, ratio = best
        wlimit = svd_rcond * w.max() if svd_rcond > 0 else 0.0
        if verbose:
            print(f"SVD regularization: svd_rcond = {svd_rcond:g} selected "
                  f"automatically (BF noise / peak = {ratio:.3f}, least "
                  "truncation within tolerance); use --svd-rcond to "
                  "override.")
    else:
        wlimit = svd_rcond * w.max() if svd_rcond > 0 else 0.0
        bf = np.ravel(np.asarray(solve(wlimit)))
        if svd_rcond <= 0 and np.max(np.abs(bf)) > 10.0:
            for rc in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
                wlimit = rc * w.max()
                bf = np.ravel(np.asarray(solve(wlimit)))
                if np.max(np.abs(bf)) <= 10.0:
                    if verbose:
                        print(f"Note: BF was unstable at svd_rcond=0 "
                              f"(continuum-dominated region); "
                              f"auto-regularized to svd_rcond={rc:g}. "
                              "Pass --svd-rcond to set it explicitly.")
                    break
    n_sv = m
    n_kept = int((w > wlimit).sum())

    srt = np.argsort(velocity)
    velocity, bf = velocity[srt], bf[srt]

    bf_smooth = gaussian_filter1d(bf, sigma_pix)

    return dict(velocity=velocity, bf=bf, bf_smooth=bf_smooth, dv=dv,
                pixel_kms=float(pix), n_kept_sv=n_kept, n_sv=n_sv)


def compute_bf_orders(orders, tpl_wl, tpl_flux, vel_range=500.0, dv=None,
                      svd_rcond=None, smooth_kms=None, inst_fwhm=None,
                      std_perc=0.1):
    """Multi-order BF following saphires bf.compute + bf.weight_combine:
    one BF per echelle order on a common velocity grid, combined with
    weights 1/std^2 where std is the scatter of each order's BF
    sidebands (the outer `std_perc` fraction of bins at both ends).
    Noisy or line-poor orders are down-weighted automatically instead
    of degrading a single merged-spectrum solution.

    A common dv (bf_default_step from the instrumental FWHM, or the
    finest median velocity pixel over the orders when that is coarser)
    and a common
    window make every order's velocity axis identical, which is what
    makes the BFs combinable. Orders that fail (no overlap, too short)
    are skipped.

    Returns the compute_bf dict plus n_orders (number combined).
    """
    if dv is None:
        pixes = []
        for wl, fx in orders:
            good = np.isfinite(wl) & np.isfinite(fx)
            if good.sum() > 10:
                w = wl[good]
                pixes.append(np.median(np.diff(w)) / np.median(w) * C_KMS)
        if not pixes:
            raise ValueError("No usable order for the BF.")
        dv = bf_default_step(inst_fwhm, min(pixes))

    results, failures = [], []
    for k, (wl, fx) in enumerate(orders):
        try:
            r = compute_bf(wl, fx, tpl_wl, tpl_flux, vel_range=vel_range,
                           dv=dv, svd_rcond=svd_rcond,
                           smooth_kms=smooth_kms, inst_fwhm=inst_fwhm,
                           verbose=False)
            results.append(r)
        except (ValueError, RuntimeError) as exc:
            failures.append((k, exc))
        print(f"  BF order {k + 1}/{len(orders)} done", end="\r")
    print()
    if not results:
        raise RuntimeError(
            "The BF could not be computed for any order; first failure: "
            f"{failures[0][1]}" if failures else "no orders given.")

    velocity = results[0]["velocity"]
    weights = []
    for r in results:
        nb = max(int(round(std_perc * velocity.size)), 5)
        side = np.concatenate([r["bf_smooth"][:nb], r["bf_smooth"][-nb:]])
        std = float(np.std(side))
        weights.append(1.0 / max(std, 1e-12) ** 2)
    weights = np.array(weights)
    weights /= weights.sum()

    bf = np.sum([w * r["bf"] for w, r in zip(weights, results)], axis=0)
    bf_smooth = np.sum([w * r["bf_smooth"]
                        for w, r in zip(weights, results)], axis=0)
    top = results[int(np.argmax(weights))]
    print(f"BF combined from {len(results)} order(s), weighted by "
          "1/std^2 of the BF sidebands (saphires weight_combine); "
          f"{len(failures)} order(s) skipped.")
    return dict(velocity=velocity, bf=bf, bf_smooth=bf_smooth, dv=dv,
                pixel_kms=min(r["pixel_kms"] for r in results),
                n_kept_sv=top["n_kept_sv"], n_sv=top["n_sv"],
                n_orders=len(results))


BF_FIT_TRIM = 20


# How much broader than the profile's own half-width the "base" of the
# sharp-peak-plus-broad-base fit has to be. The two components only mean
# something as long as their widths cannot overlap; see
# fit_peak_with_base for what happens when they can.
BF_BASE_WIDTH_RATIO = 3.0
# Hard limits on that split [km/s]: never below the velocity step (the
# sharp component needs room), never so high that the base has none.
BF_BASE_SPLIT_MIN = 5.0
BF_BASE_SPLIT_MAX = 150.0


def peak_width_split(velocity, profile, ipk):
    """(sigma0, split) for fit_peak_with_base: the Gaussian sigma implied
    by the half-width of `profile` around bin `ipk`, and the width at
    which the sharp component is separated from the broad base.

    The split is measured from the profile instead of being a fixed
    velocity, so it follows the instrument and the star: a 4 km/s
    HARPS-class BF and a 60 km/s BF of a fast rotator both get a base
    that is genuinely a base.
    """
    step = (float(np.median(np.abs(np.diff(velocity))))
            if velocity.size > 1 else 1.0)
    base = float(np.median(profile))
    half = base + 0.5 * (float(profile[ipk]) - base)
    j = ipk
    while j + 1 < profile.size and profile[j + 1] > half:
        j += 1
    k = ipk
    while k - 1 >= 0 and profile[k - 1] > half:
        k -= 1
    w50 = max(float(velocity[j] - velocity[k]), 2.0 * step)
    sigma0 = w50 / 2.35482
    split = float(np.clip(BF_BASE_WIDTH_RATIO * sigma0,
                          max(BF_BASE_SPLIT_MIN, 3.0 * step),
                          BF_BASE_SPLIT_MAX))
    return sigma0, split


def fit_peak_with_base(velocity, profile, with_offset=False, center=None,
                       fit_trim=0):
    """Reference-notebook peak fit (BF_main_fixed.ipynb): a bounded double
    Gaussian made of a sharp peak plus a broad base component, initial
    guesses taken from the profile maximum (or from `center` when an
    explicit initial RV guess is given). The broad component absorbs
    the pedestal/wings of the profile so the centre of the sharp
    component — which carries the RV — is not pulled. Bounds follow the
    reference: amplitudes >= 0, centres inside the velocity axis,
    sigma >= 1 km/s for the peak and <= 300 km/s for the base.

    The two widths are held on opposite sides of a split value
    (peak_width_split), which is what makes "sharp" and "broad"
    distinguishable states of the model rather than two names for the
    same thing. Under the reference's overlapping ranges (1-200 and
    1-300 km/s) they are not distinguishable: the model is then exactly
    symmetric under exchanging the two components, so on a profile with
    no real pedestal the fit splits the single peak into two
    near-identical Gaussians. That pair slides along itself at almost
    constant chi^2 - it is degenerate - so the covariance blows up and
    the reported centre is dragged off the peak. Measured on a
    noiseless synthetic star at exactly +42.00 km/s: the overlapping
    bounds give 42.59 +- 23.15 km/s, the split bounds 41.999 +- 0.004.
    The failure is unstable rather than systematic - a change of 1e-5 in
    dv flips the fit between the two - which is why it survived so long.

    Because the split fixes which slot holds which component, the sharp
    one is always the first triple, and that is the one reported. The
    previous "whichever has the larger amplitude" rule existed only
    because the slots were interchangeable; with the bounds separated it
    could pick the pedestal, whose centre is not the RV.

    with_offset=True adds a constant baseline term (for CCF profiles,
    whose baseline is not zero). fit_trim drops that many bins from
    each edge of the profile before fitting (saphires practice: the BF
    edges are noisy).

    Returns ([dict(rv, rv_err, amp, sigma) of the sharp component], popt).
    """
    velocity = np.asarray(velocity, dtype=float)
    profile = np.asarray(profile, dtype=float)
    fit_trim = min(int(fit_trim), velocity.size // 20)
    if fit_trim > 0 and velocity.size > 4 * fit_trim:
        velocity = velocity[fit_trim:-fit_trim]
        profile = profile[fit_trim:-fit_trim]
    if center is None:
        ipk = int(np.argmax(profile))
    else:
        ipk = int(np.argmin(np.abs(velocity - float(center))))
    vlo, vhi = float(velocity.min()), float(velocity.max())
    offset0 = float(np.median(profile)) if with_offset else 0.0
    amp0 = max(float(profile[ipk]) - offset0, 1e-6)
    sigma0, split = peak_width_split(velocity, profile, ipk)
    p0 = [amp0, float(velocity[ipk]), min(max(sigma0, 1.0), split),
          0.1 * amp0, float(velocity[ipk]), min(2.0 * split, 300.0)]
    lo = [0.0, vlo, 1.0, 0.0, vlo, split]
    hi = [np.inf, vhi, split, np.inf, vhi, 300.0]
    if with_offset:
        p0, lo, hi = p0 + [offset0], lo + [-np.inf], hi + [np.inf]
        popt, pcov = curve_fit(two_gauss, velocity, profile, p0=p0,
                               bounds=(lo, hi), maxfev=20000)
    else:
        popt, pcov = curve_fit(two_gauss_no, velocity, profile, p0=p0,
                               bounds=(lo, hi), maxfev=20000)
    err = np.sqrt(np.diag(pcov))
    # A parameter pinned at a bound gives an exactly-zero or infinite
    # variance; report that as nan rather than as a precision the fit
    # does not have (same rule as fit_bf_peaks).
    err[~np.isfinite(err) | (err <= 0.0)] = np.nan
    comp = dict(rv=popt[1], rv_err=float(err[1]),
                amp=popt[0], sigma=abs(popt[2]), kind="gauss")
    return [comp], dict(kind="gauss", popt=popt)


def find_bf_peaks(velocity, bf, n, min_sep, centers=None):
    """Initial (amp, centre, sigma) seeds for an n-component fit by
    iterative fit-and-subtract: fit k Gaussians on a shared constant
    baseline, smooth the residual with a min_sep-wide kernel, and take
    its maximum as the (k+1)-th centre. When explicit `centers` are
    given (user initial guesses), each stage uses the unused centre
    nearest to the residual maximum — the guesses anchor the components
    while the staged refits let each one's amplitude and width converge
    before the next is added.

    Picking the n highest separated local maxima (the previous approach,
    BF-rvplotter style) fails on real close-binary/triple BFs in two
    ways: noise wiggles on the shoulder of a broad primary are higher
    'local maxima' than a genuinely distant faint secondary, and a
    narrow tertiary superposed on the broad primary at nearly the same
    RV produces no separate maximum at all. Fitting-and-subtracting
    finds both, because each new component is placed on the largest
    coherent structure the current model fails to explain; min_sep
    [km/s] is the smoothing scale a residual structure must survive, so
    single-bin noise spikes are never selected.

    Returns a list of n (amp, centre, sigma) tuples from the last
    iteration's fit; the caller runs the final fit (with its own model
    and bounds) from these seeds."""
    velocity = np.asarray(velocity, dtype=float)
    bf = np.asarray(bf, dtype=float)
    step = float(np.median(np.abs(np.diff(velocity))))
    sm_sigma = max(min_sep / 2.35482 / step, 1.0)
    vlo, vhi = float(velocity.min()), float(velocity.max())
    sigma0 = max(min_sep / 2.0, 5.0)

    remaining = [float(c) for c in centers] if centers is not None else None
    popt = None
    for k in range(1, n + 1):
        model = (multi_gauss_no(velocity, *popt) if popt is not None
                 else float(np.median(bf)))
        resid_sm = gaussian_filter1d(bf - model, sm_sigma)
        j = int(np.argmax(resid_sm))
        if remaining is not None:
            c = min(remaining, key=lambda g: abs(g - velocity[j]))
            remaining.remove(c)
            i = int(np.argmin(np.abs(velocity - c)))
        else:
            i = j
        amp0 = max(float((bf - model)[i]), 1e-6)
        prev = list(popt[:-1]) if popt is not None else []
        offset0 = float(popt[-1]) if popt is not None else \
            float(np.median(bf))
        p0 = prev + [amp0, float(velocity[i]), sigma0] + [offset0]
        lo = [0.0, vlo, 1.0] * k + [-np.inf]
        hi = [np.inf, vhi, 200.0] * k + [np.inf]
        try:
            popt, _ = curve_fit(multi_gauss_no, velocity, bf, p0=p0,
                                bounds=(lo, hi), maxfev=60000)
        except RuntimeError:
            popt = np.asarray(p0, dtype=float)
    return [tuple(float(x) for x in popt[j:j + 3])
            for j in range(0, 3 * n, 3)]


def fit_bf_peaks(velocity, bf, components=1, min_sep=30.0, with_offset=False,
                 guesses=None, guess_amps=None, guess_sigmas=None,
                 profile="gauss", inst_fwhm=None, epsilon=0.6,
                 fit_trim=0):
    """Fit the BF/CCF profile following the reference notebook,
    BF-rvplotter and the DDO close-binary series.

    components=1: bounded sharp-peak + broad-base double Gaussian
    (fit_peak_with_base); the sharp component is reported.
    components=2/3: one bounded profile per stellar component, started
    from the highest separated local maxima of the profile (at least
    min_sep [km/s] apart) or from explicit initial guesses.

    profile: 'gauss' (default) fits Gaussians; 'rot' fits Gray (1992)
    rotational profiles convolved with the instrumental Gaussian
    (inst_fwhm [km/s], from the spectrograph R) — the DDO-series /
    saphires practice for the rotation-dominated components of close
    binaries: the width parameter is then vsini, not a Gaussian sigma.

    guesses: optional list of initial RVs [km/s], one per component —
    BF-rvplotter's gausspars practice. The returned components keep the
    guess order, so the C1/C2/... labels are fixed by the user and the
    star identity of every profile is unambiguous. Without guesses the
    components are returned sorted by RV (callers may re-sort by area or
    by orbital phase).

    guess_amps / guess_sigmas: optional initial amplitudes (BF peak
    heights read off the plot) and widths [km/s], one per component,
    paired with `guesses`. With amplitudes given the fit runs directly
    from the user's [amp, RV, sigma] starting point — the reference-
    notebook workflow: inspect the smoothed BF visually, type in the
    per-component estimates, and let curve_fit converge from them.
    Without them the starting widths/amplitudes come from the staged
    residual search (see find_bf_peaks).

    Multi-component fits always include a shared constant baseline
    offset (saphires d_gaussian_off / t_gaussian_off); with_offset=True
    adds the same constant term to single-component fits (used for CCF
    profiles, whose baseline is not zero).

    fit_trim: number of bins dropped from each edge of the profile
    before the fit — the BF edges are noisy (saphires bf.analysis,
    default there 20). Capped at 5% of the bins per edge so a coarse
    velocity step cannot trim real profile range away.

    Returns: (list of dict(rv, rv_err, amp, sigma, kind) per component,
    model descriptor dict for eval_bf_model).
    """
    velocity = np.asarray(velocity, dtype=float)
    bf = np.asarray(bf, dtype=float)
    fit_trim = min(int(fit_trim), velocity.size // 20)
    if fit_trim > 0 and velocity.size > 4 * fit_trim:
        velocity = velocity[fit_trim:-fit_trim]
        bf = bf[fit_trim:-fit_trim]
    if guesses is not None and len(guesses) != components:
        raise ValueError(f"{components} components but {len(guesses)} "
                         "initial guesses given.")
    for name, extra in (("amplitude", guess_amps), ("sigma", guess_sigmas)):
        if extra is not None:
            if not guesses:
                raise ValueError(f"Initial {name} guesses need the RV "
                                 "guesses as well.")
            if len(extra) != components:
                raise ValueError(f"{components} components but "
                                 f"{len(extra)} initial {name} guesses "
                                 "given.")
    if components == 1:
        center = guesses[0] if guesses else None
        return fit_peak_with_base(velocity, bf, with_offset=with_offset,
                                  center=center)
    offset0 = float(np.median(bf))
    if guesses and guess_amps:
        seeds = [(float(a), float(g),
                  float(guess_sigmas[k]) if guess_sigmas else None)
                 for k, (a, g) in enumerate(zip(guess_amps, guesses))]
    elif guesses:
        staged = sorted((float(g) for g in guesses),
                        key=lambda g: -bf[np.argmin(np.abs(velocity - g))])
        seeds = find_bf_peaks(velocity, bf, components, min_sep,
                              centers=staged)
    else:
        seeds = find_bf_peaks(velocity, bf, components, min_sep)

    use_rot = profile == "rot"
    step = float(np.median(np.abs(np.diff(velocity))))
    inst_sigma_pix = (inst_fwhm / 2.35482 / step) if (use_rot and inst_fwhm
                                                      ) else None
    vlo, vhi = float(velocity.min()), float(velocity.max())
    p0, lo, hi = [], [], []
    for a0, c, s0 in seeds:
        if use_rot:
            p0 += [max(a0, 1e-6), c, min(max(s0 or 40.0, 2.0), 300.0)]
            lo += [0.0, vlo, 2.0]
            hi += [np.inf, vhi, 300.0]
        else:
            p0 += [max(a0, 1e-6), c, min(max(s0 or 20.0, 1.0), 200.0)]
            lo += [0.0, vlo, 1.0]
            hi += [np.inf, vhi, 200.0]
    p0, lo, hi = p0 + [offset0], lo + [-np.inf], hi + [np.inf]

    if use_rot:
        def rot_model(x, *params):
            return multi_rot_profile(x, params, epsilon=epsilon,
                                     inst_sigma_pix=inst_sigma_pix)
        popt, pcov = curve_fit(rot_model, velocity, bf, p0=p0,
                               bounds=(lo, hi), maxfev=60000)
        model = dict(kind="rot", popt=popt, epsilon=epsilon,
                     inst_sigma_pix=inst_sigma_pix)
    else:
        popt, pcov = curve_fit(multi_gauss_no, velocity, bf, p0=p0,
                               bounds=(lo, hi), maxfev=60000)
        model = dict(kind="gauss", popt=popt)
    err = np.sqrt(np.diag(pcov))
    # A singular covariance - a parameter pinned at a fit bound - yields
    # exactly-zero or infinite variances; report those errors as nan
    # instead of a misleading 0.0 or an astronomically large number.
    err[~np.isfinite(err) | (err <= 0.0)] = np.nan
    kind = "rot" if use_rot else "gauss"
    comps = [dict(rv=popt[k + 1], rv_err=float(err[k + 1]), amp=popt[k],
                  sigma=abs(popt[k + 2]), kind=kind)
             for k in range(0, 3 * components, 3)]
    if guesses and guess_amps:
        pass  # p0 slots were built in guess order, which curve_fit keeps
    elif guesses:
        from itertools import permutations
        best = min(permutations(range(components)),
                   key=lambda p: sum(abs(comps[j]["rv"] - g)
                                     for j, g in zip(p, guesses)))
        comps = [comps[j] for j in best]
    else:
        comps.sort(key=lambda c: c["rv"])
    return comps, model


def undersampled_components(comps, dv):
    """Indices of components whose Gaussian sigma spans fewer than
    BF_MIN_SIGMA_BINS bins of the velocity grid — the fit has too few
    points across the profile for the covariance, and therefore rv_err,
    to mean anything."""
    return [i for i, c in enumerate(comps)
            if np.isfinite(c.get("sigma", np.nan))
            and c["sigma"] < BF_MIN_SIGMA_BINS * dv]


def fit_bf_refined(bf_result, recompute=None, dv_user=None, verbose=True,
                   **fit_kw):
    """Fit the BF components, then make sure the velocity grid actually
    resolves them.

    A Gaussian fitted over one or two bins is formally solvable but not
    constrained: curve_fit returns a centre with an error of hundreds or
    thousands of km/s (or nan when the covariance is singular). When such
    a component appears, the BF is recomputed on a grid fine enough to
    put BF_MIN_SIGMA_BINS bins inside the narrowest sigma and the fit is
    repeated, at most BF_MAX_REFINE times and never below the data pixel.

    recompute : callable dv -> bf_result, used to redo the BF on the
        finer grid. None (or a user-supplied --dv, `dv_user`) disables
        the refinement — the undersampling is then only reported, since
        an explicit step is the user's decision to make.

    Returns (bf_result, comps, model, notes); `notes` are human-readable
    lines describing every refinement and every remaining warning, meant
    for both the console and the result file.
    """
    comps, model = fit_bf_peaks(bf_result["velocity"],
                                bf_result["bf_smooth"], **fit_kw)
    notes = []
    # Two limits on how fine the grid may get. The data pixel is the hard
    # one. The second is the instrumental width: the BF is smoothed to it,
    # so an unresolved component (sigma ~ the instrumental sigma) stays
    # unresolved however fine the grid gets — sampling below it only adds
    # correlated bins, which shrink the formal error without adding
    # information. Refinement exists to make a fit *constrained*, not to
    # manufacture precision.
    pixel = BF_REFINE_PIXEL_FRAC * bf_result.get("pixel_kms", 0.0)
    inst_fwhm_kw = fit_kw.get("inst_fwhm")
    inst_floor = ((inst_fwhm_kw / 2.35482) / BF_MIN_SIGMA_BINS
                  if inst_fwhm_kw else 0.0)
    floor = max(pixel, inst_floor)
    limit_txt = ("the profile is already sampled to the instrumental "
                 f"width ({inst_fwhm_kw:.2f} km/s FWHM)"
                 if inst_floor >= pixel and inst_fwhm_kw
                 else f"the data pixel is {pixel:.2f} km/s")
    for _ in range(BF_MAX_REFINE):
        dv = float(bf_result["dv"])
        narrow = undersampled_components(comps, dv)
        if not narrow:
            break
        s_min = min(comps[i]["sigma"] for i in narrow)
        # Aim for twice the minimum sampling, not exactly the threshold, so
        # one pass lands clear of it; the floor is the data pixel.
        dv_new = max(s_min / (2.0 * BF_MIN_SIGMA_BINS), floor)
        # Components are named by velocity, not by index: the C1/C2 labels
        # are only assigned later (by BF area or orbital phase), so an
        # index here would not match the labels in the result file.
        which = ", ".join(f"the component at RV = {comps[i]['rv']:.2f} km/s "
                          f"(sigma = {comps[i]['sigma']:.2f} km/s)"
                          for i in narrow)
        if dv_new >= 0.95 * dv:
            # Undersampled implies the target step is below dv/2, so this
            # can only be the floor binding: a finer BF grid would carry no
            # new information.
            notes.append(
                f"Unresolved: {which} — narrower than "
                f"{BF_MIN_SIGMA_BINS:g} x dv = {BF_MIN_SIGMA_BINS * dv:.2f} "
                f"km/s, but {limit_txt}, so a finer BF grid would add no "
                "information. The RV error(s) are formal values from "
                "correlated bins; treat them as a lower limit.")
            break
        if recompute is None or dv_user is not None:
            notes.append(
                f"Undersampled: {which} — narrower than "
                f"{BF_MIN_SIGMA_BINS:g} bins of the dv = {dv:.2f} km/s "
                "grid, so the RV error(s) cannot be trusted; rerun with "
                f"--dv {dv_new:.2f} (or finer) to measure properly.")
            break
        try:
            new_result = recompute(dv_new)
        except (ValueError, RuntimeError) as exc:
            notes.append(f"Undersampled component {which}: the BF could "
                         f"not be recomputed at dv = {dv_new:.2f} km/s "
                         f"({exc.__class__.__name__}: {exc}); the RV error "
                         "is not trustworthy.")
            break
        new_comps, new_model = fit_bf_peaks(new_result["velocity"],
                                            new_result["bf_smooth"],
                                            **fit_kw)
        notes.append(
            f"Velocity step auto-refined {dv:.2f} -> {dv_new:.2f} km/s: "
            f"{which} — too narrow for the previous grid "
            f"(< {BF_MIN_SIGMA_BINS:g} bins). The RVs and errors reported "
            "are from the refined grid; pass --dv to fix the step yourself.")
        bf_result, comps, model = new_result, new_comps, new_model
    if verbose:
        for n in notes:
            print(f"  {n}")
    return bf_result, comps, model, notes


def blend_diagnosis(bf_result, comps, inst_fwhm=None, epsilon=0.6,
                    fit_trim=0, verbose=True):
    """Check a multi-component fit for an unresolved blend and, when it
    finds one, measure the blended profile as a single component.

    Near conjunction the two stars merge into one BF peak. Splitting it
    into two Gaussians is then mathematically degenerate — the pair can
    slide along each other at almost constant chi^2 — which shows up as a
    huge or nan rv_err. The velocities are not recoverable, but the
    blend as a whole still has a well-defined centre, so the
    single-component fit of the same profile is reported alongside as
    the one number that is actually measured.

    Returns (note, single) where `single` is the one-component dict (or
    None when the fit is not degenerate).
    """
    if len(comps) < 2:
        return None, None
    errs = [c.get("rv_err", np.nan) for c in comps]
    bad = [i for i, e in enumerate(errs)
           if not np.isfinite(e) or e > BF_DEGENERATE_ERR]
    rvs = sorted(c["rv"] for c in comps)
    sep = min(b - a for a, b in zip(rvs, rvs[1:]))
    unresolved = inst_fwhm is not None and sep < inst_fwhm
    if not bad and not unresolved:
        return None, None
    why = []
    if bad:
        why.append("C" + "/C".join(str(i + 1) for i in bad)
                   + " has an RV error of "
                   + ", ".join(("nan" if not np.isfinite(errs[i])
                                else f"{errs[i]:.1f} km/s") for i in bad))
    if unresolved:
        why.append(f"the components are {sep:.1f} km/s apart, below the "
                   f"{inst_fwhm:.1f} km/s instrumental FWHM")
    try:
        single, _ = fit_bf_peaks(bf_result["velocity"],
                                 bf_result["bf_smooth"], components=1,
                                 inst_fwhm=inst_fwhm, epsilon=epsilon,
                                 fit_trim=fit_trim)
        single = single[0]
    except (ValueError, RuntimeError):
        single = None
    if single is not None and not (np.isfinite(single.get("rv_err", np.nan))
                                   and single["rv_err"]
                                   <= BF_DEGENERATE_ERR):
        # The blend as a whole is not measurable either (usually a grid too
        # coarse for the profile); quoting it would be worse than silence.
        single = None
    note = ("Blended / degenerate fit: " + "; ".join(why)
            + ". The multi-component split is not constrained by the data.")
    if single is not None:
        note += (f" Single-component fit of the same profile: RV = "
                 f"{single['rv']:.4f} +/- {single['rv_err']:.4f} km/s, "
                 f"sigma = {single['sigma']:.2f} km/s.")
    else:
        note += (" A single-component fit of the blend is not well "
                 "constrained either.")
    if verbose:
        print(f"  {note}")
    return note, single


def identify_components(comps, phase, gamma=None):
    """Assign star identities to fitted BF components from the orbital
    phase, following the DDO close-binary series and Mennickent et al.
    (2010), where RV_i(phase) = gamma + K_i sin(2 pi phase + phi_i):
    right after the primary eclipse (0 < phase < 0.5) the primary is
    blueshifted with respect to the systemic velocity and the secondary
    redshifted; between phase 0.5 and 1 the signs swap. With three
    components the narrowest profile is taken as the tertiary — additional
    components appear as sharp peaks near the systemic velocity (Pribulla
    & Rucinski series) — and is listed last. gamma defaults to the
    area-weighted mean RV of the binary pair.

    Fit artifacts are excluded from the stellar pair first: a component
    whose error is nan/zero/huge (a parameter pinned at a fit bound) or
    whose width sits at the fit bounds is not a star, and letting it
    into the pair mislabels the one real peak (e.g. a lone secondary
    ends up in the C1 column). When only ONE credible peak remains and
    gamma is known, the peak is assigned to the primary or secondary
    from its side of gamma alone and the artifacts keep the remaining
    slots.

    Returns (ordered comps, note string).
    """
    comps = list(comps)

    def looks_stellar(c):
        err = c.get("rv_err")
        if err is None or not np.isfinite(err) or not 0.0 < err < 100.0:
            return False
        sig = c.get("sigma")
        if sig is not None and not 1.05 < sig < 195.0:
            return False
        return True

    good = [c for c in comps if looks_stellar(c)]
    junk = [c for c in comps if not any(c is g for g in good)]

    if len(good) == 1 and gamma is not None:
        c = good[0]
        blue = c["rv"] < gamma
        is_c1 = blue if (phase % 1.0) < 0.5 else not blue
        ordered = [c] + junk if is_c1 else [junk[0], c] + junk[1:]
        note = (f"Component labels from the orbital phase ({phase:.4f}) "
                f"and gamma = {gamma:+.2f} km/s: single credible peak "
                f"-> {'C1 (primary)' if is_c1 else 'C2 (secondary)'}; "
                "the remaining components are fit artifacts (nan/huge "
                "errors)")
        return ordered, note

    tertiary = None
    if len(good) >= 3:
        tertiary = min(good, key=lambda c: c["sigma"])
        pair = [c for c in good if c is not tertiary]
    elif len(good) == 2:
        pair = good
    else:
        # no (or unusable) quality information: original behaviour
        junk = []
        if len(comps) >= 3:
            tertiary = min(comps, key=lambda c: c["sigma"])
            pair = [c for c in comps if c is not tertiary]
        else:
            pair = comps
    if gamma is None:
        weights = [max(abs(c["amp"] * c["sigma"]), 1e-12) for c in pair]
        gamma = (sum(c["rv"] * w for c, w in zip(pair, weights))
                 / sum(weights))
        gamma_src = "estimated from the BF-area-weighted mean"
    else:
        gamma_src = "given"
    blue, red = sorted(pair, key=lambda c: c["rv"])
    if (phase % 1.0) < 0.5:
        ordered = [blue, red]
    else:
        ordered = [red, blue]
    note = (f"Component labels from the orbital phase ({phase:.4f}): "
            f"C1 = primary ({'blue' if (phase % 1.0) < 0.5 else 'red'}"
            f"-shifted side of gamma = {gamma:+.2f} km/s, {gamma_src})")
    if tertiary is not None:
        ordered.append(tertiary)
        note += "; C3 = narrowest peak (tertiary near systemic velocity)"
    ordered.extend(junk)
    return ordered, note


def make_ccf_figure(result, vbary=0.0, bary_applied=False):
    """CCF figure in the measured velocity frame. The legend quotes the
    RV of the Gaussian fit exactly as measured; when the barycentric
    correction is applied to the results, v_bary and the corrected value
    (RV + v_bary) follow on the lines right below it — the fit itself is
    never altered. The result text file carries raw RV, corrected RV and
    v_bary in full."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 6))
    ax0, ax1 = fig.subplots(2, 1, sharex=True)

    for ccf in result["ccf_orders"]:
        ax0.plot(result["rv_grid"], ccf, lw=0.6, alpha=0.5)
    ax0.set_ylabel("CCF (per order)")

    show_bary = bool(vbary) and bary_applied
    mask_mode = result.get("mode") == "mask"
    total_label = ("Total CCF (mean line depth)" if mask_mode
                   else "Total CCF (normalized)")
    fit_name = ("Gaussian + linear baseline" if mask_mode
                else "Gaussian fit")
    ax1.plot(result["rv_grid"], result["ccf_total"], "k-", lw=1.2,
             label=total_label)
    ax1.plot(result["rv_grid"],
             eval_bf_model(result["rv_grid"], result["popt"]),
             "r-", lw=2, alpha=0.7,
             label=(f"{fit_name}: RV = {result['rv']:.3f} "
                    f"± {result['rv_err']:.3f} km/s"))
    if show_bary:
        ax1.plot([], [], " ", label=f"v_bary = {vbary:+.3f} km/s")
        ax1.plot([], [], " ",
                 label=(f"RV + v_bary = "
                        f"{apply_barycentric(result['rv'], vbary):.3f} "
                        "km/s"))
    ax1.axvline(result["rv"], color="r", ls=":", lw=1)
    ax1.set_xlabel("RV [km/s]")
    ax1.set_ylabel("CCF")
    ax1.legend()
    fig.tight_layout()
    return fig


def make_bf_preview_figure(bf_result):
    """Smoothed BF alone, without any component fit: the assumption step
    of the guided workflow. A grid is drawn so the per-component peak
    height (Amp), position (RV) and width can be read off the axes."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    ax.plot(bf_result["velocity"], bf_result["bf_smooth"], color="0.35",
            lw=2.2, label="BF (smoothed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Radial velocity [km/s]")
    ax.set_ylabel("Broadening Function")
    ax.set_title("Smoothed BF preview — read Amp (y), RV (x) and width "
                 "per component for the assumptions", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def make_bf_figure(bf_result, comps, popt, bjd=None, phase=None,
                   vbary=0.0, bary_applied=True):
    """BF figure in the measured velocity frame. The legend and the
    annotations quote the RVs of the fit exactly as measured; when the
    barycentric correction is applied to the results, v_bary and the
    corrected values (C_i + v_bary) are listed as extra legend lines
    right below the fitted RVs — the fit and its numbers are never
    altered. The result text file carries the raw RV, the corrected RV
    and v_bary in full."""
    from matplotlib.figure import Figure
    v = bf_result["velocity"]
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    ax.plot(v, bf_result["bf"], color="0.85", lw=0.8, alpha=0.25,
            label="BF (raw)")
    ax.plot(v, bf_result["bf_smooth"], color="0.55", lw=2.5,
            label="BF (smoothed)")
    model = eval_bf_model(v, popt)
    ax.plot(v, model, "k-", lw=1.2, label="Total fit")

    offset = model_offset(popt["popt"])
    epsilon = popt.get("epsilon", 0.6)
    colors = ["C0", "C3", "C2"]
    for i, c in enumerate(comps, 1):
        if c.get("kind") == "rot":
            curve = multi_rot_profile(v, [c["amp"], c["rv"], c["sigma"]],
                                      epsilon=epsilon,
                                      inst_sigma_pix=popt.get(
                                          "inst_sigma_pix")) + offset
        else:
            curve = gauss_no(v, c["amp"], c["rv"], c["sigma"]) + offset
        ax.plot(v, curve, "--", color=colors[(i - 1) % len(colors)], lw=1.8,
                label=(f"C{i}: amp = {c['amp']:.4f}, "
                       f"RV = {c['rv']:.2f} km/s"))

    show_bary = bool(vbary) and bary_applied
    if show_bary:
        ax.plot([], [], " ", label=f"v_bary = {vbary:+.3f} km/s")
        for i, c in enumerate(comps, 1):
            ax.plot([], [], " ",
                    label=(f"C{i} + v_bary = "
                           f"{apply_barycentric(c['rv'], vbary):.2f} km/s"))

    ymin = float(min(np.min(bf_result["bf_smooth"]), np.min(model)))
    ymax = float(max(np.max(bf_result["bf_smooth"]), np.max(model)))
    span = max(ymax - ymin, 1e-12)
    ax.set_ylim(ymin - 0.15 * span, ymax + 0.35 * span)
    title = []
    if bjd is not None:
        title.append(f"BJD {bjd:.6f}")
    if phase is not None:
        title.append(f"phase {phase:.4f}")
    if title:
        ax.set_title("  |  ".join(title), fontsize=10)
    for i, c in enumerate(comps, 1):
        ax.axvline(c["rv"], color="r", ls=":", lw=1)
        ax.annotate(f"C{i}: {c['rv']:.2f} km/s",
                    (c["rv"], c["amp"] + offset),
                    textcoords="offset points", xytext=(6, 6), color="r")
    # Say which frame the axis is in: the BF is solved in the observed
    # frame, so the peaks and the C{i} annotations are measured values,
    # while result_BF.txt reports them barycentric-corrected.
    ax.set_xlabel("Radial velocity [km/s]"
                  + ("  (measured; +v_bary in the legend = reported)"
                     if show_bary else ""))
    ax.set_ylabel("Broadening Function")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


FIGURE_DPI = 600   # default PNG resolution; --dpi overrides


def save_figure(fig, outfile):
    fig.savefig(outfile, dpi=FIGURE_DPI)
    print(f"Figure saved: {outfile}")


# Diagnostic lines for the RV reliability check: two strong lines per
# 50-nm band over 400-700 nm, so whatever wavelength window the analysis
# used, the check has strong features to compare. Air wavelengths [A].
# (The 4583.84 A line is Fe II, not Fe I: the strong feature at that
# wavelength belongs to ionized iron, multiplet 38.)
DIAGNOSTIC_LINES = [
    # 400-450 nm
    ("Ca I 4226.73", 4226.728),
    ("Fe I 4383.55", 4383.545),
    # 450-500 nm
    ("Ba II 4554.03", 4554.029),
    ("Fe II 4583.84", 4583.837),
    # 500-550 nm
    ("Fe I 5269.54", 5269.541),
    ("Fe I 5328.04", 5328.039),
    # 550-600 nm
    ("Fe I 5572.84", 5572.842),
    ("Ca I 5588.75", 5588.749),
    # 600-650 nm
    ("Ca I 6122.22", 6122.217),
    ("Fe I 6302.49", 6302.494),
    # 650-700 nm
    ("Fe I 6663.44", 6663.441),
    ("Ca I 6717.68", 6717.681),
]


def build_rv_model(spec_wl, tpl_wl, tpl_flux, comps, epsilon=0.6):
    """Model spectrum on the data grid from the fitted components: the
    template broadened with each component's own profile, shifted to its
    measured (observed-frame) RV, and combined with the light
    contributions estimated from the BF areas (amp*sigma).

    The broadening is what makes the model comparable with the data at
    all. An observed spectrum is the template convolved with the
    broadening function, so a synthetic template - effectively of
    infinite resolution and rotationless - has far sharper and deeper
    lines than any real spectrum. Comparing it unbroadened understates
    the model line depths and destroys the pixel-to-pixel correlation
    whenever the star is rotating: for a vsini ~ 30 km/s star the peak
    correlation against the raw template is ~0.27, against the broadened
    one ~0.51. Each component is convolved with the profile the BF fit
    actually found for it - the rotational profile for a `rot`
    component, else a Gaussian of the fitted sigma (as a resolution
    R = c / FWHM), which already contains the instrumental,
    rotational and macroturbulent broadening together.
    """
    parts, weights = [], []
    for c in comps:
        sigma = abs(float(c.get("sigma") or 0.0))
        if c.get("kind") == "rot" and sigma > 0:
            bw, bfx = prepare_template(tpl_wl, tpl_flux, vsini=sigma,
                                       epsilon=epsilon, verbose=False)
        elif sigma > 0:
            bw, bfx = prepare_template(
                tpl_wl, tpl_flux, resolution=C_KMS / (2.35482 * sigma),
                verbose=False)
        else:
            bw, bfx = tpl_wl, tpl_flux
        parts.append(np.interp(spec_wl, doppler_shift(bw, c["rv"]), bfx))
        weights.append(abs(c.get("amp", 1.0)) * max(sigma, 1e-6))
    if len(parts) == 1:
        return parts[0]
    total = sum(weights)
    if total <= 0:
        weights, total = [1.0] * len(parts), float(len(parts))
    model = np.zeros_like(np.asarray(spec_wl, dtype=float))
    for part, wgt in zip(parts, weights):
        model += wgt * part
    return model / total


def line_check(spec_wl, spec_flux, tpl_wl, tpl_flux, comps,
               window=6.0, method="BF", epsilon=0.6, model=None):
    """Reliability check of the RV solution against real diagnostic
    lines: the observed normalized spectrum is compared with the model
    (template shifted by the measured RVs) in windows around two strong
    lines per 50-nm band over 400-700 nm (DIAGNOSTIC_LINES: Ca I 4226 /
    Fe I 4383, Ba II 4554 / Fe II 4583, Fe I 5269 / 5328, Fe I 5572 /
    Ca I 5588, Ca I 6122 / Fe I 6302, Fe I 6663 / Ca I 6717) — only the
    lines inside the analyzed wavelength range are used. Each window is
    centered on the Doppler-shifted
    position of the line for the primary component, so the comparison
    tracks the line wherever the measured RV puts it. For every line the
    residual RMS, the Pearson correlation and the line-depth ratio
    (observed/model) are computed — depth ratios near 1 and high
    correlations mean the model is trustworthy.

    `model` lets a caller supply its own model spectrum on `spec_wl`
    instead of the one built from the fitted component widths. TODCOR
    needs that: it fits no per-component width, its model is the
    composite the correlation actually maximised,
    (t1(v1) + alpha*t2(v2)) / (1 + alpha), out of templates that were
    already degraded to the instrumental resolution.

    Returns (figure, stats) where stats is a list of dicts per line.
    """
    from matplotlib.figure import Figure

    # the BF was solved on the cr_trimmed spectrum, so the check has to
    # judge the model against those same pixels: an untrimmed -1 clip
    # sentinel or cosmic spike would otherwise set the measured line
    # depth and swamp both the RMS and the correlation
    spec_flux, _ = cr_trim(spec_wl, spec_flux)
    if model is None:
        model = build_rv_model(spec_wl, tpl_wl, tpl_flux, comps,
                               epsilon=epsilon)
    else:
        model = np.asarray(model, dtype=float)
    # half-width of the line core in units of lambda, from the fitted
    # component width (sigma [km/s] -> dlambda/lambda)
    core_half = abs(float(comps[0].get("sigma") or 0.0)) / C_KMS

    shift = 1.0 + comps[0]["rv"] / C_KMS
    targets = [(name, w0 * shift) for name, w0 in DIAGNOSTIC_LINES
               if spec_wl.min() + window < w0 * shift
               < spec_wl.max() - window]
    if not targets:
        order = np.argsort(spec_flux)
        picked = []
        for idx in order:
            w0 = spec_wl[idx]
            if all(abs(w0 - p) > 4 * window for p in picked):
                picked.append(w0)
            if len(picked) == 3:
                break
        targets = [(f"line {w0:.0f}", w0) for w0 in sorted(picked)]

    stats = []
    fig = Figure(figsize=(9, 2.6 * len(targets)))
    axes = fig.subplots(len(targets), 1)
    if len(targets) == 1:
        axes = [axes]
    for ax, (name, w0) in zip(axes, targets):
        sel = (spec_wl >= w0 - window) & (spec_wl <= w0 + window)
        ws, os_, ms = spec_wl[sel], spec_flux[sel], model[sel]
        resid = os_ - ms
        rms = float(np.std(resid))
        corr = float(np.corrcoef(os_, ms)[0, 1]) if os_.size > 3 else np.nan
        # Both depths are read at the model's line centre, over the core
        # of the profile. Taking the observed depth as 1 - min(window)
        # instead would measure the deepest noise excursion of the whole
        # window: on a S/N ~ 50 spectrum that lands near 1.0 for every
        # line, whatever the star, and the ratio then says nothing.
        i_c = int(np.nanargmin(ms))
        w_c = float(ws[i_c])
        half = max(2.0 * float(np.median(np.diff(ws))), 0.5 * core_half * w_c)
        core = np.abs(ws - w_c) <= half
        d_obs = float(1.0 - np.median(os_[core]))
        d_mod = float(1.0 - ms[i_c])
        ratio = d_obs / d_mod if d_mod > 0 else np.nan
        stats.append(dict(name=name, wavelength=w0, rms=rms, corr=corr,
                          depth_obs=d_obs, depth_model=d_mod,
                          depth_ratio=ratio))

        ax.plot(ws, os_, "k-", lw=1.0, label="Observed (normalized)")
        ax.plot(ws, ms, "r-", lw=1.2, alpha=0.8, label="Model (broadened, shifted)")
        ax.plot(ws, resid + 1.0 + 1.2 * max(d_obs, d_mod), "-", color="0.6",
                lw=0.8, label="Residual (offset)")
        ax.axvline(w_c, color="0.8", ls=":", lw=1)
        ax.set_ylabel("Flux")
        ax.set_title(f"{name}  |  RMS = {rms:.4f}  r = {corr:.3f}  "
                     f"depth O/M = {ratio:.3f}", fontsize=10)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("Wavelength [Å]")
    fig.suptitle(f"{method} model reliability check", y=0.995)
    fig.tight_layout()
    return fig, stats


def save_payload(payload, directory=None):
    """Write the result text file and figure(s) of an analysis payload
    (cmd_ccf / cmd_bf). Separated from the analysis itself so the GUI
    can run first and write only when the Save button is pressed; the
    CLI saves immediately after the run as before.

    directory: write the files there under their own base names instead of
    the paths the run chose. The GUI's Save button asks the user for it;
    the CLI passes nothing and keeps writing where the command said.

    Returns the list of files written."""
    def dest(name):
        return (os.path.join(directory, os.path.basename(name))
                if directory else name)

    out = dest(payload["output"])
    with open(out, "w") as f:
        f.write(payload["file_text"])
    print(f"Results written: {out}")
    written = [out]
    for fig, name in payload["figs"]:
        figpath = dest(name)
        save_figure(fig, figpath)
        written.append(figpath)
    payload["saved"] = True
    payload["saved_files"] = written
    return written


def cmd_ccf(args):
    orders = read_spectrum(args.spectrum, args.format)
    tpl_wl, tpl_flux = load_template(args)
    if args.wave_min or args.wave_max:
        lo = args.wave_min * 10.0 if args.wave_min else -np.inf
        hi = args.wave_max * 10.0 if args.wave_max else np.inf
        orders = [(w[(w >= lo) & (w <= hi)], f[(w >= lo) & (w <= hi)])
                  for w, f in orders]
        orders = [(w, f) for w, f in orders if w.size > 10]

    tpl_wl, tpl_flux = prepare_template(tpl_wl, tpl_flux,
                                        resolution=get_resolution(args),
                                        vsini=args.vsini,
                                        epsilon=args.epsilon)

    print(f"Computing the CCF over {len(orders)} order(s)/segment(s)...")
    result = run_ccf(orders, tpl_wl, tpl_flux, args.rv_min, args.rv_max,
                     args.rv_step,
                     mode=getattr(args, "ccf_mode", "mask"),
                     mask_depth=getattr(args, "mask_depth", 0.05),
                     keep_balmer=getattr(args, "keep_balmer", False))

    ctx = get_target_context(args)
    vbary, bjd, phase = ctx["vbary"], ctx["bjd"], ctx["phase"]

    apply_bary = not getattr(args, "no_bary", False)
    rv_raw, rv_err = result["rv"], result["rv_err"]
    rv = apply_barycentric(rv_raw, vbary)
    tpl_name = args.template or f"PHOENIX T={args.teff}K"
    if result["mode"] == "mask":
        method_note = (f"weighted line mask, {result['n_lines']} lines "
                       "(Baranne 1996; Pepe 2002; Pino et al. 2018), "
                       "hydrogen series "
                       + ("kept" if getattr(args, "keep_balmer", False)
                          else "excluded"))
    else:
        method_note = "full-template cross-covariance"
    data_snr = estimate_snr(np.concatenate([f for _, f in orders]))
    snr_note = ("inf" if not np.isfinite(data_snr) else f"{data_snr:.0f}")
    lines = ["================ CCF RESULT ================",
             f"Normalized spectrum : {args.spectrum}",
             f"Synthetic template  : {tpl_name}",
             f"CCF method          : {method_note}",
             f"Continuum S/N       : ~{snr_note} per pixel"]
    if bjd is not None:
        lines.append(f"BJD = {bjd:.6f}"
                     + (f"   phase = {phase:.4f}" if phase is not None else ""))
    if apply_bary:
        lines.append(f"RV (uncorrected)           = {rv_raw:.4f} "
                     f"± {rv_err:.4f} km/s")
        lines.append(f"RV (barycentric corrected) = {rv:.4f} "
                     f"± {rv_err:.4f} km/s  (v_bary = {vbary:+.4f} km/s)")
    else:
        lines.append(f"RV = {rv_raw:.4f} ± {rv_err:.4f} km/s  "
                     "(no barycentric correction applied)")
        if vbary:
            lines.append(f"v_bary = {vbary:+.4f} km/s "
                         "(for information only, not applied)")
    if data_snr < 7.0:
        lines.append(LOW_DATA_SNR_NOTE.format(snr=data_snr))
    low_snr = result.get("peak_snr", np.inf) < 4.0
    if low_snr:
        lines.append(LOW_SNR_WARNING.format(snr=result["peak_snr"]))
    lines.append("============================================")
    summary = "\n".join(lines)
    print("\n" + summary)

    outfile = args.output or "result_CCF.txt"
    flines = [f"# Normalized spectrum : {args.spectrum}\n",
              f"# Synthetic template  : {tpl_name}\n",
              f"# CCF method : {method_note}\n",
              f"# continuum S/N ~ {snr_note} per pixel\n",
              "# barycentric correction applied: "
              + ("yes" if apply_bary else "no") + "\n"]
    if ctx.get("frame_note"):
        flines.append(f"# barycentric frame ({ctx['frame_note']})\n")
    if data_snr < 7.0:
        flines.append("# " + LOW_DATA_SNR_NOTE.format(snr=data_snr) + "\n")
    if apply_bary:
        flines.append("# method  BJD  phase  RV_raw[km/s]  RV_raw_err[km/s]  "
                      "RV_bary[km/s]  RV_bary_err[km/s]  bary_corr[km/s]\n")
        flines.append(f"CCF  {bjd if bjd is not None else 'nan'}  "
                      f"{f'{phase:.5f}' if phase is not None else 'nan'}  "
                      f"{rv_raw:.5f}  {rv_err:.5f}  "
                      f"{rv:.5f}  {rv_err:.5f}  {vbary:.5f}\n")
    else:
        flines.append(f"# v_bary [km/s] = {vbary:.5f}  "
                      "(computed for information only; NOT applied)\n")
        flines.append("# method  BJD  phase  RV[km/s]  RV_err[km/s]\n")
        flines.append(f"CCF  {bjd if bjd is not None else 'nan'}  "
                      f"{f'{phase:.5f}' if phase is not None else 'nan'}  "
                      f"{rv_raw:.5f}  {rv_err:.5f}\n")
    if low_snr:
        flines.append("# " + LOW_SNR_WARNING.format(snr=result["peak_snr"])
                      + "\n")

    plotfile = args.plot or "result_CCF.png"
    fig = make_ccf_figure(result, vbary=vbary, bary_applied=apply_bary)

    mwl = np.concatenate([w for w, _ in orders])
    mfx = np.concatenate([f for _, f in orders])
    srt = np.argsort(mwl)
    check_fig, stats = line_check(mwl[srt], mfx[srt], tpl_wl, tpl_flux,
                                  [dict(rv=result["rv"],
                                        sigma=result.get("sigma"))],
                                  method="CCF")
    print("Line check (observed vs model):")
    flines.append("# line_check: line  RMS  corr  depth_obs/model\n")
    for s in stats:
        print(f"  {s['name']:<12} RMS = {s['rms']:.4f}  "
              f"r = {s['corr']:.3f}  depth O/M = {s['depth_ratio']:.3f}")
        flines.append(f"# {s['name']}  {s['rms']:.5f}  {s['corr']:.4f}  "
                      f"{s['depth_ratio']:.4f}\n")

    payload = dict(method="CCF", fig=fig, text=summary,
                   output=outfile, plot=plotfile,
                   file_text="".join(flines),
                   figs=[(fig, plotfile),
                         (check_fig, "result_CCF_linecheck.png")],
                   saved=False)
    if not getattr(args, "no_save", False):
        save_payload(payload)
    return payload


def cosine_bell(n, k1, k2, k3, k4):
    """Brault & White (1971) cosine-bell bandpass over wavenumbers
    0..n-1: zero below k1, a raised-cosine ramp up to k2, unity to k3, a
    raised-cosine ramp down to k4, zero above. Real and even, hence
    zero-phase — it changes the shape of a profile but never its
    position, which is what a filter applied before an RV measurement
    must guarantee."""
    k = np.arange(n, dtype=float)
    f = np.zeros(n)
    f[(k >= k2) & (k <= k3)] = 1.0
    up = (k >= k1) & (k < k2)
    f[up] = 0.5 * (1.0 - np.cos(np.pi * (k[up] - k1) / max(k2 - k1, 1e-9)))
    dn = (k > k3) & (k <= k4)
    f[dn] = 0.5 * (1.0 + np.cos(np.pi * (k[dn] - k3) / max(k4 - k3, 1e-9)))
    return f


TODCOR_MIN_PIX = 256


def load_todcor_engine():
    """Import the vendored TODCOR engine, with a clear error if missing."""
    here = os.path.dirname(os.path.abspath(__file__))
    vendor = os.path.join(here, "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    try:
        import todcor as _todcor
    except ImportError as exc:
        raise SystemExit(
            "The TODCOR engine is missing. It is vendored at "
            f"{os.path.join(vendor, 'todcor.py')} (MIT, from "
            "https://github.com/Simchon/TODCOR); re-add it there.\n"
            f"({exc})")
    return _todcor


def log_lambda_grid(wl_lo, wl_hi, dv):
    """Uniform ln-lambda grid with `dv` km/s per pixel.

    TODCOR measures shifts in PIXELS, so a pixel must be a constant
    velocity: Zucker (2012) states the requirement as n = A ln(lambda)+B.
    On any other sampling the lag-to-velocity conversion is wrong.
    """
    if not (wl_hi > wl_lo > 0):
        raise SystemExit("Empty wavelength overlap between the spectrum and "
                         "the templates - check --wave-min/--wave-max and "
                         "that the templates cover the spectrum.")
    step = np.log1p(dv / C_KMS)
    n = int(np.floor((np.log(wl_hi) - np.log(wl_lo)) / step))
    if n < TODCOR_MIN_PIX:
        raise SystemExit(f"Only {n} pixels in the common wavelength range at "
                         f"--dv {dv:g} km/s; TODCOR needs at least "
                         f"{TODCOR_MIN_PIX}. Widen the range or reduce --dv.")
    return np.exp(np.log(wl_lo) + np.arange(n) * step)


def resample_orders(orders, grid):
    """Merge echelle orders onto one log-lambda grid.

    Overlapping orders are averaged where they overlap; gaps are filled
    with the continuum (1.0) so a missing order cannot create a spurious
    correlation feature. Returns the resampled flux and the fraction of
    the grid that real data actually covered.
    """
    total = np.zeros(grid.size)
    count = np.zeros(grid.size)
    for wl, fx in orders:
        wl = np.asarray(wl, float)
        fx = np.asarray(fx, float)
        good = np.isfinite(wl) & np.isfinite(fx)
        wl, fx = wl[good], fx[good]
        if wl.size < 2:
            continue
        order = np.argsort(wl)
        wl, fx = wl[order], fx[order]
        inside = (grid >= wl[0]) & (grid <= wl[-1])
        if not inside.any():
            continue
        total[inside] += np.interp(grid[inside], wl, fx)
        count[inside] += 1.0
    covered = count > 0
    flux = np.ones(grid.size)
    flux[covered] = total[covered] / count[covered]
    return flux, float(covered.mean())


def clean_for_correlation(wl, flux, poly_order=3, window=200.0):
    """Make one order fit to correlate: reject spikes, put the continuum
    at 1.

    A correlation is a variance-weighted comparison, so it is dominated
    by whatever has the largest amplitude. In a pipeline-normalized
    echelle order that is not the lines but the leftovers — cosmics,
    dead pixels, order-edge roll-off — which is why the raw Neo-Narval
    `st3` orders (flux running -1 to +10 around a continuum near 0.7)
    give a correlation peak of only ~0.1. Clipping those and re-flattening
    the continuum is what puts the stellar lines back in charge.
    """
    wl = np.asarray(wl, float)
    flux = np.asarray(flux, float)
    good = np.isfinite(wl) & np.isfinite(flux)
    if good.sum() < 32:
        return wl, np.ones_like(wl)
    wl, flux = wl[good], flux[good]
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med))) * 1.4826
    if mad <= 0:
        mad = float(np.std(flux)) or 1.0
    spike = (flux > med + 5.0 * mad) | (flux < med - 10.0 * mad)
    flux[spike] = med
    try:
        norm = np.asarray(normalize_continuum(wl, flux,
                                              poly_order=poly_order,
                                              iterations=20,
                                              window=window)[0], float)
        if norm.shape == flux.shape and np.all(np.isfinite(norm)):
            return wl, norm
    except Exception:
        pass
    return wl, flux / med if med else flux


def todcor_lag(args, dv):
    """Maximum lag m in pixels for a +-max_rv km/s search."""
    return int(round(float(getattr(args, "max_rv", None) or 200.0) / dv))


def extend_log_grid(grid, dv, m):
    """Continue a log-lambda grid by m pixels at each end.

    The grid is geometric with ratio 1 + dv/c, so extending it is a
    multiplication - the added pixels carry the same velocity step as the
    rest, which is what makes the lag-to-velocity conversion valid over
    the whole search.
    """
    ratio = 1.0 + dv / C_KMS
    return grid[0] * ratio ** np.arange(-m, grid.size + m, dtype=float)


def prep_todcor_arrays(args, orders, t1, t2):
    """Put the observed spectrum and both templates on one log-lambda
    grid, the templates on a grid extended by m pixels at each end.

    That extension is what selects the EXACT TODCOR: the engine needs
    len(t1) = len(t2) = len(obs) + 2m to normalize every correlation over
    the spectral bins that actually overlap at each lag, the way
    Tonry & Davis (1979) prescribe and saphires follows. Given equal
    lengths instead, it has no data to slide into the window, warns, and
    falls back to building the template-template term from a single 1-D
    correlation with global standard deviations - the "regular" TODCOR.
    Since that term is precisely what models the blend, the approximation
    is worst where TODCOR is used: two peaks close together.

    Returns (grid, obs, tpl1, tpl2, dv, coverage, m).
    """
    dv = float(getattr(args, "dv", None) or 1.0)
    m = todcor_lag(args, dv)
    ratio = 1.0 + dv / C_KMS
    pad = ratio ** m
    lo_obs = min(w.min() for w, _ in orders)
    hi_obs = max(w.max() for w, _ in orders)
    lo_tpl = max(t1[0].min(), t2[0].min())
    hi_tpl = min(t1[0].max(), t2[0].max())
    lo = max(lo_obs, lo_tpl * pad)
    hi = min(hi_obs, hi_tpl / pad)
    if args.wave_min:
        lo = max(lo, args.wave_min * 10.0)
    if args.wave_max:
        hi = min(hi, args.wave_max * 10.0)
    grid = log_lambda_grid(lo + 1.0, hi - 1.0, dv)
    grid_ext = extend_log_grid(grid, dv, m)
    if getattr(args, "no_clean", False):
        use = orders
    else:
        use = [clean_for_correlation(w, f) for w, f in orders]
    obs, covered = resample_orders(use, grid)
    a1 = np.interp(grid_ext, *t1)
    a2 = np.interp(grid_ext, *t2)
    return grid, obs, a1, a2, dv, covered, m


def todcor_valid_mask(corr, dv, min_sep=0.0):
    """Cells of a TODCOR surface that may hold the maximum.

    Two things are excluded.

    The v1 == v2 ridge, because it is degenerate: two components at the
    same velocity are indistinguishable from one star, so the flux ratio
    there is not a measurable quantity. With ONE template used for both
    components - the normal choice for a twin binary - that shows up as
    hard NaN: t1 and t2 are then bit-identical, so ccf1 == ccf2 and the
    template-template correlation is exactly 1 on the diagonal, which
    makes the engine's optimal-alpha expression 0/0
    (vendor/todcor.py:306, `(ccf1*ccf12 - ccf2)/(ccf2*ccf12 - ccf1)`).
    numpy's argmax then returns the first NaN and the fit rails at
    -max_rv. Masking the ridge is not a workaround for that NaN; the
    ridge was never a legitimate maximum.

    Anything else non-finite, for the same reason.

    `min_sep` widens the excluded band to |v1 - v2| < min_sep, for
    near-identical templates where the ridge is merely ill-conditioned
    rather than exactly singular.
    """
    n = corr.shape[0]
    lag = np.arange(n)
    sep = np.abs(lag[:, None] - lag[None, :]) * dv
    ok = np.isfinite(corr) & (sep > max(float(min_sep), 0.0))
    return ok


def todcor_peak_index(corr, dv, min_sep=0.0):
    """Index of the largest legitimate cell of a TODCOR surface."""
    ok = todcor_valid_mask(corr, dv, min_sep)
    if not ok.any():
        raise ValueError("the whole TODCOR surface is masked or non-finite")
    # Fill with a finite floor rather than -inf: the masked cells can fall
    # inside the sub-pixel fit patch of a peak that sits against the mask
    # edge, and infinities there destroy the quadratic fit.
    floor = float(np.min(corr[ok])) - 1.0
    masked = np.where(ok, corr, floor)
    return np.unravel_index(np.argmax(masked), corr.shape), masked


def todcor_vel_alpha(engine, obs, t1, t2, m, dv, min_sep=0.0):
    """`todcorVelAlpha` with the degenerate ridge excluded.

    The vendored routine picks its patch centre with a bare
    `np.argmax(M)`, which is exactly the step that fails on identical
    templates, and its own `ccfInput=` shortcut cannot be used to work
    around it (that branch reads a name it never binds). So the peak
    search is repeated here on the masked surface and the vendored
    3-D (v1, v2, alpha) quadratic refinement is then applied around the
    cell we chose - same fit, same error model as
    vendor/todcor.py:todcorVelAlpha, only the starting cell differs.

    Returns (vel, alpha, peak, sig) like the vendored function.
    """
    d_alpha, alpha_rad, rad = 0.01, 1, 1
    corr, alpha_m, ccfs = engine.todcor(obs, t1, t2, m, outAll=True)
    idx, _ = todcor_peak_index(corr, dv, min_sep)
    idx = np.asarray(idx)
    shape = np.array(corr.shape)
    center = shape // 2
    best_alpha = alpha_m[tuple(idx)]

    if np.any((idx < rad) | (idx >= shape - rad)):
        return ((idx - center) * dv, best_alpha, corr[tuple(idx)],
                np.full(3, np.nan))

    a_vec = np.arange(-alpha_rad, alpha_rad + 1)
    v_vec = np.arange(-rad, rad + 1)
    grid = np.empty((4, 2 * rad + 1, 2 * rad + 1, 2 * alpha_rad + 1))
    grid[1] = v_vec[:, None, None]
    grid[2] = v_vec[None, :, None]
    grid[3] = a_vec[None, None, :]
    for i, a in enumerate(a_vec):
        mi, _ = engine.todcor(ccfInput=ccfs, alpha=best_alpha + a * d_alpha)
        grid[0, :, :, i] = mi[idx[0] - rad:idx[0] + rad + 1,
                              idx[1] - rad:idx[1] + rad + 1]

    flat = grid.reshape(4, -1)
    powers = lambda x, y, z: np.array([x * x, y * y, z * z, x * y, x * z,
                                       y * z, x, y, z, np.ones_like(x)])
    A = powers(flat[1], flat[2], flat[3]).T
    cf, *_ = np.linalg.lstsq(A, flat[0], rcond=None)
    H = np.array([[2 * cf[0], cf[3], cf[4]],
                  [cf[3], 2 * cf[1], cf[5]],
                  [cf[4], cf[5], 2 * cf[2]]])
    grad = np.array([cf[6], cf[7], cf[8]])
    try:
        delta = -np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        delta = np.zeros(3)
    if not np.all(np.isfinite(delta)) or np.any(np.abs(delta[:2]) > rad):
        # The patch is a shoulder, not a maximum - the quadratic extrapolates
        # far outside it. Happens when --min-sep pushes the search onto the
        # flank of the excluded ridge. Keep the grid cell instead.
        delta = np.zeros(3)
    val = powers(delta[0], delta[1], delta[2]) @ cf
    try:
        cov = (1 - min(val, 0.99999) ** 2) / obs.size / val * -np.linalg.inv(H)
        sig = np.sqrt(np.diag(cov)) * np.array([dv, dv, d_alpha])
    except np.linalg.LinAlgError:
        sig = np.full(3, np.nan)
    return ((idx - center + delta[:2]) * dv, best_alpha + delta[2] * d_alpha,
            val, sig)


def todcor_brightest_first(v1, v2, alpha, err, twin):
    """Put the brighter component in slot 1 when the labels are free.

    With one template for both components the surface is exactly
    symmetric under (v1, v2, alpha) -> (v2, v1, 1/alpha): the two
    mirrors have identical correlation, so which one the argmax lands
    on is decided by rounding noise and the component numbers came out
    scrambled from epoch to epoch. Fix the convention here - component
    1 is the one carrying more light, i.e. alpha <= 1 - which is the
    same rule the BF path uses when it sorts C1/C2 by area, so the two
    methods' component numbers refer to the same star.

    Only legitimate when the templates are identical. With two
    different templates the labels mean 'the star that matches t1' and
    'the star that matches t2', which is a real identification and must
    not be reordered; the caller passes twin=False for that.
    """
    if not twin or not (np.isfinite(alpha) and alpha > 1.0):
        return v1, v2, alpha, err
    err = np.asarray(err, float).copy()
    err[0], err[1] = err[1], err[0]
    if err.size > 2 and np.isfinite(err[2]):
        err[2] = err[2] / alpha ** 2        # sigma(1/alpha), first order
    return v2, v1, 1.0 / alpha, err


def todcor_solve(engine, obs, t1, t2, dv, max_rv, alpha=None, m=None,
                 min_sep=0.0):
    """Run TODCOR and return (v1, v2, alpha, peak, errors).

    `todcorVelAlpha` returns 5 values normally but only 4 when its alpha
    fit degenerates, so both shapes are handled here rather than at the
    call sites. `m` comes from prep_todcor_arrays, which sized the
    template padding for it; it is recomputed only for callers that
    build their arrays themselves. The v1 == v2 ridge is excluded from
    the peak search either way - see `todcor_valid_mask`.

    When the alpha is fitted and both components share a template the
    labels are canonicalized by light ratio - see
    `todcor_brightest_first`. A fixed --alpha is the user stating which
    component is which, so that branch is left exactly as asked.
    """
    if m is None:
        m = int(round(max_rv / dv))
    if alpha is not None:
        corr, alpha_m = engine.todcor(obs=obs, t1=t1, t2=t2, m=m, alpha=alpha)
        _, masked = todcor_peak_index(corr, dv, min_sep)
        vel, peak, err = engine.todcorVel(masked, obs.size, dv=dv)
        return (float(vel[0]), float(vel[1]), float(alpha), float(peak),
                np.asarray(list(err) + [np.nan], float), corr)
    vel, al, peak, err = todcor_vel_alpha(engine, obs, t1, t2, m, dv, min_sep)
    err = np.asarray(err, float)
    v1, v2, al, err = todcor_brightest_first(float(vel[0]), float(vel[1]),
                                             float(al), err,
                                             np.array_equal(t1, t2))
    # Recomputed after the relabelling, not before: with equal templates
    # the surface at 1/alpha is the exact transpose of the surface at
    # alpha, so this is the mirror whose maximum sits at (v1, v2) as
    # reported - the figure would mark the wrong cell otherwise.
    corr, _ = engine.todcor(obs=obs, t1=t1, t2=t2, m=m, alpha=float(al))
    return (v1, v2, float(al), float(peak), err, corr)


def make_todcor_figure(corr, dv, v1, v2, alpha, path, bjd=None, phase=None,
                       vbary=0.0, bary_applied=False):
    """The TODCOR surface as an image, with the two cuts through the
    maximum. The cuts are the point of the figure: each is what an
    ordinary 1-D CCF would look like if the OTHER component's velocity
    were already known, so the peak that a 1-D CCF cannot separate is
    clean here (Zucker 2012, figs. 3-4).

    The surface is computed in the frame the spectrum was observed in,
    so the axes and the marked maximum are the MEASURED velocities and
    are never shifted - moving them would misplace the peak against its
    own surface. When the barycentric correction is applied to the
    reported RVs, the figure therefore has to say so, or the numbers on
    it differ from the numbers in result_TODCOR.txt by v_bary with
    nothing to explain the gap. Same convention as make_bf_figure /
    make_ccf_figure: measured values plotted, corrected values quoted
    beside them."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 4.6))
    axes = fig.subplots(1, 3)
    m = corr.shape[0] // 2
    ax = np.arange(-m, m + 1) * dv
    i1 = int(np.argmin(np.abs(ax - v1)))
    i2 = int(np.argmin(np.abs(ax - v2)))

    a0 = axes[0]
    im = a0.imshow(corr.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[ax[0], ax[-1], ax[0], ax[-1]])
    a0.plot(v1, v2, "r+", ms=14, mew=2)
    a0.set_xlabel("primary velocity [km/s] (measured)")
    a0.set_ylabel("secondary velocity [km/s] (measured)")
    a0.set_title(f"TODCOR surface  (alpha = {alpha:.3f})")
    fig.colorbar(im, ax=a0, fraction=0.046)

    axes[1].plot(ax, corr[:, i2], "k-", lw=1.2)
    axes[1].axvline(v1, color="r", ls=":")
    axes[1].set_xlabel("primary velocity [km/s] (measured)")
    axes[1].set_ylabel("correlation")
    axes[1].set_title(f"cut at v2 = {v2:.2f} km/s")

    axes[2].plot(ax, corr[i1, :], "k-", lw=1.2)
    axes[2].axvline(v2, color="r", ls=":")
    axes[2].set_xlabel("secondary velocity [km/s] (measured)")
    axes[2].set_title(f"cut at v1 = {v1:.2f} km/s")

    for a in axes[1:]:
        a.set_xlim(min(v1, v2) - 120, max(v1, v2) + 120)
        a.grid(alpha=0.3)
    sub = f"BJD {bjd:.5f}" if bjd is not None else ""
    if phase is not None:
        sub += f"   phase {phase:.4f}"
    fig.suptitle(os.path.basename(str(path)) + ("   " + sub if sub else ""))

    # The frame line: without it the marked maximum (measured) and the
    # RVs in result_TODCOR.txt (corrected) differ by v_bary silently.
    if bary_applied and vbary:
        note = (f"plotted (measured): v1 = {v1:.3f}, v2 = {v2:.3f} km/s"
                f"      v_bary = {vbary:+.3f} km/s      "
                f"REPORTED (barycentric): "
                f"v1 = {apply_barycentric(v1, vbary):.3f}, "
                f"v2 = {apply_barycentric(v2, vbary):.3f} km/s")
        color = "#b00000"
    elif vbary:
        note = (f"plotted = reported: v1 = {v1:.3f}, v2 = {v2:.3f} km/s"
                f"      v_bary = {vbary:+.3f} km/s computed but NOT "
                "applied")
        color = "0.25"
    else:
        note = (f"plotted = reported: v1 = {v1:.3f}, v2 = {v2:.3f} km/s"
                "      (no barycentric correction)")
        color = "0.25"
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=9,
             color=color)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


def cmd_todcor(args):
    """Two-dimensional correlation (Zucker & Mazeh 1994) — the method to
    use when the two CCF peaks blend, which is precisely where fitting
    two Gaussians to a 1-D correlation stops being trustworthy.

    Needs the SPECTRUM and one template per component: it cannot be run
    on a pre-computed correlation profile, because it has to correlate
    the spectrum against a combination of two shifted templates.
    """
    engine = load_todcor_engine()
    if not args.template2 and args.teff2 is None:
        args.template2 = args.template1     # twin binary: one template, twice
    if getattr(args, "simulate", False):
        return todcor_self_calibration(args, engine)

    orders = read_spectrum(args.spectrum, args.format)
    t1 = load_template(make_args(template=args.template1, teff=args.teff1,
                                 logg=args.logg, feh=args.feh))
    t2 = load_template(make_args(template=args.template2, teff=args.teff2,
                                 logg=args.logg, feh=args.feh))
    res = get_resolution(args)
    t1 = prepare_template(t1[0], t1[1], resolution=res, vsini=args.vsini1,
                          epsilon=args.epsilon)
    t2 = prepare_template(t2[0], t2[1], resolution=res, vsini=args.vsini2,
                          epsilon=args.epsilon)

    grid, obs, a1, a2, dv, covered, m = prep_todcor_arrays(args, orders,
                                                           t1, t2)
    print(f"Common log-lambda grid: {grid[0]:.1f} - {grid[-1]:.1f} A, "
          f"{grid.size} px at {dv:g} km/s/px "
          f"({100 * covered:.0f}% covered by real data); templates "
          f"extended by {m} px on each side (exact TODCOR)")
    if covered < 0.5:
        print("WARNING: less than half the grid is covered by the spectrum; "
              "the gaps are filled with continuum and dilute the "
              "correlation. Restrict --wave-min/--wave-max to a covered "
              "region.")

    twin = np.array_equal(a1, a2)
    if twin:
        print("Both components use the same template: the surface is exactly "
              "symmetric under (v1, v2, alpha) -> (v2, v1, 1/alpha), so the "
              "component labels are a convention, not a measurement. They "
              "are fixed here by light ratio - component 1 is the brighter "
              "star (alpha <= 1), the same rule bf uses for C1/C2.")
    v1, v2, alpha, peak, err, corr = todcor_solve(
        engine, obs, a1, a2, dv, args.max_rv, alpha=args.alpha, m=m,
        min_sep=getattr(args, "todcor_min_sep", 0.0) or 0.0)

    ctx = get_target_context(args)
    vbary, bjd, phase = ctx["vbary"], ctx["bjd"], ctx["phase"]
    apply_bary = not getattr(args, "no_bary", False)
    rv1 = apply_barycentric(v1, vbary) if apply_bary else v1
    rv2 = apply_barycentric(v2, vbary) if apply_bary else v2
    tag = "" if apply_bary else "   (no barycentric correction applied)"

    lines = ["================ TODCOR RESULT ================",
             f"Spectrum            : {args.spectrum}",
             f"Template 1 (primary): {args.template1}",
             f"Template 2 (second.): {args.template2}",
             f"Method              : 2-D correlation against "
             "t1(v1) + alpha*t2(v2) (Zucker & Mazeh 1994; Zucker 2012)",
             f"Grid                : {grid[0]:.1f}-{grid[-1]:.1f} A, "
             f"{grid.size} px, {dv:g} km/s/px, |v| <= {args.max_rv:g} km/s",
             f"Normalization       : exact TODCOR - every correlation "
             f"normalized over the {grid.size} overlapping bins "
             "(Tonry & Davis 1979 convention, as in saphires)",
             f"Light ratio alpha   : {alpha:.4f}"
             + (" (fixed by --alpha)" if args.alpha is not None
                else f" (fitted, +-{err[2]:.4f})"
                if np.isfinite(err[2]) else " (fitted)"),
             f"Peak correlation    : {peak:.4f}"]
    if bjd is not None:
        lines.append(f"BJD = {bjd:.6f}"
                     + (f"   phase = {phase:.4f}" if phase is not None else ""))
    lines.append(f"Component 1: RV = {rv1:.4f} ± {err[0]:.4f} km/s" + tag)
    lines.append(f"Component 2: RV = {rv2:.4f} ± {err[1]:.4f} km/s" + tag)
    lines.append(f"Separation  : {v2 - v1:.4f} km/s")
    if np.isfinite(alpha) and alpha > 0:
        bright, faint = ((1, rv1), (2, rv2)) if alpha < 1 else ((2, rv2),
                                                                (1, rv1))
        lines.append(f"Brighter    : component {bright[0]} "
                     f"(RV = {bright[1]:.2f} km/s), by a factor "
                     f"{max(alpha, 1 / alpha):.2f} in flux")
    if twin:
        lines.append("NOTE: one template was used for both components, so "
                     "the surface is exactly mirror-symmetric and 1/2 is a "
                     "convention, not a spectroscopic identification. The "
                     "convention applied: component 1 = the brighter star "
                     "(alpha <= 1), matching bf's C1. It is a light-ratio "
                     "ordering, so near alpha = 1 the two components can "
                     "still trade places between epochs.")
    if apply_bary:
        lines.append(f"v_bary = {vbary:+.4f} km/s (applied)")
    elif vbary:
        lines.append(f"v_bary = {vbary:+.4f} km/s "
                     "(for information only, not applied)")
    if args.alpha is None and not (0.02 <= alpha <= 50.0):
        lines.append(f"WARNING: the fitted light ratio {alpha:.3g} is "
                     "implausible. In the self-calibration tests a badly "
                     "wrong alpha is the signature of a template that does "
                     "not match the component - check Teff/vsini before "
                     "trusting these RVs.")
    lines.append("===============================================")
    summary = "\n".join(lines)
    print("\n" + summary)

    flines = [f"# spectrum : {args.spectrum}\n",
              f"# template1 : {args.template1}\n",
              f"# template2 : {args.template2}\n",
              "# method : TODCOR 2-D correlation (Zucker & Mazeh 1994, "
              "ApJ 420, 806)\n",
              f"# grid : {grid[0]:.4f}-{grid[-1]:.4f} A, {grid.size} px, "
              f"{dv:g} km/s/px, max|v| = {args.max_rv:g} km/s\n",
              f"# todcor variant : exact (templates padded by {m} px per "
              "side; overlap-window normalization)\n",
              f"# data coverage of the grid : {covered:.4f}\n",
              f"# alpha : {alpha:.6f}"
              + (" (fixed)\n" if args.alpha is not None
                 else f"  err {err[2]:.6f}\n"),
              f"# peak correlation : {peak:.6f}\n",
              "# barycentric correction applied: "
              + ("yes" if apply_bary else "no") + "\n"]
    if not apply_bary:
        flines.append(f"# v_bary [km/s] = {vbary:.5f}  "
                      "(computed for information only; NOT applied)\n")
    flines.append("# component  BJD  phase  RV[km/s]  RV_err[km/s]\n")
    for k, (rv, e) in enumerate(((rv1, err[0]), (rv2, err[1])), start=1):
        flines.append(f"{k}  {bjd if bjd is not None else 'nan'}  "
                      f"{f'{phase:.5f}' if phase is not None else 'nan'}  "
                      f"{rv:.5f}  {e:.5f}\n")

    # Reliability check against real diagnostic lines, the same one the
    # ccf and bf commands run. The model is the composite the correlation
    # maximised - t1 at v1 plus alpha*t2 at v2, on the very grid that was
    # correlated - so what is plotted is what TODCOR actually claims the
    # spectrum looks like. A high peak correlation with a model that does
    # not track the observed line cores is the signature of a solution
    # that fits something other than the star.
    model = (np.interp(grid, doppler_shift(t1[0], v1), t1[1])
             + alpha * np.interp(grid, doppler_shift(t2[0], v2), t2[1])
             ) / (1.0 + alpha)
    sig = (C_KMS / res / 2.35482) if res else None
    check_fig, stats = line_check(
        grid, obs, t1[0], t1[1],
        [dict(rv=v1, sigma=sig), dict(rv=v2, sigma=sig)],
        method="TODCOR", model=model)
    print("Line check (observed vs model):")
    flines.append("# line_check: line  RMS  corr  depth_obs/model\n")
    for s in stats:
        print(f"  {s['name']:<12} RMS = {s['rms']:.4f}  "
              f"r = {s['corr']:.3f}  depth O/M = {s['depth_ratio']:.3f}")
        flines.append(f"# {s['name']}  {s['rms']:.5f}  {s['corr']:.4f}  "
                      f"{s['depth_ratio']:.4f}\n")

    outfile = args.output or "result_TODCOR.txt"
    plotfile = args.plot or "result_TODCOR.png"
    fig = make_todcor_figure(corr, dv, v1, v2, alpha, args.spectrum,
                             bjd=bjd, phase=phase, vbary=vbary,
                             bary_applied=apply_bary)
    payload = dict(method="TODCOR", fig=fig, text=summary, output=outfile,
                   plot=plotfile, file_text="".join(flines),
                   figs=[(fig, plotfile),
                         (check_fig, "result_TODCOR_linecheck.png")],
                   saved=False)
    if not getattr(args, "no_save", False):
        save_payload(payload)
    return payload


def todcor_self_calibration(args, engine):
    """Torres-style check (Zucker 2012, sect. 3, after Torres et al.
    2009): build NOISELESS composite spectra from the two templates at
    known velocities, run TODCOR on them, and report the residual bias.

    This is the honest way to decide whether a template pair is good
    enough, and how far the method can be trusted at the separations a
    given system actually shows — no observed spectrum is involved.
    """
    from scipy.ndimage import shift as ndshift
    t1 = load_template(make_args(template=args.template1, teff=args.teff1,
                                 logg=args.logg, feh=args.feh))
    t2 = load_template(make_args(template=args.template2, teff=args.teff2,
                                 logg=args.logg, feh=args.feh))
    res = get_resolution(args)
    t1 = prepare_template(t1[0], t1[1], resolution=res, vsini=args.vsini1,
                          epsilon=args.epsilon)
    t2 = prepare_template(t2[0], t2[1], resolution=res, vsini=args.vsini2,
                          epsilon=args.epsilon)

    dv = float(args.dv or 1.0)
    m = todcor_lag(args, dv)
    pad = (1.0 + dv / C_KMS) ** m
    lo = max(t1[0].min(), t2[0].min()) * pad + 1.0
    hi = min(t1[0].max(), t2[0].max()) / pad - 1.0
    if args.wave_min:
        lo = max(lo, args.wave_min * 10.0)
    if args.wave_max:
        hi = min(hi, args.wave_max * 10.0)
    grid = log_lambda_grid(lo, hi, dv)
    grid_ext = extend_log_grid(grid, dv, m)
    a1 = np.interp(grid_ext, *t1)
    a2 = np.interp(grid_ext, *t2)

    alpha = args.alpha if args.alpha is not None else 0.6
    v1 = 0.0
    seps = ([float(s) for s in args.sim_sep] if args.sim_sep
            else [10., 15., 20., 25., 30., 40., 60., 90.])

    lines = ["=========== TODCOR SELF-CALIBRATION ===========",
             f"Template 1 : {args.template1}",
             f"Template 2 : {args.template2}",
             f"Grid       : {grid[0]:.1f}-{grid[-1]:.1f} A, {grid.size} px, "
             f"{dv:g} km/s/px",
             f"Composite  : t1(v1) + {alpha:g} x t2(v2), noiseless",
             "Method     : Torres et al. (2009) style - measure the bias of "
             "TODCOR on spectra it should fit exactly",
             "",
             f"{'sep_true':>9}{'v1_rec':>9}{'v2_rec':>9}{'bias1':>8}"
             f"{'bias2':>8}{'sep_err':>9}{'alpha':>8}{'swap':>6}"]
    rows = []
    swaps = 0
    twin = np.array_equal(a1, a2)
    for sep in seps:
        v2 = v1 + sep
        obs = (ndshift(a1, v1 / dv, mode="grid-wrap", order=3)
               + alpha * ndshift(a2, v2 / dv, mode="grid-wrap", order=3))
        obs = obs[m:obs.size - m] / (1.0 + alpha)
        try:
            r1, r2, al, _peak, _err, _corr = todcor_solve(
                engine, obs, a1, a2, dv, args.max_rv, m=m)
        except Exception as exc:
            lines.append(f"{sep:9.1f}   FAILED: {exc}")
            continue
        # The surface has a mirror maximum at (v2, v1, 1/alpha); with equal
        # templates it is exactly as high, so which component gets label 1
        # is decided by rounding noise. Score the bias on the pair, not on
        # the labels, and count the flip separately - otherwise a perfectly
        # recovered pair is reported as a bias of one whole separation.
        flip = abs(r1 - v1) + abs(r2 - v2) > abs(r1 - v2) + abs(r2 - v1)
        if flip:
            r1, r2, al = r2, r1, (1.0 / al if al else np.inf)
            swaps += 1
        rows.append((sep, r1 - v1, r2 - v2))
        lines.append(f"{sep:9.1f}{r1:9.2f}{r2:9.2f}{r1 - v1:+8.2f}"
                     f"{r2 - v2:+8.2f}{(r2 - r1) - sep:+9.2f}{al:8.3f}"
                     f"{'yes' if flip else '-':>6}")
    if rows:
        worst = max(abs(b) for _, b1, b2 in rows for b in (b1, b2))
        lines += ["",
                  f"Largest bias over the tested separations: {worst:.2f} "
                  "km/s. This is the floor set by the templates and the "
                  "grid alone - the real measurement can only be worse.",
                  "A bias that grows at small separations, or an alpha far "
                  "from the input, means the template pair is wrong for "
                  "this system."]
        if swaps:
            lines.append(
                f"'swap' = the mirror solution won at {swaps} of "
                f"{len(rows)} separations: v1/v2 came back exchanged and "
                "alpha inverted. Velocities above are re-matched to the "
                "input pair; the raw run had them the other way round."
                + (" With one template for both components this is expected "
                   "at every separation - the two maxima are exactly equal, "
                   "so the label is a coin toss and only alpha identifies "
                   "the brighter star." if twin else
                   " Between epochs of a real system this is what makes "
                   "component labels jump; identify components by alpha or "
                   "by orbital phase, not by the label."))
    lines.append("===============================================")
    summary = "\n".join(lines)
    print("\n" + summary)
    outfile = args.output or "result_TODCOR_selfcal.txt"
    if not getattr(args, "no_save", False):
        with open(outfile, "w") as f:
            f.write(summary + "\n")
        print(f"Results written: {outfile}")
    return dict(method="TODCOR-selfcal", text=summary, output=outfile,
                fig=None, figs=[], file_text=summary + "\n",
                saved=not getattr(args, "no_save", False))


def solve_bf(args):
    """Load the spectrum and template of a `bf` run and solve the BF via
    the SVD. Shared by cmd_bf and the GUI's smoothed-BF preview: the
    assumption step of the guided workflow shows exactly the profile the
    component fit will run on, just without the fit.

    Returns (bf_result, spec_wl, spec_flux, tpl_wl, tpl_flux, inst_fwhm).
    """
    orders = read_spectrum(args.spectrum, args.format)

    if args.wave_min or args.wave_max:
        lo = args.wave_min * 10.0 if args.wave_min else -np.inf
        hi = args.wave_max * 10.0 if args.wave_max else np.inf
        orders = [(w[(w >= lo) & (w <= hi)], f[(w >= lo) & (w <= hi)])
                  for w, f in orders]
        orders = [(w, f) for w, f in orders if w.size > 10]
        if not orders:
            raise ValueError("No data left in the requested wavelength "
                             "range.")

    wl = np.concatenate([w for w, _ in orders])
    fx = np.concatenate([f for _, f in orders])
    order_ = np.argsort(wl)
    spec_wl, spec_flux = wl[order_], fx[order_]

    tpl_wl, tpl_flux = load_template(args)
    resolution = get_resolution(args)
    inst_fwhm = C_KMS / resolution if resolution else None
    tpl_wl, tpl_flux = prepare_template(
        tpl_wl, tpl_flux,
        resolution=(resolution if getattr(args, "degrade_template", False)
                    else None),
        vsini=args.vsini, epsilon=args.epsilon)

    def solve(dv, verbose=True):
        if len(orders) > 1:
            return compute_bf_orders(orders, tpl_wl, tpl_flux,
                                     vel_range=args.vel_range, dv=dv,
                                     svd_rcond=args.svd_rcond,
                                     smooth_kms=args.smooth,
                                     inst_fwhm=inst_fwhm)
        return compute_bf(spec_wl, spec_flux, tpl_wl, tpl_flux,
                          vel_range=args.vel_range, dv=dv,
                          svd_rcond=args.svd_rcond,
                          smooth_kms=args.smooth,
                          inst_fwhm=inst_fwhm, verbose=verbose)

    print("Solving the BF via SVD...")
    bf_result = solve(args.dv)
    print(f"  velocity step dv = {bf_result['dv']:.3f} km/s, "
          f"singular values kept: {bf_result['n_kept_sv']}/{bf_result['n_sv']}")
    return (bf_result, spec_wl, spec_flux, tpl_wl, tpl_flux, inst_fwhm,
            lambda dv: solve(dv, verbose=False))


def cmd_bf_preview(args):
    """Solve the BF and show only the smoothed profile — no component
    fit. The assumption step of the guided workflow: the user reads each
    component's approximate amp (y-axis), RV (x-axis) and width off this
    curve, then runs the full fit starting from those values."""
    bf_result = solve_bf(args)[0]
    fig = make_bf_preview_figure(bf_result)
    text = ("Smoothed BF preview - no components fitted yet.\n"
            f"velocity step dv = {bf_result['dv']:.3f} km/s\n"
            "Read the approximate peak height (Amp), position (RV) and "
            "width (sigma)\nof each component off the curve, enter them "
            "as the assumptions, then Run.")
    return dict(method="BF", fig=fig, text=text, preview=True, saved=False)


def cmd_bf(args):
    (bf_result, spec_wl, spec_flux,
     tpl_wl, tpl_flux, inst_fwhm, recompute) = solve_bf(args)

    bf_result, comps, popt, fit_notes = fit_bf_refined(
        bf_result, recompute=recompute, dv_user=args.dv,
        components=args.components,
        min_sep=args.min_sep,
        guesses=getattr(args, "guess", None),
        guess_amps=getattr(args, "guess_amp", None),
        guess_sigmas=getattr(args, "guess_sigma", None),
        profile=getattr(args, "profile", "gauss"),
        inst_fwhm=inst_fwhm,
        epsilon=args.epsilon,
        fit_trim=BF_FIT_TRIM)

    ctx = get_target_context(args)
    vbary, bjd, phase = ctx["vbary"], ctx["bjd"], ctx["phase"]
    apply_bary = not getattr(args, "no_bary", False)

    label_note = None
    if args.components >= 2:
        if getattr(args, "guess", None):
            label_note = ("Component labels follow the given initial "
                          "guesses: C1 = first guess, C2 = second, ...")
        elif phase is not None:
            comps, label_note = identify_components(
                comps, phase, gamma=getattr(args, "gamma", None))
        else:
            comps.sort(key=lambda c: abs(c["amp"] * c["sigma"]),
                       reverse=True)
            label_note = ("Component labels sorted by BF area: C1 = "
                          "largest light contribution. Use --guess or an "
                          "ephemeris (phase) to fix the identities.")

    # After the labelling, so the note refers to the final C1/C2/... names.
    blend_note, blend_single = blend_diagnosis(
        bf_result, comps, inst_fwhm=inst_fwhm, epsilon=args.epsilon,
        fit_trim=BF_FIT_TRIM)
    if blend_note:
        fit_notes = fit_notes + [blend_note]

    tpl_name = args.template or f"PHOENIX T={args.teff}K"
    data_snr = estimate_snr(spec_flux)
    snr_note = ("inf" if not np.isfinite(data_snr) else f"{data_snr:.0f}")
    lines = ["================ BF RESULT =================",
             f"Normalized spectrum : {args.spectrum}",
             f"Synthetic template  : {tpl_name}",
             f"Continuum S/N       : ~{snr_note} per pixel"]
    if bjd is not None:
        lines.append(f"BJD = {bjd:.6f}"
                     + (f"   phase = {phase:.4f}" if phase is not None else ""))
    for i, c in enumerate(comps, 1):
        wname = "vsini" if c.get("kind") == "rot" else "sigma"
        area = component_area(c, args.epsilon)
        if apply_bary:
            lines.append(f"Component {i}: RV (uncorrected)           = "
                         f"{c['rv']:.4f} ± {c['rv_err']:.4f} km/s"
                         f"   (amp={c['amp']:.4f}, area={area:.3f}, "
                         f"{wname}={c['sigma']:.2f} km/s)")
            lines.append(f"             RV (barycentric corrected) = "
                         f"{apply_barycentric(c['rv'], vbary):.4f} "
                         f"± {c['rv_err']:.4f} km/s")
        else:
            lines.append(f"Component {i}: RV = {c['rv']:.4f} "
                         f"± {c['rv_err']:.4f} km/s"
                         f"   (amp={c['amp']:.4f}, area={area:.3f}, "
                         f"{wname}={c['sigma']:.2f} km/s)")
    if vbary:
        if apply_bary:
            lines.append(f"v_bary = {vbary:+.4f} km/s")
        else:
            lines.append(f"v_bary = {vbary:+.4f} km/s "
                         "(for information only, not applied)")
    if label_note:
        lines.append(label_note)
    lines.extend(fit_notes)
    if args.components >= 2 and not getattr(args, "guess", None):
        lines.append("Tip: inspect the smoothed BF in the figure, then "
                     "rerun with --guess RV1 RV2 ... --guess-amp A1 A2 ... "
                     "(optionally --guess-sigma S1 S2 ...) to start the "
                     "fit from your visual estimates.")
    if data_snr < 7.0:
        lines.append(LOW_DATA_SNR_NOTE.format(snr=data_snr))
    bf_snr = profile_peak_snr(bf_result["velocity"], bf_result["bf_smooth"],
                              comps)
    if bf_snr < 4.0:
        lines.append(LOW_SNR_WARNING.format(snr=bf_snr))
    if len(comps) >= 2 and all(c["amp"] > 0 for c in comps):
        a1 = component_area(comps[0], args.epsilon)
        if a1 > 0:
            for i, c in enumerate(comps[1:], 2):
                lines.append(f"Light ratio (C{i}/C1, from BF areas) ≈ "
                             f"{component_area(c, args.epsilon) / a1:.3f}")
    lines.append("============================================")
    summary = "\n".join(lines)
    print("\n" + summary)

    outfile = args.output or "result_BF.txt"
    flines = [f"# Normalized spectrum : {args.spectrum}\n",
              f"# Synthetic template  : {tpl_name}\n",
              f"# continuum S/N ~ {snr_note} per pixel\n"]
    if data_snr < 7.0:
        flines.append("# " + LOW_DATA_SNR_NOTE.format(snr=data_snr) + "\n")
    if bjd is not None:
        flines.append(f"# BJD = {bjd:.6f}"
                      + (f"   phase = {phase:.5f}" if phase is not None
                         else "") + "\n")
    flines.append("# barycentric correction applied: "
                  + ("yes" if apply_bary else "no") + "\n")
    if ctx.get("frame_note"):
        flines.append(f"# barycentric frame ({ctx['frame_note']})\n")
    if getattr(args, "profile", "gauss") == "rot":
        flines.append("# fit profile: rotational (Gray 1992); the sigma "
                      "column is vsini [km/s]\n")
    if label_note:
        flines.append(f"# {label_note}\n")
    flines.append(f"# BF velocity step dv = {bf_result['dv']:.5f} km/s"
                  + ("" if args.dv else " (default from the instrumental "
                     "resolution)") + "\n")
    for n in fit_notes:
        flines.append(f"# {n}\n")
    if blend_single is not None:
        flines.append("# blend_single  RV[km/s]  RV_err[km/s]  amp  "
                      "sigma[km/s]\n")
        flines.append(f"# 1  {blend_single['rv']:.5f}  "
                      f"{blend_single['rv_err']:.5f}  "
                      f"{blend_single['amp']:.5f}  "
                      f"{blend_single['sigma']:.5f}\n")
    if apply_bary:
        flines.append("# component  RV_raw[km/s]  RV_raw_err[km/s]  "
                      "RV_bary[km/s]  RV_bary_err[km/s]  amp  sigma[km/s]  "
                      "bary_corr[km/s]\n")
        for i, c in enumerate(comps, 1):
            flines.append(f"{i}  {c['rv']:.5f}  {c['rv_err']:.5f}  "
                          f"{apply_barycentric(c['rv'], vbary):.5f}  "
                          f"{c['rv_err']:.5f}  "
                          f"{c['amp']:.5f}  {c['sigma']:.5f}  {vbary:.5f}\n")
    else:
        flines.append(f"# v_bary [km/s] = {vbary:.5f}  "
                      "(computed for information only; NOT applied)\n")
        flines.append("# component  RV[km/s]  RV_err[km/s]  amp  "
                      "sigma[km/s]\n")
        for i, c in enumerate(comps, 1):
            flines.append(f"{i}  {c['rv']:.5f}  {c['rv_err']:.5f}  "
                          f"{c['amp']:.5f}  {c['sigma']:.5f}\n")
    if bf_snr < 4.0:
        flines.append("# " + LOW_SNR_WARNING.format(snr=bf_snr) + "\n")

    plotfile = args.plot or "result_BF.png"
    fig = make_bf_figure(bf_result, comps, popt,
                         bjd=bjd, phase=phase, vbary=vbary,
                         bary_applied=apply_bary)

    check_fig, stats = line_check(spec_wl, spec_flux, tpl_wl, tpl_flux,
                                  comps, method="BF", epsilon=args.epsilon)
    print("Line check (observed vs model):")
    flines.append("# line_check: line  RMS  corr  depth_obs/model\n")
    for s in stats:
        print(f"  {s['name']:<12} RMS = {s['rms']:.4f}  "
              f"r = {s['corr']:.3f}  depth O/M = {s['depth_ratio']:.3f}")
        flines.append(f"# {s['name']}  {s['rms']:.5f}  {s['corr']:.4f}  "
                      f"{s['depth_ratio']:.4f}\n")

    payload = dict(method="BF", fig=fig, text=summary,
                   output=outfile, plot=plotfile,
                   file_text="".join(flines),
                   figs=[(fig, plotfile),
                         (check_fig, "result_BF_linecheck.png")],
                   saved=False)
    if not getattr(args, "no_save", False):
        save_payload(payload)
    return payload


def make_norm_figure(diag, wl, nf, tpl=None):
    """Two-panel normalization figure: raw + continuum fits, and the
    normalized spectrum overplotted on the synthetic template (the
    'interactive comparison with a synthetic spectrum' step). The
    wavelength axis is in nm, matching the unit of the output file."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 6))
    ax0, ax1 = fig.subplots(2, 1, sharex=True)

    for w, f, cont in diag:
        ax0.plot(w / 10.0, f, lw=0.5, alpha=0.6)
        ax0.plot(w / 10.0, cont, "r-", lw=1.0, alpha=0.8)
    ax0.set_ylabel("Raw flux")
    ax0.set_title("Raw spectrum and fitted continuum (red)")

    ax1.plot(wl / 10.0, nf, "k-", lw=0.6, label="Normalized spectrum")
    if tpl is not None:
        tw, tf = tpl
        sel = (tw >= wl.min()) & (tw <= wl.max())
        if sel.any():
            ax1.plot(tw[sel] / 10.0, tf[sel] / np.nanmax(tf[sel]), "C0-",
                     lw=0.8, alpha=0.7, label="Synthetic template")
    ax1.axhline(1.0, color="0.6", ls=":", lw=1)
    ax1.set_ylim(-0.1, 1.6)
    ax1.set_xlabel("Wavelength [nm]")
    ax1.set_ylabel("Normalized flux")
    ax1.legend()
    fig.tight_layout()
    return fig


def cmd_normalize(args):
    """Normalize a raw spectrum to the continuum and write an ASCII file
    that can be fed directly into the CCF/BF analysis."""
    window_a = args.window * 10.0 if getattr(args, "window", None) else None
    method = getattr(args, "fit_method", "poly") or "poly"
    desc = ("cubic B-spline" if method == "bspline"
            else f"poly order {args.poly_order}")
    print(f"Normalizing {args.spectrum} "
          f"({desc}, {args.iterations} iterations"
          + (f", {args.window:g} nm windows" if window_a else "") + ")...")
    # A Libre-ESpRIT level-2/3 'Intensity' column has already been
    # divided by the pipeline continuum, so this run re-flattens what is
    # left (usually harmless, sometimes useful) rather than normalizing a
    # raw spectrum. Worth saying, because the step is not needed at all
    # for ccf/bf/todcor, which read these files directly.
    if neonarval_header_info(args.spectrum):
        print("  Note: this is a Libre-ESpRIT product - its 'Intensity' "
              "column is ALREADY continuum-normalized by the DRS (the "
              "divisor is the 'Continuum' column). ccf/bf/todcor read "
              "the file directly; normalizing again only re-flattens "
              "what the pipeline left behind.")
    wl, nf, diag = normalize_spectrum_file(
        args.spectrum, args.format, poly_order=args.poly_order,
        iterations=args.iterations, low_clip=args.low_clip,
        high_clip=args.high_clip, window=window_a, method=method)

    tpl = None
    if args.template or args.teff is not None:
        tpl = prepare_template(*load_template(args),
                               resolution=get_resolution(args),
                               vsini=args.vsini, epsilon=args.epsilon)

    # Carry the target metadata into the output header so the later
    # CCF/BF run reads RA/Dec/DATE-OBS straight from the normalized file
    # (ascii_header_info) and can compute the barycentric correction.
    src = header_info(args.spectrum)
    ra_u, dec_u = parse_coord(getattr(args, "ra", None),
                              getattr(args, "dec", None))
    meta = {"object": getattr(args, "object", None) or src["object"],
            "obstime": getattr(args, "obstime", None) or src["obstime"],
            "ra": ra_u if ra_u is not None else src["ra"],
            "dec": dec_u if dec_u is not None else src["dec"],
            "exptime": src["exptime"]}
    hdr_lines = [f"wavelength [nm]  normalized flux; from {args.spectrum} "
                 f"({desc}, {args.iterations} iterations)"]
    if meta["object"]:
        hdr_lines.append(f"OBJECT = {meta['object']}")
    if meta["obstime"]:
        hdr_lines.append(f"DATE-OBS = {meta['obstime']}")
    if meta["exptime"] is not None:
        hdr_lines.append(f"EXPTIME = {meta['exptime']}")
    if src.get("epoch_note"):
        hdr_lines.append(str(src["epoch_note"]))
    if meta["ra"] is not None:
        hdr_lines.append(f"RA = {meta['ra']:.6f}")
    if meta["dec"] is not None:
        hdr_lines.append(f"DEC = {meta['dec']:.6f}")
    # Keep the observatory with the spectrum: without it the normalized
    # copy would fall back to the default site and quietly lose up to
    # 0.46 km/s of barycentric velocity.
    site_src = header_site(args.spectrum)
    if site_src:
        lon, lat, alt, _cards = site_src
        hdr_lines += [f"SITELON = {lon:.4f}", f"SITELAT = {lat:.4f}",
                      f"SITEALT = {alt:.1f}"]
    else:
        # No position in the source header (SOPHIE, HARPS s1d, ...): fall
        # back to the home observatory of the instrument, so the
        # normalized copy does not end up at the default site either.
        inst_site, inst_name = instrument_site(args.spectrum)
        coords = observatory_coords(inst_site) if inst_site else None
        if coords:
            hdr_lines += [f"SITELON = {coords[0]:.4f}",
                          f"SITELAT = {coords[1]:.4f}",
                          f"SITEALT = {coords[2]:.1f}",
                          f"SITEFROM = instrument {inst_name} "
                          f"({inst_site})"]

    # Carry the barycentric-frame evidence of the source file into the
    # output header so barycheck (and the batch header check) still
    # works on the normalized ASCII spectrum.
    try:
        bary = barycentric_frame_check(args.spectrum)
    except Exception:
        bary = None
    if bary is not None:
        # header_instrument() as the fallback: a SOPHIE file names its
        # instrument in no card at all, and without the name the
        # normalized copy could not resolve its observatory either.
        inst_txt = bary["instrument"] or header_instrument(args.spectrum)
        if inst_txt:
            hdr_lines.append(f"INSTRUME = {inst_txt}")
        if bary["frame"] is not None:
            hdr_lines.append(f"SPECSYS = {bary['frame']}")
        elif bary["verdict"] != "unknown":
            hdr_lines.append(f"BARYFRAME = {bary['verdict']}")
        if bary["berv"] is not None:
            hdr_lines.append(f"BERV = {bary['berv']:.5f}")

    outfile = args.output or \
        os.path.splitext(os.path.basename(args.spectrum))[0] + "_norm.txt"
    np.savetxt(outfile, np.column_stack([wl / 10.0, nf]),
               fmt="%.5f %.5f", header="\n".join(hdr_lines))
    print(f"Normalized spectrum written: {outfile} (wavelength in nm)")
    if meta["ra"] is not None or meta["obstime"]:
        print("  target metadata (RA/Dec/DATE-OBS) copied into the header "
              "for the barycentric correction.")
    if bary is not None and (bary["frame"] is not None
                             or bary["verdict"] != "unknown"):
        print("  barycentric-frame evidence copied into the header "
              f"(verdict: {bary['verdict']}) - barycheck keeps working "
              "on the normalized file.")

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


def compute_bjd(obstime, ra_deg, dec_deg, site, exptime=None):
    """Mid-exposure BJD_TDB (thesis: light curves and RVs are phased in
    Barycentric Julian Date, never local/UTC calendar time).

    `obstime` may be an ISOT stamp (converted: +exptime/2, light travel
    time to the barycenter, TDB scale) or a JD/BJD number, which is
    returned unchanged — the data are then assumed to be in BJD already,
    as in the reference document.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    t, is_jd = parse_time_input(obstime)
    if is_jd:
        return float(str(obstime).strip())
    if exptime:
        t = t + (float(exptime) / 2.0) * u.s
    loc = earth_location(site)
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

    colors = ["C0", "C3", "C2"]
    for j in range(ncomp):
        rv = [r["rv"][j] for r in rows]
        err = [r["rv_err"][j] for r in rows]
        ax.errorbar(x, rv, yerr=err, fmt="o", ms=5, capsize=2,
                    color=colors[j % len(colors)],
                    label=f"Component {j + 1}")
    ax.set_xlabel("Orbital phase" if folded else "BJD_TDB")
    ax.set_ylabel("RV [km/s]")
    ax.set_title("Radial velocity curve")
    ax.legend()
    fig.tight_layout()
    return fig


def make_bf_stack_figure(profiles):
    """BF profiles of all epochs stacked and sorted by orbital phase:
    each profile is offset vertically, its double-Gaussian fit
    overplotted, and labelled with its phase so the geometry near the
    primary (phase 0) and secondary (phase 0.5) eclipses is visible at
    a glance."""
    from matplotlib.figure import Figure

    have_phase = all(p.get("phase") is not None for p in profiles)
    key = "phase" if have_phase else "bjd"
    profiles = sorted(profiles, key=lambda p: (p.get(key) is None,
                                               p.get(key, 0.0)))
    amp = max(float(np.nanmax(p["bf"])) for p in profiles)
    step = 1.3 * amp

    fig = Figure(figsize=(9, max(5, 1.2 * len(profiles))))
    ax = fig.subplots()
    for k, p in enumerate(profiles):
        off = k * step
        ax.plot(p["velocity"], p["bf"] + off, "b-", lw=1.2)
        if p.get("fit") is not None:
            ax.plot(p["velocity"], p["fit"] + off, "r--", lw=1.0, alpha=0.8)
        if have_phase:
            label = f"φ = {p['phase']:.3f}"
        elif p.get("bjd") is not None and np.isfinite(p["bjd"]):
            label = f"BJD {p['bjd']:.4f}"
        else:
            label = p.get("name", "")
        ax.annotate(label, (p["velocity"][-1], off),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8, va="center")
    ax.set_xlabel("Radial velocity [km/s]")
    ax.set_ylabel("Broadening function + offset")
    title = "Broadening functions"
    if have_phase:
        title += " sorted by orbital phase"
    ax.set_title(title)
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


def cmd_batch(args):
    """Process a spectral time series into an RV curve file.

    For every input spectrum: optional continuum normalization (raw FITS
    series), BF or CCF measurement, barycentric correction and BJD_TDB
    computation. Times are always handled in BJD: they come from --bjd
    (one value per file, as in the reference document), a numeric
    --obstime, or the FITS header DATE-OBS (converted to mid-exposure
    BJD_TDB). In SB2/SB3 mode the component with the larger BF area
    (larger light contribution) is always reported as component 1, so
    the labels do not swap between epochs (initial guesses or an
    ephemeris phase override this ordering when given).

    Outputs: result_RV_curve.txt (file, BJD, phase, RV per component),
    result_RV_curve.png and, in BF mode, result_BF_profiles.png with all
    BF profiles stacked by orbital phase.
    """
    import glob
    files = sorted(set(sum((glob.glob(p) for p in args.spectra), [])))
    if not files:
        sys.exit("No files match the given pattern(s).")
    if args.bjd and len(args.bjd) != len(files):
        sys.exit(f"--bjd expects one value per file "
                 f"({len(files)} files, {len(args.bjd)} values given).")
    print(f"{len(files)} spectra to process.\n")

    tpl_wl, tpl_flux = load_template(args)
    resolution = get_resolution(args)
    inst_fwhm = C_KMS / resolution if resolution else None
    degrade = args.method == "ccf" or getattr(args, "degrade_template",
                                              False)
    tpl_wl, tpl_flux = prepare_template(
        tpl_wl, tpl_flux,
        resolution=(resolution if degrade else None),
        vsini=args.vsini, epsilon=args.epsilon)

    ra, dec = parse_coord(args.ra, args.dec)
    if ra is None and args.object:
        try:
            sim = query_simbad(args.object)
            ra, dec = sim["ra"], sim["dec"]
            extra = ""
            if sim["sptype"]:
                extra += f", SpT = {sim['sptype']}"
            if sim["vmag"] is not None:
                extra += f", V = {sim['vmag']:.2f}"
            print(f"SIMBAD: '{args.object}' -> RA = {ra:.5f} deg, "
                  f"Dec = {dec:.5f} deg{extra}\n")
        except ValueError as exc:
            print(f"Warning: {exc}\n")

    t0_is_hjd = False
    if args.t0 is None and getattr(args, "varastro", False) and args.object:
        try:
            va = query_varastro(args.object)
            args.t0, args.period = va["t0"], va["period"]
            t0_is_hjd = True
            print(f"VarAstro: {va['star']} -> T0 = {va['t0']:.5f} HJD, "
                  f"P = {va['period']:.7f} d  ({va['url']})\n")
        except ValueError as exc:
            print(f"Warning: {exc}\n")
    if args.t0 is not None and getattr(args, "t0_to_bjd", False):
        if ra is not None and dec is not None:
            t0_bjd = hjd_to_bjd(args.t0, ra, dec)
            print(f"Ephemeris T0 converted: HJD {args.t0:.6f} -> "
                  f"BJD {t0_bjd:.6f} (used for the analysis)\n")
            args.t0 = t0_bjd
        else:
            print("Note: HJD -> BJD conversion of T0 skipped "
                  "(needs target coordinates).\n")
    elif t0_is_hjd:
        print("Note: T0 is kept in HJD (VarAstro native unit); "
              "use --t0-to-bjd to convert it to BJD.\n")

    apply_bary = not getattr(args, "no_bary", False)

    # Header check: is the wavelength scale already barycentric?
    # (FITS headers, or the '# KEY = VALUE' comment header of
    # normalized ASCII spectra - normalize copies the evidence there)
    # Keyed by path, not a bare list: the per-file barycentric cross-check
    # below needs the pipeline BERV of THIS file, and files whose header
    # cannot be read are skipped here, so positions would not line up.
    checks = {}
    for p in files:
        try:
            checks[p] = barycentric_frame_check(p)
        except Exception:
            pass
    if checks:
        n_cor = sum(c["verdict"] == "corrected" for c in checks.values())
        n_not = sum(c["verdict"] == "not-corrected" for c in checks.values())
        if n_cor and apply_bary:
            msg = (f"{n_cor}/{len(checks)} file headers say the "
                   "wavelength scale is already barycentric (e.g. "
                   "SPECSYS = BARYCENT): applying the correction again "
                   "would double it.")
            if getattr(args, "force_bary", False):
                print("WARNING: " + msg
                      + " Proceeding anyway (--force-bary).\n")
            else:
                sys.exit(
                    "ERROR: " + msg + "\n"
                    "  -> re-run with --no-bary (v_bary is then recorded "
                    "in the result file as a note only), or override "
                    "with --force-bary if you are sure.\n"
                    "  -> per-file evidence: run the 'barycheck' "
                    "command.")
        elif n_not and not apply_bary:
            print(f"WARNING: {n_not}/{len(checks)} file headers say the "
                  "wavelength scale is topocentric (correction not "
                  "applied) but --no-bary is given - the measured RVs "
                  "will still contain the barycentric motion (run the "
                  "'barycheck' command for the per-file evidence).\n")
        elif n_cor == 0 and n_not == 0:
            print("Header check: no wavelength-frame information in the "
                  "file headers (run the 'barycheck' command for "
                  "details).\n")
        else:
            print(f"Header check: wavelength frame already barycentric "
                  f"in {n_cor}/{len(checks)} files - consistent with "
                  + ("--no-bary" if not apply_bary
                     else "applying the correction") + ".\n")

    ncomp = args.components
    if args.method == "ccf" and ncomp > 1:
        print(f"Note: the CCF method fits a single peak; --components "
              f"{ncomp} reduced to 1 (use --method bf for SB2/SB3).\n")
        ncomp = 1
    rows, profiles = [], []
    site_notes_seen = set()
    for i_file, path in enumerate(files):
        print(f"--- {os.path.basename(path)} ---")
        try:
            if args.normalize:
                wl, fx, _ = normalize_spectrum_file(
                    path, args.format, poly_order=args.poly_order,
                    iterations=args.iterations,
                    window=(args.window * 10.0
                            if getattr(args, "window", None) else None),
                    method=getattr(args, "fit_method", "poly") or "poly")
                orders = [(wl, fx)]
            else:
                orders = read_spectrum(path, args.format)
            if args.wave_min or args.wave_max:
                lo = args.wave_min * 10.0 if args.wave_min else -np.inf
                hi = args.wave_max * 10.0 if args.wave_max else np.inf
                orders = [(w[(w >= lo) & (w <= hi)],
                           f[(w >= lo) & (w <= hi)]) for w, f in orders]
                orders = [(w, f) for w, f in orders if w.size > 10]
                if not orders:
                    raise ValueError("no data in the wavelength range")
            wl = np.concatenate([w for w, _ in orders])
            fx = np.concatenate([f for _, f in orders])
            srt = np.argsort(wl)
            wl, fx = wl[srt], fx[srt]

            hdr = header_info(path)
            obstime = (args.bjd[i_file] if args.bjd
                       else args.obstime or hdr["obstime"])
            ra_i = ra if ra is not None else hdr["ra"]
            dec_i = dec if dec is not None else hdr["dec"]

            popt = None
            fit_notes, blend_single = [], None
            if args.method == "bf":
                def solve_epoch(dv, _orders=orders, _wl=wl, _fx=fx):
                    if len(_orders) > 1:
                        return compute_bf_orders(
                            _orders, tpl_wl, tpl_flux,
                            vel_range=args.vel_range, dv=dv,
                            svd_rcond=args.svd_rcond,
                            smooth_kms=args.smooth, inst_fwhm=inst_fwhm)
                    return compute_bf(_wl, _fx, tpl_wl, tpl_flux,
                                      vel_range=args.vel_range, dv=dv,
                                      svd_rcond=args.svd_rcond,
                                      smooth_kms=args.smooth,
                                      inst_fwhm=inst_fwhm, verbose=False)

                bf_result = solve_epoch(args.dv)
                bf_result, comps, popt, fit_notes = fit_bf_refined(
                    bf_result, recompute=solve_epoch, dv_user=args.dv,
                    verbose=False,
                    components=ncomp,
                    min_sep=args.min_sep,
                    guesses=getattr(args, "guess", None),
                    guess_amps=getattr(args, "guess_amp", None),
                    guess_sigmas=getattr(args, "guess_sigma", None),
                    profile=getattr(args, "profile", "gauss"),
                    inst_fwhm=inst_fwhm,
                    epsilon=args.epsilon,
                    fit_trim=BF_FIT_TRIM)
            else:
                result = run_ccf([(wl, fx)], tpl_wl, tpl_flux,
                                 args.rv_min, args.rv_max, args.rv_step,
                                 mode=getattr(args, "ccf_mode", "mask"),
                                 mask_depth=getattr(args, "mask_depth",
                                                    0.05),
                                 keep_balmer=getattr(args, "keep_balmer",
                                                     False))
                comps = [dict(rv=result["rv"], rv_err=result["rv_err"])]

            vbary, bjd = 0.0, np.nan
            have_coords = ra_i is not None and dec_i is not None
            if obstime is not None:
                _, is_jd = parse_time_input(obstime)
                site_i, site_note = observing_site(args, path=path)
                # Which observatory was used is the one input that is
                # worth 0.46 km/s and shows up nowhere in the numbers, so
                # batch says it too - once per distinct answer, so a
                # 30-epoch run does not repeat the same line 30 times.
                if site_note and site_note not in site_notes_seen:
                    site_notes_seen.add(site_note)
                    print(f"  {site_note}")
                if is_jd:
                    bjd = float(str(obstime).strip())
                elif have_coords:
                    bjd = compute_bjd(obstime, ra_i, dec_i, site_i,
                                      exptime=hdr.get("exptime"))
                if have_coords:
                    vbary = barycentric_correction(ra_i, dec_i, obstime,
                                                   site_i)
                    # Same cross-check the single-spectrum commands run:
                    # our v_bary and BJD against whatever the instrument
                    # pipeline computed for this file.
                    cross = crosscheck_pipeline_bary(
                        path, vbary, (bjd if np.isfinite(bjd) else None),
                        berv_hdr=(checks.get(path) or {}).get("berv"))
                    for line in cross["messages"]:
                        print(f"  {line.strip()}")
            phase = (orbital_phase(bjd, args.t0, args.period)
                     if np.isfinite(bjd) and args.t0 is not None
                     and args.period else None)
            if ncomp >= 2 and not getattr(args, "guess", None):
                if phase is not None:
                    comps, _ = identify_components(
                        comps, phase, gamma=getattr(args, "gamma", None))
                else:
                    comps.sort(key=lambda c: abs(c["amp"] * c["sigma"]),
                               reverse=True)
            if args.method == "bf":
                # After the labelling, so the note names the final C1/C2/...
                blend_note, blend_single = blend_diagnosis(
                    bf_result, comps, inst_fwhm=inst_fwhm,
                    epsilon=args.epsilon, fit_trim=BF_FIT_TRIM,
                    verbose=False)
                if blend_note:
                    fit_notes = fit_notes + [blend_note]

            vb_used = vbary if apply_bary else 0.0
            rows.append(dict(file=os.path.basename(path), bjd=bjd,
                             phase=phase,
                             rv=[apply_barycentric(c["rv"], vb_used)
                                 for c in comps],
                             rv_raw=[c["rv"] for c in comps],
                             rv_err=[c["rv_err"] for c in comps],
                             vbary=vbary, notes=fit_notes,
                             blend_single=blend_single,
                             dv=(bf_result["dv"] if args.method == "bf"
                                 else None)))
            if args.method == "bf":
                profiles.append(dict(
                    name=os.path.basename(path),
                    velocity=bf_result["velocity"],
                    bf=bf_result["bf_smooth"],
                    fit=eval_bf_model(bf_result["velocity"], popt),
                    bjd=bjd, phase=phase))
            msg = ", ".join(f"RV{j + 1} = "
                            f"{apply_barycentric(c['rv'], vb_used):8.3f} "
                            f"± {c['rv_err']:.3f}"
                            for j, c in enumerate(comps))
            ph_txt = f"  phase = {phase:.4f}" if phase is not None else ""
            print(f"  BJD = {bjd:.6f}{ph_txt}  {msg} km/s")
            for n in fit_notes:
                print(f"    ! {n}")
        except Exception as exc:
            print(f"  SKIPPED ({exc.__class__.__name__}: {exc})")
    if not rows:
        sys.exit("No spectrum could be processed.")

    # Epoch-to-epoch variability flag (Katz et al. 2025, A&A 704, A294:
    # stars varying by more than 3 km/s between observations are flagged
    # as binary/variable candidates).
    variability_notes = []
    for j in range(ncomp):
        vals = np.array([r["rv"][j] for r in rows
                         if j < len(r["rv"]) and np.isfinite(r["rv"][j])])
        errs = np.array([r["rv_err"][j] for r in rows
                         if j < len(r["rv_err"])
                         and np.isfinite(r["rv_err"][j])])
        if vals.size < 2:
            continue
        ptp = float(vals.max() - vals.min())
        med_err = float(np.median(errs)) if errs.size else 0.0
        if ptp > max(3.0, 3.0 * med_err):
            variability_notes.append(
                f"C{j + 1}: peak-to-peak RV variation = {ptp:.2f} km/s "
                f"over {vals.size} epochs (median error {med_err:.2f} "
                "km/s) - above the 3 km/s level Katz et al. (2025) flag "
                "as binarity/variability.")

    outfile = args.output or "result_RV_curve.txt"
    with open(outfile, "w") as f:
        f.write(f"# RV curve, method = {args.method.upper()}, "
                f"template = {args.template or f'PHOENIX T={args.teff}K'}\n")
        f.write("# barycentric correction applied: "
                + ("yes" if apply_bary else "no") + "\n")
        if checks:
            nc = sum(c["verdict"] == "corrected" for c in checks.values())
            nn = sum(c["verdict"] == "not-corrected"
                     for c in checks.values())
            f.write(f"# barycentric frame (header check): {nc} already "
                    f"corrected, {nn} topocentric, "
                    f"{len(checks) - nc - nn} unknown of {len(checks)} "
                    "files\n")
        for note in variability_notes:
            f.write("# " + note + "\n")
        flagged = [r for r in rows if r.get("notes")]
        if flagged:
            f.write("# --- fit-quality notes (see the per-epoch BF "
                    "figures) ---\n")
            for r in flagged:
                for note in r["notes"]:
                    f.write(f"# {r['file']}: {note}\n")
                s = r.get("blend_single")
                if s is not None:
                    f.write(f"# {r['file']}: blend_single  RV = "
                            f"{s['rv']:.5f}  RV_err = {s['rv_err']:.5f}  "
                            f"sigma = {s['sigma']:.5f} km/s\n")
        if apply_bary:
            cols = "  ".join(f"RV{j + 1}_raw[km/s]  RV{j + 1}_raw_err  "
                             f"RV{j + 1}_bary[km/s]  RV{j + 1}_bary_err"
                             for j in range(ncomp))
            f.write(f"# file  BJD  phase  {cols}  bary_corr[km/s]\n")
        else:
            f.write("# v_bary column: computed for information only; "
                    "NOT applied to the RVs\n")
            cols = "  ".join(f"RV{j + 1}[km/s]  RV{j + 1}_err"
                             for j in range(ncomp))
            f.write(f"# file  BJD  phase  {cols}  v_bary[km/s]\n")
        for r in rows:
            if apply_bary:
                vals = "  ".join(
                    f"{r['rv_raw'][j]:.5f}  {r['rv_err'][j]:.5f}  "
                    f"{r['rv'][j]:.5f}  "
                    f"{r['rv_err'][j]:.5f}"
                    for j in range(len(r["rv"])))
            else:
                vals = "  ".join(
                    f"{r['rv_raw'][j]:.5f}  {r['rv_err'][j]:.5f}"
                    for j in range(len(r["rv"])))
            ph = f"{r['phase']:.5f}" if r["phase"] is not None else "nan"
            f.write(f"{r['file']}  {r['bjd']:.6f}  {ph}  {vals}  "
                    f"{r['vbary']:.5f}\n")
    print(f"\nRV curve written: {outfile}")
    for note in variability_notes:
        print(note)

    plotfile = args.plot or "result_RV_curve.png"
    fig = make_rv_curve_figure(rows, ncomp, t0=args.t0, period=args.period)
    save_figure(fig, plotfile)

    if profiles:
        stackfile = "result_BF_profiles.png"
        stack_fig = make_bf_stack_figure(profiles)
        save_figure(stack_fig, stackfile)

    return dict(method="batch", fig=fig, output=outfile, plot=plotfile,
                text=f"{len(rows)} spectra -> {outfile}")


def cmd_header(args):
    """Standalone feature: inspect the FITS header of a spectrum.

    Prints a parsed summary (OBJECT, DATE-OBS, RA/Dec, EXPTIME and the
    mid-exposure BJD_TDB when coordinates and time are present) followed
    by the primary-header cards. Use --output to save everything to a
    text file.
    """
    from astropy.io import fits

    info = header_info(args.spectrum)
    lines = ["=============== SPECTRUM HEADER =============",
             f"File     : {args.spectrum}",
             f"OBJECT   : {info['object']}"]
    for key, val, comment in info.get("rv_cards", []):
        note = f"   ({comment})" if comment else ""
        lines.append(f"RV info  : {key} = {val:g}{note}")
    lines += [f"DATE-OBS : {info['obstime']}",
             f"RA [deg] : "
             + (f"{info['ra']:.5f}" if info['ra'] is not None else "-"),
             f"Dec [deg]: "
             + (f"{info['dec']:.5f}" if info['dec'] is not None else "-"),
             f"EXPTIME  : {info['exptime']}"
             + (f"   (DATE-OBS is already the mid-time; EXPTIME of one "
                f"sub-exposure = {info['neonarval']['exptime']:g} s)"
                if (info.get("neonarval") or {}).get("exptime") else "")]
    if info.get("epoch_note"):
        lines.append(f"Epoch    : {info['epoch_note']}")
    site_src = header_site(args.spectrum)
    if site_src:
        lon, lat, alt, cards = site_src
        lines.append(f"Site     : lon {lon:+.4f} E, lat {lat:+.4f}, "
                     f"{alt:.0f} m   (from {cards})")
    if info["ra"] is not None and info["obstime"]:
        try:
            site, _ = observing_site(args)
            bjd = compute_bjd(info["obstime"], info["ra"], info["dec"],
                              site, exptime=info["exptime"])
            lines.append(f"BJD_TDB  : {bjd:.6f}  (mid-exposure)")
            vb = barycentric_correction(info["ra"], info["dec"],
                                        info["obstime"], site)
            lines.append(f"v_bary   : {vb:+.4f} km/s  (computed here)")
            chk0 = barycentric_frame_check(args.spectrum)
            for msg in crosscheck_pipeline_bary(
                    args.spectrum, vb, bjd,
                    berv_hdr=chk0.get("berv"))["messages"]:
                lines.append(msg.strip())
        except Exception as exc:
            lines.append(f"BJD_TDB  : could not be computed ({exc})")
    try:
        chk = barycentric_frame_check(args.spectrum)
        bary_txt = chk["verdict"]
        if chk["frame"]:
            bary_txt += f" ({chk['frame']})"
        if chk["berv"] is not None:
            bary_txt += f", v_bary = {chk['berv']:.4f} km/s"
        lines.append(f"Bary frame: {bary_txt}")
    except Exception:
        pass
    lines.append("=============================================")

    if str(args.spectrum).lower().endswith((".fits", ".fit", ".fits.gz",
                                            ".fit.gz")):
        try:
            with fits.open(args.spectrum) as hdul:
                lines.append(f"HDUs: {[type(h).__name__ for h in hdul]}")
                lines.append("--- primary header cards ---")
                lines.extend(repr(hdul[0].header).splitlines())
        except Exception as exc:
            lines.append(f"(primary header could not be read: {exc})")
    else:
        # ASCII spectrum: show the '# KEY = VALUE' comment header
        # instead of attempting a FITS parse.
        header_lines = []
        try:
            with open(args.spectrum, "r", errors="ignore") as fh:
                for raw in fh:
                    s = raw.strip()
                    if not s:
                        continue
                    if s[0] not in "#;!%":
                        break        # header ends at the first data row
                    header_lines.append(s)
        except OSError as exc:
            header_lines.append(f"(file could not be read: {exc})")
        lines.append("--- ASCII comment header ---")
        if header_lines:
            lines.extend(header_lines)
        else:
            lines.append("(no comment header in this file - it was "
                         "probably normalized by an older version; "
                         "retrofit the frame evidence with: barycheck "
                         "--spectra <file> --from-fits <original FITS>)")

    text = "\n".join(lines)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Header written: {args.output}")
    return dict(method="header", text=text, info=info)


# ---------------------------------------------------------------------------
# Barycentric-frame header vocabulary
#
# Different observatories and pipelines name the barycentric correction
# differently; the tables below collect the conventions of the big
# spectrum archives (ESO, NASA/HST, SDSS, Keck, CfA, NOT, LBT, ...).
# Extend the dictionaries when a new instrument shows up - the check
# logic does not need to change.
# ---------------------------------------------------------------------------

# Explicit wavelength reference-frame keywords (FITS WCS standard and
# variants). Their VALUE names the frame the wavelengths are expressed in.
BARY_FRAME_KEYS = ("SPECSYS", "SPECSYS1", "SPECSYS2", "SPECSYS3",
                   "SSYSOBS", "SSYSSRC", "VFRAME", "WAVEFRAM")

# Frame values meaning the wavelengths ARE / ARE NOT already corrected.
BARY_FRAME_CORRECTED = ("BARYCENT", "HELIOCEN", "BARY", "HELIO", "SOURCE")
BARY_FRAME_TOPOCENTRIC = ("TOPOCENT", "TOPO", "GEOCENTR", "OBSERVER", "OBS")

# Cards whose numeric VALUE is the barycentric/heliocentric velocity
# [km/s unless noted]. Hierarchical prefixes (HIERARCH ESO QC / DRS /
# VRAD, CARACAL, TNG, OHP, ...) plus spaces and underscores are stripped
# before matching, so 'HIERARCH ESO QC VRAD BARYCOR' matches 'BARYCOR'
# and 'ESO DRS BERV' matches 'BERV'. Presence alone only proves the
# pipeline COMPUTED the velocity - the frame keywords above (or the
# flags below) prove it was applied.
BARY_VALUE_KEYS = {
    "BERV":     "HARPS / HARPS-N / SOPHIE / CORALIE DRS, ESPRESSO & "
                "NIRPS QC, CARMENES CARACAL, MAROON-X, GRACES",
    "BARYCORR": "FEROS FERN DRS; generic BARYCORR convention",
    "BARYCOR":  "ESO UVES / GIRAFFE / X-shooter 'QC VRAD BARYCOR'",
    "HELICOR":  "ESO 'QC VRAD HELICOR' (heliocentric)",
    "BARYVEL":  "generic barycentric velocity card",
    "BARYRV":   "generic barycentric velocity card",
    "BARYVCOR": "generic barycentric velocity card",
    "BVC":      "HPF, iSHELL-style pipelines",
    "BCV":      "IRAF rvsao bcvcorr, CfA/TRES, CHIRON",
    "HCV":      "IRAF rvsao bcvcorr (heliocentric)",
    "VHELIO":   "IRAF rvcorrect, SDSS APOGEE apVisit",
    "HELIORV":  "SDSS spPlate / LAMOST 'HELIO_RV'",
    "HELIOVEL": "Keck HIRES makee",
    "HELVEL":   "generic heliocentric velocity card",
    "VBARY":    "generic barycentric velocity card",
    "SSBRV":    "NEID (solar-system-barycentre RV cards)",
    "SSBVEL":   "LBT PEPSI",
    "BSSVHEL":  "BeSS spectra 'BSS_VHEL'",
    "VELOSYS":  "FITS WCS standard frame velocity [m/s]",
    "VCORR":    "IRAF / MIDAS generic velocity correction",
}

# Auxiliary cards that look like the above but are NOT the epoch value
# (yearly maxima, lists, minima): never read these as v_bary.
BARY_IGNORE_MARKS = ("MAX", "MIN", "LIST")

# String-valued processing switches proving the correction WAS applied
# (or explicitly skipped). Maps normalized keyword -> tuple of values
# meaning 'applied'; None means mere presence of the card counts.
BARY_FLAG_KEYS = {
    "HELCORR":  ("COMPLETE", "DONE", "PERFORMED", "YES", "T"),
    "DOPCOR":   None,   # IRAF dopcor: card written when the shift is applied
    "BARYDONE": None,
    "HELDONE":  None,
}
BARY_FLAG_SKIPPED = ("OMIT", "SKIPPED", "NO", "F", "NONE")


def _bary_norm_key(keyword):
    """Normalize a header keyword for vocabulary matching: upper-case,
    HIERARCH prefix, spaces, underscores and dashes removed."""
    name = str(keyword).upper().replace("HIERARCH ", "")
    return name.replace(" ", "").replace("_", "").replace("-", "")


def _bary_match_value_key(norm):
    for key, source in BARY_VALUE_KEYS.items():
        if norm == key or norm.endswith(key) or norm.startswith(key):
            return key, source
    return None, None


def _bary_verdict(frame, flag_applied, flag_skipped, phase3,
                  topocentric_rule=False):
    """`topocentric_rule`: a known pipeline that leaves the wavelengths
    in the observer frame. It ranks below the explicit frame keywords
    and the processing switches (a file that says what it is wins over
    what its pipeline usually does) but above everything else."""
    if frame is not None and any(frame.startswith(v)
                                 for v in BARY_FRAME_CORRECTED):
        return "corrected"
    if frame is not None and any(frame.startswith(v)
                                 for v in BARY_FRAME_TOPOCENTRIC):
        return "not-corrected"
    if flag_applied:
        return "corrected"
    if flag_skipped:
        return "not-corrected"
    if topocentric_rule:
        return "not-corrected"
    if phase3:
        return "corrected"
    return "unknown"


def barycentric_frame_check(path):
    """Decide from a spectrum file's header whether its wavelength
    scale is already in the barycentric rest frame.

    FITS files: every HDU header is scanned against the vocabulary
    tables above, in decreasing order of authority:
      * BARY_FRAME_KEYS - explicit reference-frame keywords
        (SPECSYS = BARYCENT / TOPOCENT, ...): decisive;
      * BARY_FLAG_KEYS - processing switches (HST HELCORR = COMPLETE,
        IRAF DOPCOR, ...): the correction was applied / skipped;
      * PRODCATG = SCIENCE.SPECTRUM - ESO phase-3 1D science products
        (ADP files) are barycentric by the ESO Science Data Product
        standard: supporting evidence;
      * BARY_VALUE_KEYS - pipeline velocity cards (BERV, BARYCORR,
        BCV, VHELIO, SSBRV, ...): value reported for information only;
      * HISTORY/COMMENT lines mentioning a barycentric or heliocentric
        correction.

    ASCII files (e.g. the output of the normalize step, which copies
    the frame evidence of its source FITS into the '# KEY = VALUE'
    comment header): the comment header is parsed with the same
    vocabulary.

    Returns a dict: verdict ('corrected' / 'not-corrected' /
    'unknown'), frame (the explicit frame keyword value or None),
    berv (the pipeline velocity [km/s] or None), instrument, and an
    evidence list of human-readable strings.
    """
    if str(path).lower().endswith((".fits", ".fit", ".fits.gz",
                                   ".fit.gz")):
        return _barycentric_check_fits(path)
    return _barycentric_check_ascii(path)


def _barycentric_check_ascii(path):
    """Barycentric-frame check on the '# KEY = VALUE' comment header of
    an ASCII spectrum, using the same vocabulary as the FITS check. The
    BARYFRAME = corrected / not-corrected summary card (written by the
    normalize step when its source FITS had no explicit frame keyword)
    is honoured too."""
    frame = None
    berv = None
    instrument = None
    flag_applied = False
    flag_skipped = False
    evidence = []
    try:
        with open(path, "r", errors="ignore") as fh:
            for raw in fh:
                s = raw.strip()
                if not s:
                    continue
                if s[0] not in "#;!%":
                    break            # header ends at the first data row
                body = s.lstrip("#;!% \t")
                if "=" not in body:
                    continue
                key, _, val = body.partition("=")
                key = key.strip()
                val = val.strip()
                norm = _bary_norm_key(key)
                if norm in ("INSTRUME", "INSTRUMENT"):
                    if instrument is None:
                        instrument = val
                    continue
                if norm in BARY_FRAME_KEYS and frame is None and val:
                    frame = val.upper()
                    evidence.append(f"{key} = {val} "
                                    "(explicit frame keyword)")
                    continue
                if norm == "BARYFRAME":
                    v = val.upper()
                    if v.startswith("CORRECTED"):
                        flag_applied = True
                    elif v.startswith("NOT"):
                        flag_skipped = True
                    evidence.append(f"{key} = {val} (verdict carried "
                                    "over from the source file by "
                                    "normalize)")
                    continue
                for fkey, ok_values in BARY_FLAG_KEYS.items():
                    if norm == fkey or norm.endswith(fkey):
                        v = val.upper()
                        if ok_values is None or v in ok_values:
                            flag_applied = True
                            evidence.append(f"{key} = {val} "
                                            "(processing switch: "
                                            "correction applied)")
                        elif v in BARY_FLAG_SKIPPED:
                            flag_skipped = True
                            evidence.append(f"{key} = {val} "
                                            "(processing switch: "
                                            "correction skipped)")
                if any(mark in norm for mark in BARY_IGNORE_MARKS) \
                        or norm.endswith("MX") or norm.endswith("MN"):
                    continue
                vkey, source = _bary_match_value_key(norm)
                if vkey is not None:
                    try:
                        value = float(val.split()[0])
                    except (ValueError, IndexError):
                        continue
                    if vkey == "VELOSYS":
                        value /= 1000.0  # FITS standard stores m/s
                    if berv is None:
                        berv = value
                    evidence.append(f"{key} = {value:g} km/s "
                                    f"(pipeline barycentric velocity; "
                                    f"convention: {source})")
    except OSError as exc:
        evidence.append(f"file could not be read ({exc})")
    verdict = _bary_verdict(frame, flag_applied, flag_skipped, False)
    return dict(verdict=verdict, frame=frame, berv=berv,
                instrument=instrument, evidence=evidence)


def _barycentric_check_fits(path):
    """FITS half of barycentric_frame_check (see there)."""
    import re
    from astropy.io import fits

    frame = None
    berv = None
    instrument = None
    phase3 = False
    pipeline_rule = False
    pipeline_topocentric = False
    flag_applied = False
    flag_skipped = False
    evidence = []

    normalize = _bary_norm_key
    match_value_key = _bary_match_value_key

    with fits.open(path) as hdul:
        for ihdu, hdu in enumerate(hdul):
            hdr = hdu.header
            if instrument is None:
                # Neo-Narval writes INSTRUM, not the usual INSTRUME; a
                # SOPHIE file writes neither and is known by its 'OHP ...'
                # prefix alone (header_instrument)
                instrument = (hdr.get("INSTRUME") or hdr.get("INSTRUM")
                              or ("SOPHIE"
                                  if (hdr.get("OHP DRS VERSION")
                                      or hdr.get("OHP TARG NAME"))
                                  else None))
            # Libre-ESpRIT (NARVAL / Neo-Narval / ESPaDOnS) records BERV
            # and BTLA but leaves the wavelength scale in the observer
            # frame. Verified on the data rather than assumed: telluric
            # lines (O2 A band) of two epochs 18.5 km/s apart in BERV
            # coincide to 0.2 km/s, where a corrected scale would put
            # them 18.5 km/s apart.
            if "NARVAL" in str(hdr.get("INSTRUM")
                               or hdr.get("INSTRUME") or "").upper():
                pipeline_topocentric = True
                evidence.append(
                    f"HDU{ihdu} INSTRUM = "
                    f"{hdr.get('INSTRUM') or hdr.get('INSTRUME')}: the "
                    "Libre-ESpRIT pipeline computes BERV/BTLA but does "
                    "NOT shift the wavelengths - the scale is "
                    "topocentric and the correction must be applied "
                    "(checked on telluric lines, see the manual)")
            # HARPS-family DRS (HARPS / HARPS-N / SOPHIE / CORALIE): the
            # 's1d' product is re-binned onto its linear wavelength grid
            # IN THE BARYCENTRIC REST FRAME, while the 'e2ds' of the same
            # exposure is left topocentric. The DRS BERV card alone
            # therefore does not say which of the two you are holding -
            # the layout does, so both are required here.
            # Verified on the data, as for Libre-ESpRIT above: the O2
            # B-band tellurics of a SOPHIE s1d sit 17.0 km/s from the
            # topocentric frame (measured against a Neo-Narval spectrum
            # of the same band) with a header BERV of -17.24 km/s, and
            # between two SOPHIE epochs they move by exactly the BERV
            # difference (-0.48 measured, -0.47 in the headers).
            if (int(hdr.get("NAXIS", 0) or 0) == 1
                    and ("CDELT1" in hdr or "CD1_1" in hdr)
                    and any(_bary_norm_key(c.keyword).endswith("DRSBERV")
                            for c in hdr.cards)):
                pipeline_rule = True
                evidence.append(
                    f"HDU{ihdu}: a DRS BERV card together with a 1-D "
                    "linear wavelength grid identifies a HARPS-family "
                    "'s1d' product (HARPS / HARPS-N / SOPHIE / CORALIE), "
                    "which the DRS re-bins in the BARYCENTRIC rest frame "
                    "- unlike the 'e2ds' of the same exposure (checked "
                    "on the telluric O2 B band)")
            for key in BARY_FRAME_KEYS:
                val = hdr.get(key)
                if val and frame is None:
                    frame = str(val).strip().upper()
                    evidence.append(f"HDU{ihdu} {key} = {val} "
                                    "(explicit frame keyword)")
            for card in hdr.cards:
                name = str(card.keyword).upper()
                if name in ("HISTORY", "COMMENT"):
                    text = str(card.value).upper()
                    if "BARY" in text or "HELIO" in text:
                        evidence.append(f"HDU{ihdu} {name}: {card.value}")
                    continue
                norm = normalize(name)
                for fkey, ok_values in BARY_FLAG_KEYS.items():
                    if norm == fkey or norm.endswith(fkey):
                        val = str(card.value).strip().upper()
                        if ok_values is None or val in ok_values:
                            flag_applied = True
                            evidence.append(
                                f"HDU{ihdu} {name} = {card.value} "
                                "(processing switch: correction applied)")
                        elif val in BARY_FLAG_SKIPPED:
                            flag_skipped = True
                            evidence.append(
                                f"HDU{ihdu} {name} = {card.value} "
                                "(processing switch: correction skipped)")
                if any(mark in norm for mark in BARY_IGNORE_MARKS) \
                        or norm.endswith("MX") or norm.endswith("MN"):
                    continue
                vkey, source = match_value_key(norm)
                if vkey is not None:
                    try:
                        value = float(card.value)
                    except (TypeError, ValueError):
                        continue
                    if vkey == "VELOSYS":
                        value /= 1000.0  # FITS standard stores m/s
                    if berv is None:
                        berv = value
                    evidence.append(f"HDU{ihdu} {name} = {value:g} km/s "
                                    f"(pipeline barycentric velocity; "
                                    f"convention: {source})")
            # ESO-MIDAS descriptor blocks (old FEROS / ESO 1.52m
            # products): HISTORY lines of the form
            #   'BARY_CORR'  ,'R*4 ',  1,  1,'5E14.7',' ',' '
            # followed by the value on the next HISTORY line(s).
            midas_bary = False
            if "HISTORY" in hdr:
                hist = [str(x) for x in hdr["HISTORY"]]
                for i, line in enumerate(hist):
                    m = re.match(r"\s*'([A-Za-z0-9_ ]+)'\s*,\s*"
                                 r"'[RID]\*\d+\s*'", line)
                    if not m:
                        continue
                    dname = m.group(1).strip()
                    vkey, source = match_value_key(normalize(dname))
                    if vkey is None or any(mark in normalize(dname)
                                           for mark in BARY_IGNORE_MARKS):
                        continue
                    for j in range(i + 1, min(i + 6, len(hist))):
                        nums = re.findall(
                            r"[-+]?\d+\.?\d*(?:[EeDd][-+]?\d+)?", hist[j])
                        if nums:
                            value = float(nums[0].replace("D", "E")
                                          .replace("d", "e"))
                            if berv is None:
                                berv = value
                            midas_bary = True
                            evidence.append(
                                f"HDU{ihdu} MIDAS descriptor {dname} = "
                                f"{value:g} km/s (pipeline barycentric "
                                f"velocity; convention: {source})")
                            break
            if (midas_bary and str(hdr.get("ORIGIN", "")).upper()
                    .startswith("ESO-MIDAS")
                    and str(hdr.get("INSTRUME", "")).upper() == "FEROS"):
                pipeline_rule = True
                evidence.append(
                    f"HDU{ihdu} ORIGIN = ESO-MIDAS + INSTRUME = FEROS + "
                    "BARY_CORR descriptor: the FEROS-DRS (MIDAS) "
                    "pipeline delivers barycentric-corrected wavelengths "
                    "by default (note: its correction is only accurate "
                    "to ~60 m/s - Mueller et al. 2013, A&A 556, A3)")
            if str(hdr.get("PRODCATG", "")).upper().startswith(
                    "SCIENCE.SPECTRUM"):
                phase3 = True
                evidence.append(f"HDU{ihdu} PRODCATG = "
                                f"{hdr['PRODCATG']} (ESO phase-3 1D "
                                "product: barycentric by the SDP "
                                "standard)")

    verdict = _bary_verdict(frame, flag_applied, flag_skipped,
                            phase3 or pipeline_rule,
                            topocentric_rule=pipeline_topocentric)
    return dict(verdict=verdict, frame=frame, berv=berv,
                instrument=instrument, evidence=evidence)


def _bary_file_stem(path):
    """File-name stem used to match a normalized ASCII spectrum to its
    original FITS: base name, extension(s) and a trailing '_norm'
    stripped, lower-case."""
    stem = os.path.basename(path)
    for ext in (".fits.gz", ".fit.gz", ".fits", ".fit", ".txt", ".dat",
                ".obs", ".asc"):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    if stem.lower().endswith("_norm"):
        stem = stem[:-5]
    return stem.lower()


def _stamp_ascii_header(path, chk):
    """Insert the barycentric-frame evidence cards of `chk` into the
    comment header of an ASCII spectrum, in place. Used by
    barycheck --from-fits to retrofit files that were normalized before
    the evidence was propagated. Returns True when cards were written."""
    cards = []
    if chk["instrument"]:
        cards.append(("INSTRUME", f"# INSTRUME = {chk['instrument']}"))
    if chk["frame"] is not None:
        cards.append(("SPECSYS", f"# SPECSYS = {chk['frame']}"))
    elif chk["verdict"] != "unknown":
        cards.append(("BARYFRAME", f"# BARYFRAME = {chk['verdict']}"))
    if chk["berv"] is not None:
        cards.append(("BERV", f"# BERV = {chk['berv']:.5f}"))
    if not cards:
        return False
    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()
    insert = 0
    existing = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if s and s[0] in "#;!%":
            insert = i + 1
            body = s.lstrip("#;!% \t")
            if "=" in body:
                existing.add(_bary_norm_key(body.partition("=")[0].strip()))
        elif s:
            break
    cards = [(k, text) for k, text in cards
             if _bary_norm_key(k) not in existing]
    if not cards:
        return False
    lines[insert:insert] = [text + "\n" for _, text in cards]
    with open(path, "w") as fh:
        fh.writelines(lines)
    return True


def cmd_barycheck(args):
    """Check whether the wavelengths of spectra are already
    barycentric: reads the header of every input file - FITS HDU
    headers, or the '# KEY = VALUE' comment header of normalized ASCII
    spectra (see barycentric_frame_check) - prints a per-file table
    with the evidence cards, and recommends whether the batch/bf/ccf
    runs should use --no-bary.

    With --from-fits, an ASCII spectrum that carries no frame
    information is matched to its original FITS file by name (the
    '_norm' suffix is ignored) and the original's evidence is written
    into the ASCII comment header in place, so later runs see it too.
    """
    import glob
    files = sorted(set(sum((glob.glob(p) for p in args.spectra), [])))
    if not files:
        sys.exit("No files match the given pattern(s).")

    sources = {}
    if getattr(args, "from_fits", None):
        for p in sorted(set(sum((glob.glob(g) for g in args.from_fits),
                                []))):
            sources[_bary_file_stem(p)] = p

    lines = [f"{len(files)} file(s) to check.", "",
             f"{'file':<42} {'instrument':<10} {'frame':<9} "
             f"{'v_bary[km/s]':>12}  verdict"]
    results, details = [], []
    for path in files:
        name = os.path.basename(path)
        try:
            chk = barycentric_frame_check(path)
        except Exception as exc:
            chk = dict(verdict="unknown", frame=None, berv=None,
                       instrument=None,
                       evidence=[f"header could not be read "
                                 f"({exc.__class__.__name__}: {exc})"])
        is_fits = path.lower().endswith((".fits", ".fit", ".fits.gz",
                                         ".fit.gz"))
        if chk["verdict"] == "unknown" and not is_fits and sources:
            src_path = sources.get(_bary_file_stem(path))
            if src_path is not None:
                try:
                    src_chk = barycentric_frame_check(src_path)
                except Exception:
                    src_chk = None
                if src_chk and src_chk["verdict"] != "unknown" \
                        and _stamp_ascii_header(path, src_chk):
                    chk = barycentric_frame_check(path)
                    chk["evidence"].append(
                        "header updated in place from "
                        + os.path.basename(src_path))
        results.append(chk)
        berv_txt = (f"{chk['berv']:>12.4f}" if chk["berv"] is not None
                    else f"{'-':>12}")
        lines.append(f"{name:<42} {str(chk['instrument'] or '-'):<10} "
                     f"{str(chk['frame'] or '-'):<9} {berv_txt}  "
                     f"{chk['verdict']}")
        details.append(f"\n--- {name} ---")
        details.extend("  " + e for e in (chk["evidence"]
                       or ["(no relevant header cards found)"]))
    lines.extend(details)

    n_cor = sum(r["verdict"] == "corrected" for r in results)
    n_not = sum(r["verdict"] == "not-corrected" for r in results)
    n_unk = len(results) - n_cor - n_not
    lines += ["", f"Summary: {n_cor} corrected, {n_not} not corrected, "
                  f"{n_unk} unknown."]
    if n_cor == len(results):
        lines.append(
            "All headers say the wavelength scale is ALREADY barycentric: "
            "run batch/bf/ccf with --no-bary so the correction is not "
            "applied twice (v_bary is then written to the result file as "
            "a note only).")
    elif n_not == len(results):
        lines.append(
            "The wavelength scale is topocentric (correction NOT applied): "
            "run batch/bf/ccf normally, without --no-bary.")
    elif n_unk == len(results):
        lines.append(
            "No frame information found: inspect the full headers with "
            "the 'header' command or check the data provenance.")
    else:
        lines.append(
            "MIXED frames: process the corrected and uncorrected files "
            "in separate runs (with and without --no-bary).")

    text = "\n".join(lines)
    print(text)
    if getattr(args, "output", None):
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Report written: {args.output}")
    return dict(method="barycheck", text=text)


def cmd_demo(args):
    """Generate a synthetic SB2 spectrum and solve it with both CCF and BF.

    The true RVs are known, so the behavioural difference of the two
    methods is measured directly: the BF shows sharp separated peaks while
    the CCF peaks are broader and prone to blending.
    """
    rng = np.random.default_rng(42)
    rv1_true, rv2_true = -80.0, 120.0
    light_ratio = 0.45

    wl = log_wave_grid(4950.0, 5550.0, 1.0)
    tpl = np.ones_like(wl)
    n_lines = 160
    centers = rng.uniform(4960, 5540, n_lines)
    depths = rng.uniform(0.15, 0.85, n_lines)
    widths = rng.uniform(0.08, 0.25, n_lines)
    for c0, d, w in zip(centers, depths, widths):
        tpl -= d * np.exp(-((wl - c0) ** 2) / (2 * w ** 2))
    tpl = np.clip(tpl, 0.02, None)

    def rot_broaden(flux, vsini_kms):
        sigma_pix = vsini_kms / 1.0 / 2.35482
        return gaussian_filter1d(flux, sigma_pix)

    f1 = np.interp(wl, doppler_shift(wl, rv1_true), tpl)
    f2 = np.interp(wl, doppler_shift(wl, rv2_true), tpl)
    f1 = rot_broaden(f1, 40.0)
    f2 = rot_broaden(f2, 25.0)
    obs = (f1 + light_ratio * f2) / (1.0 + light_ratio)
    obs += rng.normal(0, 0.004, obs.size)

    print("Synthetic SB2 system generated:")
    print(f"  True RV1 = {rv1_true} km/s, RV2 = {rv2_true} km/s, "
          f"light ratio = {light_ratio}\n")

    print("--- BF method ---")
    bf_result = compute_bf(wl, obs, wl, tpl, vel_range=300.0, dv=2.0,
                           svd_rcond=5e-4, smooth_kms=10.0)
    comps, popt = fit_bf_peaks(bf_result["velocity"], bf_result["bf_smooth"],
                               components=2, min_sep=50.0,
                               fit_trim=BF_FIT_TRIM)
    for i, c in enumerate(comps, 1):
        print(f"  Component {i}: RV = {c['rv']:8.3f} ± {c['rv_err']:.3f} km/s")

    print("--- CCF method (same data) ---")
    rv_grid = np.arange(-200, 251, 2.0)
    ccf = calculate_ccf(wl, 1.0 - obs, wl, 1.0 - tpl, rv_grid)
    ccf /= ccf.max()
    ccomp, cpopt = fit_bf_peaks(rv_grid, ccf, components=2, min_sep=50.0,
                                with_offset=True)
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
        axes[1].plot(v, eval_bf_model(v, popt), "r--", lw=1.5,
                     label="Double Gaussian fit")
        for rvt in (rv1_true, rv2_true):
            axes[1].axvline(rvt, color="0.5", ls=":", lw=1)
        axes[1].set_ylabel("BF")
        axes[1].set_title("Broadening Function: sharp, separated components")
        axes[1].legend()

        axes[2].plot(rv_grid, ccf, "k-", lw=1.5, label="CCF")
        axes[2].plot(rv_grid, eval_bf_model(rv_grid, cpopt), "r--", lw=1.5,
                     label="Double Gaussian fit")
        for rvt in (rv1_true, rv2_true):
            axes[2].axvline(rvt, color="0.5", ls=":", lw=1)
        axes[2].set_xlabel("Radial velocity [km/s]")
        axes[2].set_ylabel("Normalized CCF")
        axes[2].set_title("CCF: broader peaks, prone to blending")
        axes[2].legend()

        fig.tight_layout()
        save_figure(fig, args.plot)


def make_args(**overrides):
    """Namespace with every field expected by cmd_ccf/cmd_bf."""
    base = dict(spectrum=None, format="auto", template=None,
                teff=None, logg=4.5, feh=0.0,
                wave_min=None, wave_max=None,
                object=None, ra=None, dec=None, obstime=None, site=None,
                no_bary=False,
                t0=None, period=None, varastro=False, t0_to_bjd=False,
                plot=None, output=None,
                rv_min=None, rv_max=None, rv_step=0.5,
                ccf_mode="mask", mask_depth=0.05, keep_balmer=False,
                vel_range=500.0, dv=None, svd_rcond=None, smooth=None,
                components=1, min_sep=30.0, guess=None, guess_amp=None,
                guess_sigma=None, profile="gauss",
                gamma=None, degrade_template=False,
                spectrograph=None, resolution=None, vsini=None, epsilon=0.6,
                poly_order=3, iterations=20, low_clip=1.0, high_clip=4.0,
                window=20.0, fit_method="poly", no_save=False,
                # todcor: one template per component
                template1=None, template2=None, teff1=None, teff2=None,
                vsini1=None, vsini2=None, max_rv=200.0, alpha=None,
                no_clean=False, simulate=False, sim_sep=None,
                # deliberately not `min_sep` - that is BF's, and defaults
                # to 30 km/s, which would silently gate the TODCOR peak
                todcor_min_sep=0.0)
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
            sim = query_simbad(name)
            extra = ""
            if sim["sptype"]:
                extra += f", SpT = {sim['sptype']}"
            if sim["vmag"] is not None:
                extra += f", V = {sim['vmag']:.2f}"
            print(f"  SIMBAD: RA = {sim['ra']:.5f} deg, "
                  f"Dec = {sim['dec']:.5f} deg{extra}")
            kw["object"], kw["ra"], kw["dec"] = name, sim["ra"], sim["dec"]
        except ValueError as exc:
            print(f"  {exc}")
            name = None
    if not name:
        manual = ask("Enter coordinates manually? (y/n)", default="n",
                     cast=str, validate=lambda s: s.lower() in ("y", "n"))
        if manual.lower() == "y":
            kw["ra"] = ask("  RA (deg or sexagesimal hh:mm:ss)")
            kw["dec"] = ask("  Dec (deg or sexagesimal dd:mm:ss)")
    if kw.get("ra") is not None:
        kw["obstime"] = ask("Observation time (ISOT, BJD number or "
                            "DD/MM/YY; empty: read from FITS header)",
                            allow_empty=True)
        kw["site"] = ask("Observatory (built-in name paranal/lasilla/ctio/"
                         "ohp/tug/kreiken/lco/... or 'lonE,lat,alt'; "
                         "empty: read it from the file header)",
                         allow_empty=True)
        already_bary = False
        try:
            if kw.get("spectrum"):
                already_bary = (barycentric_frame_check(kw["spectrum"])
                                ["verdict"] == "corrected")
        except Exception:
            already_bary = False
        if already_bary:
            print("  Header check: the spectrum's wavelength scale is "
                  "ALREADY barycentric - the correction will NOT be "
                  "applied (v_bary is computed and kept in the result "
                  "file for information only).")
            kw["no_bary"] = True
        else:
            apply = ask("Apply the barycentric correction to the "
                        "reported RVs? (n: v_bary is computed and kept "
                        "in the result file for information only)",
                        default="y", cast=str,
                        validate=lambda s: s.lower() in ("y", "n"))
            kw["no_bary"] = apply.lower() == "n"
    if kw.get("object"):
        fetch = ask("Fetch ephemeris T0/P from VarAstro (var.astro.cz)? "
                    "(y/n)", default="y", cast=str,
                    validate=lambda s: s.lower() in ("y", "n"))
        if fetch.lower() == "y":
            try:
                va = query_varastro(kw["object"])
                print(f"  VarAstro: {va['star']} -> T0 = {va['t0']:.5f} HJD, "
                      f"P = {va['period']:.7f} d  ({va['url']})")
                t0 = ask("Ephemeris T0 [HJD]", default=va["t0"], cast=float)
                if kw.get("ra") is not None and kw.get("dec") is not None:
                    conv = ask("Convert T0 from HJD to BJD and use it for "
                               "the analysis? (y/n)", default="y", cast=str,
                               validate=lambda s: s.lower() in ("y", "n"))
                    if conv.lower() == "y":
                        t0 = hjd_to_bjd(t0, kw["ra"], kw["dec"])
                        print(f"  T0 converted to BJD: {t0:.6f}")
                else:
                    print("  HJD -> BJD conversion needs target coordinates; "
                          "T0 is kept in HJD.")
                kw["t0"] = t0
                kw["period"] = ask("Orbital period [days]",
                                   default=va["period"], cast=float)
                return kw
            except ValueError as exc:
                print(f"  {exc}")
    kw["t0"] = ask("Ephemeris T0 [BJD] for the orbital phase (empty: skip)",
                   allow_empty=True, cast=float)
    if kw.get("t0") is not None:
        kw["period"] = ask("Orbital period [days]", cast=float)
    return kw


def _ask_common_inputs(kw):
    """Inputs shared by both methods: spectrograph, files and wavelength
    range — the spectrograph is asked first, before any file is read.
    Mutates and returns kw (which may already hold the target info)."""
    print("\nBuilt-in spectrographs:")
    for i, (n, r, note) in enumerate(SPECTROGRAPHS, 1):
        print(f"  [{i:>2}] {n:<14} R = {r:<7} ({note})")
    sel = ask("Spectrograph number (empty: enter R manually / skip)",
              allow_empty=True, cast=int,
              validate=lambda i: 1 <= i <= len(SPECTROGRAPHS))
    if sel is not None:
        name, r, _ = SPECTROGRAPHS[sel - 1]
        kw["resolution"] = float(r)
        print(f"  {name}: R = {r}  (instrumental broadening "
              f"c/R = {C_KMS / r:.2f} km/s)")
    else:
        kw["resolution"] = ask("Resolution R, e.g. 48000 (empty: skip)",
                               allow_empty=True, cast=float)

    r_val = kw.get("resolution")
    if r_val:
        floor = min_vsini(r_val)
        print(f"  Minimum resolvable vsini at this R: c/R = {floor:.1f} km/s")
        kw["vsini"] = ask(f"Template vsini [km/s], >= {floor:.1f} "
                          "(empty: skip)", allow_empty=True, cast=float,
                          validate=lambda v: v >= floor)
    else:
        kw["vsini"] = ask("Template vsini [km/s] (empty: skip)",
                          allow_empty=True, cast=float)

    have_norm = ask("Do you already have a normalized spectrum? (y/n)",
                    default="y", cast=str,
                    validate=lambda s: s.lower() in ("y", "n"))
    if have_norm.lower() == "y":
        spectrum = ask("Normalized spectrum file (ASCII or FITS)",
                       cast=str, validate=os.path.isfile)
    else:
        raw = ask("Raw spectrum file (FITS or ASCII)",
                  cast=str, validate=os.path.isfile)
        order = ask("  Continuum polynomial order", default=3, cast=int)
        iters = ask("  Clipping iterations", default=20, cast=int)
        fitm = ask("  Continuum model: [1] windowed polynomial "
                   "[2] B-spline", default="1", cast=str,
                   validate=lambda s: s in ("1", "2"))
        payload = cmd_normalize(make_args(spectrum=raw, poly_order=order,
                                          iterations=iters,
                                          fit_method=("bspline"
                                                      if fitm == "2"
                                                      else "poly")))
        spectrum = payload["output"]
        print(f"  Using normalized spectrum: {spectrum}\n")
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
    kw["wave_min"] = ask("Minimum wavelength [nm] (empty: all)",
                         allow_empty=True, cast=float)
    kw["wave_max"] = ask("Maximum wavelength [nm] (empty: all)",
                         allow_empty=True, cast=float)

    # Double-correction guard: the target questions ran before the
    # spectrum was chosen, so re-check now that the file is known.
    if not kw.get("no_bary"):
        try:
            if barycentric_frame_check(spectrum)["verdict"] == "corrected":
                print("\n  Header check: the spectrum's wavelength scale "
                      "is ALREADY barycentric - the correction will NOT "
                      "be applied (v_bary is computed and kept in the "
                      "result file for information only).")
                kw["no_bary"] = True
        except Exception:
            pass
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
    choice = ask("Choice", default="2", cast=str,
                 validate=lambda s: s in ("1", "2"))

    if choice == "1":
        kw["rv_min"] = ask("RV scan lower limit [km/s] (empty: automatic "
                           "two-stage +-900 km/s search)",
                           allow_empty=True, cast=float)
        kw["rv_max"] = ask("RV scan upper limit [km/s] (empty: automatic)",
                           allow_empty=True, cast=float)
        kw["rv_step"] = ask("Fine RV step [km/s]", default=0.5, cast=float)
        print()
        cmd_ccf(make_args(**kw))
    else:
        kw["vel_range"] = ask("BF window half-width [km/s]",
                              default=500.0, cast=float)
        kw["components"] = ask("Number of components (SB1=1, SB2=2, SB3=3)",
                               default=2, cast=int,
                               validate=lambda n: n in (1, 2, 3))
        if kw["components"] >= 2:
            g = ask("Initial RV guesses, comma-separated [km/s] "
                    "(empty: automatic peak search; giving them fixes "
                    "the C1/C2/... labels)", allow_empty=True)
            if g:
                kw["guess"] = [float(t) for t in
                               str(g).replace(",", " ").split()]
                ga = ask("Initial amplitudes, comma-separated (peak "
                         "heights read off the BF plot; empty: "
                         "automatic)", allow_empty=True)
                if ga:
                    kw["guess_amp"] = [float(t) for t in
                                       str(ga).replace(",", " ").split()]
                    gs = ask("Initial widths sigma, comma-separated "
                             "[km/s] (empty: automatic)",
                             allow_empty=True)
                    if gs:
                        kw["guess_sigma"] = [
                            float(t) for t in
                            str(gs).replace(",", " ").split()]
            kw["gamma"] = ask("Systemic velocity gamma [km/s] for the "
                              "phase-based identification (empty: "
                              "estimate)", allow_empty=True, cast=float)
        kw["smooth"] = ask("BF smoothing FWHM [km/s] (empty: auto)",
                           allow_empty=True, cast=float)
        kw["svd_rcond"] = ask("SVD cutoff (empty: automatic; "
                              "0 = no truncation)",
                              allow_empty=True, cast=float)
        print()
        cmd_bf(make_args(**kw))

    print("\nDone. Result files and the fit figure were saved "
          "to the working directory.")


def run_gui():
    """Minimal tkinter widget.

    Sections, top to bottom: target (SIMBAD lookup with manual-coordinate
    fallback, for the barycentric correction), input files (with a
    'Normalize raw...' dialog for un-normalized spectra), wavelength
    range, a 'Method...' button opening a small dialog (CCF, BF or
    TODCOR; for BF also the components and the mode: 'with
    assumptions' shows the smoothed BF with one click so per-component
    RV/Amp/Sigma can be read off and typed in, 'automatic component
    search' disables all assumption inputs; TODCOR reveals a second
    template row in the main window and can run its self-calibration
    without any spectrum), Run, result text and the embedded fit plot.
    The analysis core is shared with the CLI (cmd_ccf/cmd_bf/
    cmd_todcor); the result files (result_*.txt, result_*.png) are
    written to disk as usual.
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

    target_var = tk.StringVar()
    ra_var, dec_var = tk.StringVar(), tk.StringVar()
    time_var = tk.StringVar()
    site_var = tk.StringVar(value=AUTO_SITE_LABEL)
    custom_site = {"coords": None}   # (lonE, lat, alt) entered by the user

    trow = ttk.LabelFrame(main, text="Target (optional, for barycentric "
                                     "correction)", padding=6)
    trow.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))

    ttk.Label(trow, text="Name:").grid(row=0, column=0, sticky="w")
    ttk.Entry(trow, textvariable=target_var, width=18).grid(row=0, column=1,
                                                            padx=2)

    siminfo_var = tk.StringVar(value="")

    def simbad_lookup():
        name = target_var.get().strip()
        if not name:
            messagebox.showinfo("SIMBAD", "Enter a star name first.")
            return
        try:
            sim = query_simbad(name)
        except ValueError as exc:
            messagebox.showwarning(
                "SIMBAD", f"{exc}\n\nEnter the coordinates manually "
                          "in the RA/Dec fields.")
            return
        ra_var.set(f"{sim['ra']:.5f}")
        dec_var.set(f"{sim['dec']:.5f}")
        parts = []
        if sim["sptype"]:
            parts.append(f"SpT: {sim['sptype']}")
        if sim["vmag"] is not None:
            parts.append(f"V: {sim['vmag']:.2f}")
        siminfo_var.set("   ".join(parts) if parts else "SpT/V: not in SIMBAD")

    ttk.Button(trow, text="SIMBAD", command=simbad_lookup).grid(row=0, column=2,
                                                                padx=(2, 12))
    ttk.Label(trow, text="RA:").grid(row=0, column=3)
    ttk.Entry(trow, textvariable=ra_var, width=12).grid(row=0, column=4, padx=2)
    ttk.Label(trow, text="Dec:").grid(row=0, column=5)
    ttk.Entry(trow, textvariable=dec_var, width=12).grid(row=0, column=6, padx=2)
    ttk.Label(trow, textvariable=siminfo_var, foreground="gray"
              ).grid(row=0, column=7, sticky="w", padx=(8, 0))

    ttk.Label(trow, text="Obs time (ISOT, BJD or DD/MM/YY):").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
    ttk.Entry(trow, textvariable=time_var, width=20).grid(row=1, column=2,
                                                          columnspan=2,
                                                          sticky="w",
                                                          pady=(4, 0))
    ttk.Label(trow, text="Observatory:").grid(row=1, column=4, sticky="e",
                                              pady=(4, 0))
    obs_choices = [AUTO_SITE_LABEL] + [o[0] for o in OBSERVATORIES]

    def on_observatory_selected(_e=None):
        custom_site["coords"] = None      # a built-in choice replaces custom

    obs_box = ttk.Combobox(trow, textvariable=site_var, values=obs_choices,
                           width=20, state="readonly")
    obs_box.grid(row=1, column=5, sticky="w", pady=(4, 0))
    obs_box.bind("<<ComboboxSelected>>", on_observatory_selected)

    def custom_site_dialog():
        """Coordinates of an observatory not in the list, in a small
        sub-window: lon [deg East], lat [deg], alt [m]. The values stay
        stored behind the Observatory field."""
        dlg = tk.Toplevel(root)
        dlg.title("Custom observatory")
        dlg.transient(root)
        frm = ttk.Frame(dlg, padding=10)
        frm.grid(row=0, column=0)
        lo_v, la_v, al_v = tk.StringVar(), tk.StringVar(), tk.StringVar()
        cur = custom_site["coords"] or observatory_coords(site_var.get())
        if cur:
            lo_v.set(f"{cur[0]:.4f}")
            la_v.set(f"{cur[1]:.4f}")
            al_v.set(f"{cur[2]:.0f}")
        for r, (lbl, var) in enumerate([("Longitude [deg East]:", lo_v),
                                        ("Latitude [deg]:", la_v),
                                        ("Altitude [m]:", al_v)]):
            ttk.Label(frm, text=lbl).grid(row=r, column=0, sticky="w",
                                          pady=2)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=r, column=1,
                                                            padx=4, pady=2)

        def do_use():
            try:
                lon = float(lo_v.get())
                lat = float(la_v.get())
                alt = float(al_v.get() or 0.0)
            except ValueError:
                messagebox.showerror("Custom observatory",
                                     "Enter numeric longitude/latitude "
                                     "(and altitude).", parent=dlg)
                return
            custom_site["coords"] = (lon, lat, alt)
            site_var.set(f"Custom ({lon:.4f}E, {lat:.4f}, {alt:.0f} m)")
            dlg.destroy()

        ttk.Button(frm, text="Use", command=do_use).grid(row=3, column=0,
                                                         columnspan=2,
                                                         pady=(8, 0))

    ttk.Button(trow, text="Custom...", command=custom_site_dialog
               ).grid(row=1, column=6, sticky="w", padx=(4, 0), pady=(4, 0))
    # Ticked by default, like the CLI: there, applying the correction is
    # what happens unless --no-bary is given, and the same file must not
    # give a barycentric RV from one interface and a topocentric one from
    # the other. update_bary_guard() unticks and locks it for a spectrum
    # whose header says the wavelengths are already corrected.
    bary_var = tk.BooleanVar(value=True)
    bary_chk = ttk.Checkbutton(trow, text="Apply barycentric correction",
                               variable=bary_var)
    bary_chk.grid(row=1, column=7, sticky="w", padx=(8, 0), pady=(4, 0))

    def resolve_site():
        """Observatory string for the analysis: the stored custom
        coordinates when a custom site was entered, the selected built-in
        observatory name otherwise - and None while the selection is
        'Automatic', which lets the analysis read the observatory out of
        the file's own header (and fall back to the default site only if
        the header has none)."""
        if custom_site["coords"] and site_var.get().startswith("Custom"):
            lon, lat, alt = custom_site["coords"]
            return f"{lon},{lat},{alt}"
        name = site_var.get().strip()
        if name == AUTO_SITE_LABEL:
            return None
        return name if observatory_coords(name) else None

    t0_var, per_var = tk.StringVar(), tk.StringVar()
    t0_unit_var = tk.StringVar(value="BJD")
    ttk.Label(trow, text="Ephemeris T0:").grid(row=2, column=0,
                                               columnspan=2, sticky="w",
                                               pady=(4, 0))
    ttk.Entry(trow, textvariable=t0_var, width=14).grid(row=2, column=2,
                                                        sticky="w", pady=(4, 0))
    ttk.Label(trow, textvariable=t0_unit_var, foreground="gray"
              ).grid(row=2, column=3, sticky="w", pady=(4, 0))
    ttk.Label(trow, text="Period [d]:").grid(row=2, column=4, sticky="e",
                                             pady=(4, 0))
    ttk.Entry(trow, textvariable=per_var, width=10).grid(row=2, column=5,
                                                         sticky="w",
                                                         pady=(4, 0))

    def varastro_lookup():
        """Fetch T0/P from var.astro.cz by the target name; the values come
        in VarAstro's native units (T0 in HJD, P in days) and the fields
        stay editable for manual override."""
        name = target_var.get().strip()
        if not name:
            messagebox.showinfo("VarAstro", "Enter a star name first.")
            return
        try:
            va = query_varastro(name)
        except ValueError as exc:
            messagebox.showwarning(
                "VarAstro", f"{exc}\n\nEnter T0 and Period manually.")
            return
        t0_var.set(f"{va['t0']:.5f}")
        per_var.set(f"{va['period']:.7f}")
        t0_unit_var.set("HJD")
        messagebox.showinfo(
            "VarAstro", f"{va['star']}\nT0 = {va['t0']:.5f} HJD\n"
                        f"P = {va['period']:.7f} d\n\n{va['url']}\n"
                        "(fields remain editable; press 'HJD -> BJD' to "
                        "convert T0 and use the BJD value in the analysis)")

    def convert_t0_to_bjd():
        """Convert the T0 field from HJD to BJD_TDB; the converted value
        replaces the field content and is used in the analysis."""
        s = t0_var.get().strip()
        if not s:
            messagebox.showinfo("HJD -> BJD", "Enter or fetch a T0 first.")
            return
        if t0_unit_var.get() == "BJD":
            messagebox.showinfo("HJD -> BJD", "T0 is already in BJD.")
            return
        ra_s, dec_s = ra_var.get().strip(), dec_var.get().strip()
        if not ra_s or not dec_s:
            messagebox.showwarning(
                "HJD -> BJD", "Target coordinates are required: fill the "
                              "RA/Dec fields (or use the SIMBAD button) "
                              "first.")
            return
        try:
            bjd = hjd_to_bjd(float(s), float(ra_s), float(dec_s))
        except Exception as exc:
            messagebox.showerror("HJD -> BJD", str(exc))
            return
        t0_var.set(f"{bjd:.6f}")
        t0_unit_var.set("BJD")
        messagebox.showinfo(
            "HJD -> BJD", f"T0 converted to BJD: {bjd:.6f}\n"
                          "The BJD value will be used in the analysis.")

    ttk.Button(trow, text="VarAstro", command=varastro_lookup
               ).grid(row=2, column=6, sticky="w", padx=(6, 0), pady=(4, 0))
    ttk.Button(trow, text="HJD -> BJD", command=convert_t0_to_bjd
               ).grid(row=2, column=7, sticky="w", padx=(4, 0), pady=(4, 0))

    spec_var = tk.StringVar()
    tpl_var = tk.StringVar()
    spec_label_var = tk.StringVar(value="Normalized spectrum:")

    ttk.Label(main, textvariable=spec_label_var).grid(row=2, column=0,
                                                      sticky="w")
    ttk.Entry(main, textvariable=spec_var, width=48).grid(row=2, column=1,
                                                          sticky="ew", padx=4)
    fbtns = ttk.Frame(main)
    fbtns.grid(row=2, column=2, sticky="w")
    ttk.Button(fbtns, text="Browse...",
               command=lambda: browse(spec_var, "Select normalized spectrum")
               ).pack(side="left")

    bary_forced_off = {"on": False}

    def update_bary_guard(*_):
        """Double-correction guard: when the selected spectrum's header
        says its wavelengths are already barycentric, untick and disable
        'Apply barycentric correction' so it cannot be applied twice."""
        path = spec_var.get().strip()
        verdict = None
        if path and os.path.isfile(path):
            try:
                verdict = barycentric_frame_check(path)["verdict"]
            except Exception:
                verdict = None
        if verdict == "corrected":
            if bary_var.get():
                bary_forced_off["on"] = True
            bary_var.set(False)
            bary_chk.configure(
                text="Apply barycentric correction "
                     "(blocked: header says already corrected)",
                state="disabled")
        else:
            # Picking an already-corrected file and then a topocentric one
            # must not leave the box silently unticked from the first: put
            # back what the guard took away.
            if bary_forced_off["on"]:
                bary_var.set(True)
                bary_forced_off["on"] = False
            bary_chk.configure(text="Apply barycentric correction",
                               state="normal")

    def update_profile_hint(*_):
        """Say so, next to the file, when the chosen file is a correlation
        profile rather than a spectrum — the output of a correlation, not
        an input any method here can use."""
        if correlation_profile_note(spec_var.get().strip()):
            profile_hint_var.set("correlation profile (velocity axis) — "
                                 "not a spectrum: use the spectrum that "
                                 "went into it")
        else:
            profile_hint_var.set("")

    profile_hint_var = tk.StringVar(value="")
    ttk.Label(main, textvariable=profile_hint_var, foreground="#b06000",
              wraplength=200
              ).grid(row=2, column=3, sticky="w", padx=(8, 0))

    spec_var.trace_add("write", update_bary_guard)
    spec_var.trace_add("write", update_profile_hint)

    tpl_label = ttk.Label(main, text="Synthetic spectrum:")
    tpl_label.grid(row=3, column=0, sticky="w")
    tpl_entry = ttk.Entry(main, textvariable=tpl_var, width=48)
    tpl_entry.grid(row=3, column=1, sticky="ew", padx=4)
    tpl_browse = ttk.Button(main, text="Browse...",
                            command=lambda: browse(tpl_var,
                                                   "Select synthetic spectrum"))
    tpl_browse.grid(row=3, column=2, sticky="w")

    # Second template: TODCOR correlates against t1(v1) + alpha*t2(v2), so
    # it needs one template per component. The row is shown only for that
    # method; every other method uses the single template above.
    tpl2_var = tk.StringVar()
    tpl2_label = ttk.Label(main,
                           text="Template 2 (empty = same as template 1):")
    tpl2_entry = ttk.Entry(main, textvariable=tpl2_var, width=48)
    tpl2_browse = ttk.Button(
        main, text="Browse...",
        command=lambda: browse(tpl2_var, "Select the secondary template"))
    tpl2_row = [(tpl2_label, dict(row=6, column=0, sticky="w")),
                (tpl2_entry, dict(row=6, column=1, sticky="ew", padx=4)),
                (tpl2_browse, dict(row=6, column=2, sticky="w"))]
    for _w, _opts in tpl2_row:
        _w.grid(**_opts)

    def show_tpl2(show):
        for w, opts in tpl2_row:
            if show:
                w.grid(**opts)
            else:
                w.grid_remove()

    show_tpl2(False)

    wmin_var, wmax_var = tk.StringVar(), tk.StringVar()
    res_var, vsini_var = tk.StringVar(), tk.StringVar()
    wrow = ttk.Frame(main)
    wrow.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Label(wrow, text="Wavelength range [nm]:").pack(side="left")
    ttk.Entry(wrow, textvariable=wmin_var, width=8).pack(side="left", padx=(4, 2))
    ttk.Label(wrow, text="–").pack(side="left")
    ttk.Entry(wrow, textvariable=wmax_var, width=8).pack(side="left", padx=2)
    ttk.Label(wrow, text="(empty: full range)").pack(side="left", padx=4)

    prow = ttk.Frame(main)
    prow.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

    spectro_var = tk.StringVar()
    min_vsini_var = tk.StringVar(value="")
    spectro_choices = ([f"{n} (R={r})" for n, r, _ in SPECTROGRAPHS]
                       + ["Custom (enter R manually)"])

    def update_min_vsini_hint(*_):
        """Δv_inst ≈ c/R: below this vsini cannot be resolved."""
        s = res_var.get().strip()
        try:
            r = float(s)
            min_vsini_var.set(f"min vsini ≈ c/R = {C_KMS / r:.1f} km/s")
        except ValueError:
            min_vsini_var.set("")

    def on_spectrograph_selected(_event=None):
        sel = spectro_var.get()
        for n, r, _note in SPECTROGRAPHS:
            if sel.startswith(n + " "):
                res_var.set(str(r))
                update_min_vsini_hint()
                return
        res_var.set("")
        update_min_vsini_hint()

    ttk.Label(prow, text="Spectrograph:").pack(side="left")
    spectro_box = ttk.Combobox(prow, textvariable=spectro_var,
                               values=spectro_choices, width=24,
                               state="readonly")
    spectro_box.pack(side="left", padx=(4, 10))
    spectro_box.bind("<<ComboboxSelected>>", on_spectrograph_selected)

    ttk.Label(prow, text="R:").pack(side="left")
    res_entry = ttk.Entry(prow, textvariable=res_var, width=9)
    res_entry.pack(side="left", padx=(2, 10))
    res_entry.bind("<KeyRelease>", update_min_vsini_hint)

    ttk.Label(prow, text="vsini [km/s]:").pack(side="left")
    ttk.Entry(prow, textvariable=vsini_var, width=7).pack(side="left", padx=2)
    ttk.Label(prow, textvariable=min_vsini_var, foreground="gray"
              ).pack(side="left", padx=6)

    # Method and BF-mode settings. They are edited in the small "Method..."
    # dialog (opened like "Normalize raw..."); the gray summary label next
    # to the button reflects the current choice.
    method_var = tk.StringVar(value="BF")
    bf_mode_var = tk.StringVar(value="auto")  # "assume" | "auto"
    rvmin_var = tk.StringVar()
    rvmax_var = tk.StringVar()
    rvstep_var = tk.StringVar(value="0.5")
    ccf_mode_var = tk.StringVar(value="Full template")
    exclude_h_var = tk.BooleanVar(value=True)
    vrange_var = tk.StringVar(value="500")
    comp_var = tk.IntVar(value=2)
    gamma_var = tk.StringVar()
    bf_profile_var = tk.StringVar(value="Gaussian")
    guess_var = tk.StringVar()
    guess_amp_var = tk.StringVar()
    guess_sigma_var = tk.StringVar()
    method_summary_var = tk.StringVar()
    # TODCOR: two templates, a log-lambda grid in km/s per pixel
    tod_dv_var = tk.StringVar(value="1.0")
    tod_maxrv_var = tk.StringVar(value="200")
    tod_alpha_var = tk.StringVar()
    tod_vsini2_var = tk.StringVar()
    tod_clean_var = tk.BooleanVar(value=True)
    tod_sim_var = tk.BooleanVar(value=False)
    tod_simsep_var = tk.StringVar()

    def update_method_summary():
        """Also switches the file section: TODCOR needs a second
        template, every other method a single one."""
        show_tpl2(method_var.get() == "TODCOR")
        spec_label_var.set("Normalized spectrum:")
        for w in (tpl_entry, tpl_browse):
            w.configure(state="normal")
        tpl_label.configure(text="Synthetic spectrum:", foreground="")
        if method_var.get() == "CCF":
            method_summary_var.set("CCF (single star)")
        elif method_var.get() == "TODCOR":
            tpl_label.configure(text="Template 1 (primary):", foreground="")
            if tod_sim_var.get():
                method_summary_var.set("TODCOR self-calibration")
            else:
                method_summary_var.set(
                    f"TODCOR, {_f(tod_dv_var, 1.0):g} km/s/px, "
                    f"|v| <= {_f(tod_maxrv_var, 200.0):g} km/s, "
                    + ("alpha fixed at "
                       f"{_f(tod_alpha_var):g}" if tod_alpha_var.get().strip()
                       else "alpha fitted"))
        else:
            mode = ("with assumptions" if bf_mode_var.get() == "assume"
                    else "automatic component search")
            method_summary_var.set(f"BF, {int(comp_var.get())} "
                                   f"component(s), {mode}")

    update_method_summary()

    result_text = tk.Text(main, height=8, width=90, state="disabled",
                          font=("Courier", 10))
    result_text.grid(row=8, column=0, columnspan=3, sticky="ew", pady=6)

    plot_frame = ttk.Frame(main)
    plot_frame.grid(row=9, column=0, columnspan=3, sticky="nsew")
    main.rowconfigure(9, weight=1)
    canvas_holder = {"canvas": None}

    def show_text(txt):
        result_text.configure(state="normal")
        result_text.delete("1.0", "end")
        result_text.insert("1.0", txt)
        result_text.configure(state="disabled")

    def show_spectrum_window(path):
        """Plot the selected spectrum in its own window, with the matplotlib
        navigation toolbar so it can be zoomed, panned and re-scaled the way
        a MIDAS plot can. One line per echelle order."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        try:
            orders = [(np.asarray(w, float), np.asarray(f, float))
                      for w, f in read_spectrum(path)]
        except Exception as exc:
            messagebox.showerror(
                "Spectrum", f"Could not plot {os.path.basename(path)}:\n{exc}")
            return
        orders = [(w, f) for w, f in orders if w.size > 1]
        if not orders:
            messagebox.showinfo("Spectrum", "No plottable data in that file.")
            return

        win = tk.Toplevel(root)
        win.title(f"Spectrum - {os.path.basename(path)}")
        fig = Figure(figsize=(11, 5))
        ax = fig.add_subplot(111)
        for w, f in orders:
            ax.plot(w, f, lw=0.6)
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("Flux")
        lo = min(w[0] for w, _ in orders)
        hi = max(w[-1] for w, _ in orders)
        ax.set_title(f"{os.path.basename(path)} - {len(orders)} order(s), "
                     f"{lo:.1f}-{hi:.1f} A")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        NavigationToolbar2Tk(canvas, win).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_header():
        """Standalone header inspector: dump the FITS header into the
        result box, plot the spectrum in its own zoomable window, and
        auto-fill the Target fields from the header."""
        path = spec_var.get().strip()
        if not path:
            messagebox.showinfo("Header", "Select a spectrum file first.")
            return
        show_spectrum_window(path)
        try:
            payload = cmd_header(make_args(spectrum=path,
                                           site=resolve_site()))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        show_text(payload["text"])
        info = payload["info"]
        if info["object"] and not target_var.get().strip():
            target_var.set(str(info["object"]))
        if info["ra"] is not None and not ra_var.get().strip():
            ra_var.set(f"{info['ra']:.5f}")
            dec_var.set(f"{info['dec']:.5f}")
        if info["obstime"] and not time_var.get().strip():
            time_var.set(str(info["obstime"]))

    def _f(var, default=None):
        s = var.get().strip()
        return float(s) if s else default

    def show_payload(payload):
        if payload.get("preview"):
            note = ("Preview only - nothing written to disk; press Run "
                    "for the full fit.")
        elif payload.get("saved", True):
            # after a Save the real destinations are on the payload; before
            # one (CLI runs) fall back to the names the run chose
            written = payload.get("saved_files") or (
                [payload["output"]] + [n for _, n in payload.get("figs", [])])
            note = "Saved: " + ", ".join(written)
        else:
            note = ("Results NOT saved yet - press 'Save' to write the "
                    "result files.")
        result_text.configure(state="normal")
        result_text.delete("1.0", "end")
        result_text.insert("1.0", payload["text"] + "\n" + note)
        result_text.configure(state="disabled")

        if canvas_holder["canvas"] is not None:
            canvas_holder["canvas"].get_tk_widget().destroy()
            canvas_holder["canvas"] = None
        # The TODCOR self-calibration is a table, not a fit: no figure.
        if payload.get("fig") is None:
            return
        canvas = FigureCanvasTkAgg(payload["fig"], master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas_holder["canvas"] = canvas

    def normalize_dialog():
        dlg = tk.Toplevel(root)
        dlg.title("Normalize raw spectrum")
        dlg.transient(root)
        frm = ttk.Frame(dlg, padding=10)
        frm.grid(row=0, column=0)

        raw_var = tk.StringVar()
        ord_var = tk.StringVar(value="3")
        it_var = tk.StringVar(value="20")
        fitm_var = tk.StringVar(value="Polynomial")

        ttk.Label(frm, text="Raw spectrum (FITS/ASCII):").grid(row=0, column=0,
                                                               sticky="w")
        ttk.Entry(frm, textvariable=raw_var, width=42).grid(row=0, column=1,
                                                            padx=4)
        ttk.Button(frm, text="Browse...",
                   command=lambda: browse(raw_var, "Select raw spectrum")
                   ).grid(row=0, column=2)

        prow = ttk.Frame(frm)
        prow.grid(row=1, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(prow, text="Continuum fit").pack(side="left")
        fitm_box = ttk.Combobox(prow, textvariable=fitm_var, width=14,
                                state="readonly",
                                values=["Polynomial", "B-spline"])
        fitm_box.pack(side="left", padx=(2, 12))
        ttk.Label(prow, text="Polynomial order").pack(side="left")
        ord_entry = ttk.Entry(prow, textvariable=ord_var, width=4)
        ord_entry.pack(side="left", padx=(2, 12))
        ttk.Label(prow, text="Iterations").pack(side="left")
        ttk.Entry(prow, textvariable=it_var, width=4).pack(side="left",
                                                           padx=2)

        def on_fit_method_change(_event=None):
            """The B-spline is always cubic, so the polynomial order is
            meaningless for it; the clipping iterations are used by
            both models and stay active."""
            ord_entry.configure(
                state="disabled" if fitm_var.get().startswith("B-")
                else "normal")

        fitm_box.bind("<<ComboboxSelected>>", on_fit_method_change)

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
                          iterations=int(it_var.get()),
                          fit_method=("bspline"
                                      if fitm_var.get().startswith("B-")
                                      else "poly"),
                          resolution=_f(res_var), vsini=_f(vsini_var))
                state["payload"] = cmd_normalize(make_args(**kw))
            except Exception as exc:
                messagebox.showerror("Error", str(exc), parent=dlg)
                return
            show_payload(state["payload"])
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
    ttk.Button(fbtns, text="Header", command=show_header
               ).pack(side="left", padx=(4, 0))

    def method_dialog():
        """Method settings in a small sub-window (like 'Normalize raw...'):
        CCF or BF; for BF also the components and the mode — either
        'with assumptions' (per-component RV/Amp/Sigma read off a
        smoothed-BF preview shown with one button) or the automatic
        component search, which disables all assumption inputs."""
        dlg = tk.Toplevel(root)
        dlg.title("Method")
        dlg.transient(root)
        frm = ttk.Frame(dlg, padding=10)
        frm.grid(row=0, column=0)

        mrow = ttk.Frame(frm)
        mrow.grid(row=0, column=0, sticky="w")
        ttk.Label(mrow, text="Method:").pack(side="left")

        ccf_sec = ttk.LabelFrame(frm, text="CCF settings", padding=6)
        bf_sec = ttk.LabelFrame(frm, text="BF settings", padding=6)
        tod_sec = ttk.LabelFrame(frm, text="TODCOR settings", padding=6)

        def refresh_method():
            for sec in (ccf_sec, bf_sec, tod_sec):
                sec.grid_remove()
            {"CCF": ccf_sec, "BF": bf_sec,
             "TODCOR": tod_sec}[method_var.get()].grid(
                row=1, column=0, sticky="ew", pady=(6, 0))
            update_method_summary()

        ttk.Radiobutton(mrow, text="CCF (single star)", variable=method_var,
                        value="CCF", command=refresh_method
                        ).pack(side="left", padx=8)
        ttk.Radiobutton(mrow, text="BF (binary/multiple)",
                        variable=method_var, value="BF",
                        command=refresh_method).pack(side="left", padx=8)
        ttk.Radiobutton(mrow, text="TODCOR (blended SB2)",
                        variable=method_var, value="TODCOR",
                        command=refresh_method).pack(side="left", padx=8)

        for j, (lbl, var) in enumerate([("RV min", rvmin_var),
                                        ("RV max", rvmax_var),
                                        ("step", rvstep_var)]):
            ttk.Label(ccf_sec, text=lbl).grid(row=0, column=2 * j,
                                              sticky="w")
            ttk.Entry(ccf_sec, textvariable=var, width=8
                      ).grid(row=0, column=2 * j + 1, padx=(2, 10))
        ttk.Label(ccf_sec, text="[km/s]  (empty: auto wide search)",
                  foreground="gray").grid(row=0, column=6, sticky="w")
        ttk.Label(ccf_sec, text="Mode:").grid(row=1, column=0, sticky="w",
                                              pady=(6, 0))
        ttk.Combobox(ccf_sec, textvariable=ccf_mode_var, width=14,
                     state="readonly",
                     values=["Line mask", "Full template"]
                     ).grid(row=1, column=1, columnspan=2, sticky="w",
                            padx=2, pady=(6, 0))
        ttk.Checkbutton(ccf_sec, text="Exclude hydrogen lines (hot stars)",
                        variable=exclude_h_var
                        ).grid(row=1, column=3, columnspan=4, sticky="w",
                               pady=(6, 0))

        # --- TODCOR ----------------------------------------------------
        trow1 = ttk.Frame(tod_sec)
        trow1.grid(row=0, column=0, columnspan=8, sticky="w")
        for lbl, var, w, unit in [("Grid step", tod_dv_var, 6, "km/s/px"),
                                  ("   Search ±", tod_maxrv_var, 6, "km/s"),
                                  ("   Light ratio alpha", tod_alpha_var, 6,
                                   "(empty: fitted)"),
                                  ("   vsini template 2", tod_vsini2_var, 6,
                                   "km/s")]:
            ttk.Label(trow1, text=lbl).pack(side="left")
            e = ttk.Entry(trow1, textvariable=var, width=w)
            e.pack(side="left", padx=2)
            e.bind("<KeyRelease>", lambda _e: update_method_summary())
            ttk.Label(trow1, text=unit, foreground="gray").pack(side="left",
                                                                padx=(0, 4))

        ttk.Checkbutton(tod_sec, text="Clean the observed spectrum first",
                        variable=tod_clean_var
                        ).grid(row=1, column=0, columnspan=8, sticky="w",
                               pady=(6, 0))

        trow2 = ttk.Frame(tod_sec)
        trow2.grid(row=2, column=0, columnspan=8, sticky="w", pady=(6, 0))
        ttk.Checkbutton(trow2, text="Self-calibration (no spectrum needed)",
                        variable=tod_sim_var,
                        command=update_method_summary).pack(side="left")
        ttk.Label(trow2, text="   separations [km/s]:").pack(side="left")
        ttk.Entry(trow2, textvariable=tod_simsep_var,
                  width=24).pack(side="left", padx=2)
        ttk.Label(trow2, text="(empty: 10...90)",
                  foreground="gray").pack(side="left", padx=4)

        top = ttk.Frame(bf_sec)
        top.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(top, text="Velocity window ±").pack(side="left")
        ttk.Entry(top, textvariable=vrange_var, width=7).pack(side="left",
                                                              padx=2)
        ttk.Label(top, text="km/s").pack(side="left", padx=(0, 10))
        ttk.Label(top, text="Components").pack(side="left")
        comp_box = ttk.Combobox(top, textvariable=comp_var,
                                values=[1, 2, 3], width=3,
                                state="readonly")
        comp_box.pack(side="left", padx=4)
        comp_box.bind("<<ComboboxSelected>>",
                      lambda _e: update_method_summary())
        ttk.Label(top, text="Systemic gamma [km/s]:").pack(side="left",
                                                           padx=(10, 2))
        ttk.Entry(top, textvariable=gamma_var, width=7).pack(side="left")
        ttk.Label(top, text="Fit profile:").pack(side="left", padx=(10, 2))
        ttk.Combobox(top, textvariable=bf_profile_var, width=15,
                     state="readonly",
                     values=["Gaussian", "Rotational (Gray)"]
                     ).pack(side="left")

        assume_widgets = []

        def refresh_mode():
            st = "normal" if bf_mode_var.get() == "assume" else "disabled"
            for w in assume_widgets:
                w.configure(state=st)
            update_method_summary()

        ttk.Radiobutton(bf_sec, text="With assumptions — read the values "
                                     "off the smoothed BF first",
                        variable=bf_mode_var, value="assume",
                        command=refresh_mode
                        ).grid(row=1, column=0, columnspan=3, sticky="w",
                               pady=(8, 0))

        def show_smoothed_bf():
            try:
                if not spec_var.get().strip():
                    raise ValueError("No normalized spectrum file selected "
                                     "in the main window.")
                if not tpl_var.get().strip():
                    raise ValueError("No synthetic spectrum file selected "
                                     "in the main window.")
                kw = dict(spectrum=spec_var.get().strip(),
                          template=tpl_var.get().strip(),
                          wave_min=_f(wmin_var), wave_max=_f(wmax_var),
                          resolution=_f(res_var), vsini=_f(vsini_var),
                          vel_range=_f(vrange_var, 500.0))
                payload = cmd_bf_preview(make_args(**kw))
            except Exception as exc:
                messagebox.showerror("Error", str(exc), parent=dlg)
                return
            show_payload(payload)

        prevrow = ttk.Frame(bf_sec)
        prevrow.grid(row=2, column=0, columnspan=3, sticky="w",
                     padx=(18, 0))
        preview_btn = ttk.Button(prevrow, text="Show smoothed BF",
                                 command=show_smoothed_bf)
        preview_btn.pack(side="left")
        ttk.Label(prevrow, text="— read the values, fill in the blanks",
                  foreground="gray").pack(side="left", padx=6)
        assume_widgets.append(preview_btn)

        arow = ttk.Frame(bf_sec)
        arow.grid(row=3, column=0, columnspan=3, sticky="w",
                  padx=(18, 0), pady=(4, 0))
        ttk.Label(arow, text="RV assumption [km/s]:").grid(row=0, column=0,
                                                           sticky="w")
        rv_entry = ttk.Entry(arow, textvariable=guess_var, width=18)
        rv_entry.grid(row=0, column=1, sticky="w", padx=(2, 12))
        ttk.Label(arow, text="Amp assumption:").grid(row=1, column=0,
                                                     sticky="w",
                                                     pady=(4, 0))
        amp_entry = ttk.Entry(arow, textvariable=guess_amp_var, width=18)
        amp_entry.grid(row=1, column=1, sticky="w", padx=(2, 12),
                       pady=(4, 0))
        ttk.Label(arow, text="Sigma assumption [km/s]:").grid(row=1,
                                                              column=2,
                                                              sticky="w",
                                                              pady=(4, 0))
        sigma_entry = ttk.Entry(arow, textvariable=guess_sigma_var,
                                width=14)
        sigma_entry.grid(row=1, column=3, sticky="w", padx=2, pady=(4, 0))
        ttk.Label(arow, text="One value per component, comma-separated — "
                             "largest Gaussian first, smallest last "
                             "(sets C1, C2, ...).",
                  foreground="gray").grid(row=2, column=0, columnspan=4,
                                          sticky="w", pady=(2, 0))
        assume_widgets += [rv_entry, amp_entry, sigma_entry]

        ttk.Radiobutton(bf_sec, text="Automatic component search — no "
                                     "input needed",
                        variable=bf_mode_var, value="auto",
                        command=refresh_mode
                        ).grid(row=4, column=0, columnspan=3, sticky="w",
                               pady=(8, 0))

        ttk.Button(frm, text="Use",
                   command=lambda: (update_method_summary(), dlg.destroy())
                   ).grid(row=2, column=0, pady=(10, 0))

        refresh_method()
        refresh_mode()

    mrow_main = ttk.Frame(main)
    mrow_main.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Button(mrow_main, text="Method...", command=method_dialog
               ).pack(side="left")
    ttk.Label(mrow_main, textvariable=method_summary_var, foreground="gray"
              ).pack(side="left", padx=8)

    run_payload = {"payload": None}

    def run_analysis():
        run_payload["payload"] = None
        # Recognise a correlation profile BEFORE asking for a template:
        # every method here works on a spectrum, so this file is simply
        # the wrong input and saying so now saves a confusing failure
        # deep inside the reader.
        note = correlation_profile_note(spec_var.get().strip())
        if note:
            messagebox.showerror("This file is a correlation profile", note)
            return
        try:
            todcor = method_var.get() == "TODCOR"
            simulate = todcor and bool(tod_sim_var.get())
            # The self-calibration builds its composites from the templates
            # alone, so it is the one run that needs no observed spectrum.
            if not spec_var.get().strip() and not simulate:
                raise ValueError("No normalized spectrum file selected "
                                 "(use 'Normalize raw...' if you only have "
                                 "a raw spectrum).")
            if not tpl_var.get().strip():
                raise ValueError("No template 1 (primary) file selected."
                                 if todcor else
                                 "No synthetic spectrum file selected.")
            # Template 2 may be left empty: TODCOR always needs two template
            # arrays, but for a twin binary the same file is used for both,
            # which is the usual case and keeps the surface exactly
            # symmetric. cmd_todcor does the defaulting.
            r_val, v_val = _f(res_var), _f(vsini_var)
            if r_val and v_val and v_val < min_vsini(r_val):
                messagebox.showwarning(
                    "vsini below the instrumental limit",
                    f"vsini = {v_val:g} km/s is below the instrumental "
                    f"broadening c/R = {min_vsini(r_val):.1f} km/s at "
                    f"R = {r_val:g}.\n\nRotation this small cannot be "
                    "resolved; the rotational broadening will be skipped. "
                    f"Use vsini ≥ {min_vsini(r_val):.1f} km/s to apply it.")
            kw = dict(spectrum=spec_var.get().strip(),
                      template=tpl_var.get().strip(),
                      wave_min=_f(wmin_var), wave_max=_f(wmax_var),
                      object=target_var.get().strip() or None,
                      ra=ra_var.get().strip() or None,
                      dec=dec_var.get().strip() or None,
                      obstime=time_var.get().strip() or None,
                      site=resolve_site(),
                      no_bary=not bary_var.get(),
                      t0=_f(t0_var), period=_f(per_var),
                      resolution=_f(res_var), vsini=_f(vsini_var),
                      no_save=True)
            if method_var.get() == "CCF":
                kw.update(rv_min=_f(rvmin_var),
                          rv_max=_f(rvmax_var),
                          rv_step=_f(rvstep_var, 0.5),
                          ccf_mode=("mask"
                                    if ccf_mode_var.get() == "Line mask"
                                    else "template"),
                          keep_balmer=not exclude_h_var.get())
                payload = cmd_ccf(make_args(**kw))
            elif todcor:
                sep = [float(t) for t in
                       tod_simsep_var.get().replace(",", " ").split()]
                kw.update(spectrum=spec_var.get().strip() or None,
                          template1=tpl_var.get().strip(),
                          template2=tpl2_var.get().strip(),
                          vsini1=_f(vsini_var), vsini2=_f(tod_vsini2_var),
                          dv=_f(tod_dv_var, 1.0),
                          max_rv=_f(tod_maxrv_var, 200.0),
                          alpha=_f(tod_alpha_var),
                          no_clean=not tod_clean_var.get(),
                          simulate=simulate, sim_sep=sep or None)
                payload = cmd_todcor(make_args(**kw))
            else:
                guess = guess_amp = guess_sigma = None
                if bf_mode_var.get() == "assume":
                    gtxt = guess_var.get().replace(",", " ").split()
                    gatxt = guess_amp_var.get().replace(",", " ").split()
                    gstxt = guess_sigma_var.get().replace(",", " ").split()
                    if not gtxt:
                        raise ValueError(
                            "The assumption mode needs at least the RV "
                            "assumptions: open 'Method...', press 'Show "
                            "smoothed BF' and read the values off the "
                            "curve — or switch to the automatic "
                            "component search.")
                    guess = [float(t) for t in gtxt]
                    guess_amp = [float(t) for t in gatxt] or None
                    guess_sigma = [float(t) for t in gstxt] or None
                kw.update(vel_range=_f(vrange_var, 500.0),
                          components=int(comp_var.get()),
                          guess=guess, guess_amp=guess_amp,
                          guess_sigma=guess_sigma,
                          gamma=_f(gamma_var),
                          profile=("rot" if bf_profile_var.get().startswith(
                              "Rot") else "gauss"))
                payload = cmd_bf(make_args(**kw))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return
        run_payload["payload"] = payload
        show_payload(payload)

    def save_results():
        payload = run_payload["payload"]
        if payload is None:
            messagebox.showinfo("Save", "Run an analysis first.")
            return
        start = os.path.dirname(os.path.abspath(payload["output"]))
        outdir = filedialog.askdirectory(
            title="Save results in...",
            initialdir=start if os.path.isdir(start) else os.getcwd(),
            mustexist=False)
        if not outdir:            # cancelled - write nothing
            return
        try:
            files = save_payload(payload, directory=outdir)
        except Exception as exc:
            messagebox.showerror("Save", str(exc))
            return
        show_payload(payload)
        messagebox.showinfo("Save", "Files written:\n" + "\n".join(files))

    btnrow = ttk.Frame(main)
    btnrow.grid(row=7, column=0, columnspan=3, pady=6)
    ttk.Button(btnrow, text="Run", command=run_analysis).pack(side="left",
                                                              padx=4)
    ttk.Button(btnrow, text="Save", command=save_results).pack(side="left",
                                                               padx=4)

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
    p.add_argument("--wave-min", type=float,
                   help="Minimum wavelength to use [nm]")
    p.add_argument("--wave-max", type=float,
                   help="Maximum wavelength to use [nm]")
    p.add_argument("--spectrograph",
                   help="Built-in instrument; sets R automatically. Known: "
                        + ", ".join(n for n, _, _ in SPECTROGRAPHS))
    p.add_argument("--resolution", type=float,
                   help="Resolving power R = lambda/dlambda entered manually "
                        "(overrides --spectrograph). CCF: the template is "
                        "degraded to R before the measurement. BF: R sets "
                        "the BF smoothing FWHM c/R (saphires practice); "
                        "add --degrade-template to also degrade a "
                        "synthetic template")
    p.add_argument("--vsini", type=float,
                   help="Rotational broadening vsini [km/s] applied to the "
                        "template before the measurement")
    p.add_argument("--epsilon", type=float, default=0.6,
                   help="Linear limb-darkening coefficient for the "
                        "rotational profile (default 0.6)")
    p.add_argument("--object", help="Star name; coordinates are resolved via "
                                    "SIMBAD for the barycentric correction")
    p.add_argument("--ra", help="RA for the barycentric correction: decimal "
                                "degrees or sexagesimal hours "
                                "(12:50:39.7 / '12 50 39.7')")
    p.add_argument("--dec", help="Dec for the barycentric correction: "
                                 "decimal degrees or sexagesimal "
                                 "(-03:07:49.8 / '-03 07 49.8')")
    p.add_argument("--obstime", help="Observation time: ISOT "
                                     "(2024-12-03T02:30:00), BJD number "
                                     "(2453254.847090365) or old FITS "
                                     "date DD/MM/YY (16/10/98, optionally "
                                     "+ UT time)")
    p.add_argument("--site", default=None,
                   help="Observatory: a built-in name (paranal, lasilla, "
                        "ctio, ohp, tug, kreiken, lco, soar, ... — see "
                        "--list-observatories), a 'lonE,lat,alt' string "
                        "in degrees/metres, or an astropy site name. "
                        + AUTO_SITE_HELP)
    p.add_argument("--no-bary", action="store_true",
                   help="Compute the barycentric correction but do NOT "
                        "apply it to the RVs: the value is kept in the "
                        "result file for information only and does not "
                        "appear on the figure")
    p.add_argument("--force-bary", action="store_true",
                   help="Apply the barycentric correction even when the "
                        "spectrum header says the wavelengths are "
                        "already barycentric (the header check "
                        "otherwise stops the run to prevent a double "
                        "correction)")
    p.add_argument("--t0", type=float,
                   help="Ephemeris T0 [BJD] (primary minimum) for the "
                        "orbital phase")
    p.add_argument("--period", type=float,
                   help="Orbital period [days] for the orbital phase")
    p.add_argument("--varastro", action="store_true",
                   help="Fetch T0/P from the VarAstro database "
                        "(var.astro.cz) using the --object name when "
                        "--t0/--period are not given; values come in "
                        "VarAstro's native units (T0 in HJD, P in days)")
    p.add_argument("--t0-to-bjd", action="store_true",
                   help="Convert the ephemeris T0 from HJD to BJD_TDB "
                        "before the analysis (needs target coordinates); "
                        "the converted value is then used for the "
                        "orbital phase")
    p.add_argument("--plot", help="Figure PNG file name "
                                  "(default: result_CCF.png / result_BF.png)")
    p.add_argument("--output", help="Result text file name "
                                    "(default: result_CCF.txt / result_BF.txt)")
    p.add_argument("--dpi", type=float,
                   help="PNG figure resolution in dots per inch "
                        "(default 600)")


def print_observatories():
    print("Built-in observatories (longitude East-positive):")
    print(f"  {'name':<34} {'lon[deg]':>10} {'lat[deg]':>9} {'alt[m]':>7}  "
          "instruments / note")
    for oname, lon, lat, alt, _aliases, note in OBSERVATORIES:
        print(f"  {oname:<34} {lon:10.4f} {lat:9.4f} {alt:7.0f}  {note}")
    print("\nGive any of these to --site (or a 'lonE,lat,alt' string, or an "
          "astropy site name).")


def print_spectrographs():
    print("Built-in spectrographs (representative resolving power R):")
    for name, r, note in SPECTROGRAPHS:
        print(f"  {name:<16} R = {r:>7}   {note}")
    print("\nGive any of these to --spectrograph, or set R with "
          "--resolution.")


ESO_TAP_URL = "https://archive.eso.org/tap_obs/sync"
ESO_DATAPORTAL_URL = "https://dataportal.eso.org/dataportal_new/file/"


def eso_tap_query(adql, timeout=120, retries=3):
    """Run a synchronous ADQL query against the ESO TAP service and
    return the result rows as a list of dicts (csv-parsed). The service
    intermittently drops SSL handshakes, so failed attempts are retried
    (with a short pause) before giving up."""
    import csv
    import io
    import time
    from urllib.request import urlopen
    from urllib.parse import urlencode
    url = ESO_TAP_URL + "?" + urlencode(dict(
        REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=adql))
    for attempt in range(retries):
        try:
            text = urlopen(url, timeout=timeout).read().decode(
                "utf-8", "replace")
            return list(csv.DictReader(io.StringIO(text)))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3.0 * (attempt + 1))


def cmd_fetch_adp(args):
    """Download the reduced (ADP phase-3) counterpart of raw ESO frames.

    For every input FITS file the primary header gives the instrument,
    the pointing (RA/DEC) and the exposure start (MJD-OBS). The ESO TAP
    service (ivoa.ObsCore) is searched for reduced spectra of that
    instrument within --radius arcsec of the pointing, and the product
    whose start time t_min matches MJD-OBS to within --tmatch seconds
    is downloaded from the dataportal into the raw file's directory
    (or --dest). Existing files are not downloaded again.

    This is the cure for the 'this is a RAW ... frame' loader error:
    raw echelle CCD exposures cannot be analyzed directly, but their
    pipeline-reduced ADP products can.
    """
    import glob
    from urllib.request import urlretrieve
    from astropy.io import fits

    files = sorted(set(sum((glob.glob(p) for p in args.spectra), [])))
    files = [f for f in files if not os.path.basename(f).startswith("ADP.")]
    if not files:
        sys.exit("No files match the given pattern(s) "
                 "(ADP.* files are skipped - they are already reduced).")

    tol_day = args.tmatch / 86400.0
    radius_deg = args.radius / 3600.0
    fetched, skipped, failed = [], [], []
    for path in files:
        name = os.path.basename(path)
        try:
            hdr = fits.getheader(path)
        except Exception as exc:
            print(f"{name}: cannot read header ({exc}); skipped.")
            failed.append(name)
            continue
        inst = str(hdr.get("INSTRUME", "")).strip()
        ra, dec, mjd = hdr.get("RA"), hdr.get("DEC"), hdr.get("MJD-OBS")
        if not inst or None in (ra, dec, mjd):
            print(f"{name}: header lacks INSTRUME/RA/DEC/MJD-OBS; skipped.")
            failed.append(name)
            continue
        adql = (
            "SELECT dp_id, t_min, target_name, snr FROM ivoa.ObsCore "
            "WHERE dataproduct_type = 'spectrum' "
            f"AND instrument_name = '{inst}' "
            f"AND CONTAINS(POINT('ICRS', s_ra, s_dec), "
            f"CIRCLE('ICRS', {float(ra):.6f}, {float(dec):.6f}, "
            f"{radius_deg:.6f})) = 1")
        try:
            rows = eso_tap_query(adql)
        except Exception as exc:
            print(f"{name}: ESO TAP query failed ({exc}).")
            failed.append(name)
            continue
        best, best_dt = None, tol_day
        for r in rows:
            try:
                dt = abs(float(r["t_min"]) - float(mjd))
            except (KeyError, TypeError, ValueError):
                continue
            if dt <= best_dt:
                best, best_dt = r, dt
        if best is None:
            print(f"{name}: no reduced {inst} spectrum within "
                  f"{args.tmatch:g} s of MJD {mjd} found in the archive "
                  f"({len(rows)} products at this position).")
            failed.append(name)
            continue
        dp_id = best["dp_id"]
        dest_dir = args.dest or os.path.dirname(os.path.abspath(path))
        dest = os.path.join(dest_dir, dp_id + ".fits")
        if os.path.exists(dest):
            print(f"{name} -> {dp_id}.fits (already present, "
                  f"dt = {best_dt * 86400:.1f} s)")
            skipped.append(dest)
            continue
        if args.dry_run:
            print(f"{name} -> would fetch {dp_id}.fits "
                  f"(dt = {best_dt * 86400:.1f} s, "
                  f"target {best.get('target_name', '?')})")
            fetched.append(dest)
            continue
        err = None
        for attempt in range(3):
            try:
                os.makedirs(dest_dir, exist_ok=True)
                urlretrieve(ESO_DATAPORTAL_URL + dp_id, dest)
                err = None
                break
            except Exception as exc:
                err = exc
                if os.path.exists(dest):
                    os.remove(dest)
        if err is not None:
            print(f"{name}: download of {dp_id} failed ({err}).")
            failed.append(name)
            continue
        print(f"{name} -> {dp_id}.fits "
              f"(dt = {best_dt * 86400:.1f} s, "
              f"target {best.get('target_name', '?')}, "
              f"snr {best.get('snr', '?')})")
        fetched.append(dest)

    print(f"\n{len(fetched)} downloaded"
          + (" (dry run)" if args.dry_run else "")
          + f", {len(skipped)} already present, {len(failed)} without a "
          "match.")
    got = fetched + skipped
    if got and not args.dry_run:
        d = os.path.dirname(got[0])
        print("Analyze the reduced spectra with e.g.:\n"
              f"  rv_analysis.py batch --spectra '{os.path.join(d, 'ADP.*.fits')}' ...")


def run_cli():
    if len(sys.argv) == 1:
        run_interactive()
        return
    if "--list-observatories" in sys.argv:
        print_observatories()
        return
    if "--list-spectrographs" in sys.argv:
        print_spectrographs()
        return

    parser = argparse.ArgumentParser(
        description="Radial velocity from spectra: CCF and BF methods. "
                    "Run without arguments for the interactive mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-observatories", action="store_true",
                        help="Print the built-in observatory list and exit")
    parser.add_argument("--list-spectrographs", action="store_true",
                        help="Print the built-in spectrograph list and exit")
    # --log / --no-log are taken out of sys.argv by pop_log_args() before
    # the parser runs, so that the log is already open while the command
    # line is parsed. They are declared here only so that --help lists
    # them; argparse itself never sees them.
    parser.add_argument("--log", metavar="FILE",
                        help=f"Run log file (default: {LOG_DEFAULT_NAME} in "
                             "the current directory; runs are appended). "
                             "Works before or after the command name")
    parser.add_argument("--no-log", action="store_true",
                        help="Do not write the run log file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gui = sub.add_parser("gui", help="Open the widget (same as no arguments)")
    p_gui.set_defaults(func=lambda a: run_interactive())

    p_ccf = sub.add_parser("ccf", help="Cross-correlation (weighted line "
                                       "mask; Baranne 1996 / Pepe 2002 / "
                                       "Pino et al. 2018)")
    add_common_args(p_ccf)
    p_ccf.add_argument("--rv-min", type=float, default=None,
                       help="RV scan lower limit [km/s] (default: "
                            "automatic two-stage search — coarse "
                            "+-900 km/s at 10 km/s, then refined around "
                            "the peak; Katz et al. 2025 SOPHIE practice)")
    p_ccf.add_argument("--rv-max", type=float, default=None,
                       help="RV scan upper limit [km/s] (default: "
                            "automatic two-stage search)")
    p_ccf.add_argument("--rv-step", type=float, default=0.5,
                       help="Fine RV step [km/s]")
    p_ccf.add_argument("--ccf-mode", choices=["mask", "template"],
                       default="mask",
                       help="'mask' (default): weighted line-mask CCF — "
                            "reliable for hot (OBA) stars because the "
                            "broad hydrogen wings drop out; 'template': "
                            "the previous full-spectrum cross-covariance")
    p_ccf.add_argument("--mask-depth", type=float, default=0.05,
                       help="Minimum template line depth for a mask entry "
                            "(mask mode)")
    p_ccf.add_argument("--keep-balmer", action="store_true",
                       help="Keep the hydrogen Balmer/Paschen lines in "
                            "the mask (default: excluded — hot-star "
                            "practice)")
    p_ccf.set_defaults(func=cmd_ccf)

    p_tod = sub.add_parser(
        "todcor",
        help="TODCOR: two-dimensional correlation with one template per "
             "component (Zucker & Mazeh 1994) - the method for SB2s whose "
             "CCF peaks blend",
        description="Correlates the spectrum against the combination "
                    "t1(v1) + alpha*t2(v2) and maximises over both "
                    "velocities at once, so a blend is modelled instead of "
                    "being fitted around. Use it when the two peaks of a "
                    "1-D CCF/BF are no longer cleanly separated: at small "
                    "separations TODCOR beats the BF, while at low S/N the "
                    "BF is the more robust of the two, and TODCOR is the "
                    "method most sensitive to a template mismatch - its "
                    "flux ratio first of all (saphires, Tofflemire). It "
                    "needs the SPECTRUM and two templates - it cannot run "
                    "on a pre-computed correlation profile.")
    p_tod.add_argument("--spectrum",
                       help="Normalized observed spectrum (not needed with "
                            "--simulate)")
    p_tod.add_argument("--format", default="auto",
                       choices=["auto", "s2d", "text"])
    p_tod.add_argument("--template1", required=True,
                       help="Template for the primary component")
    p_tod.add_argument("--template2",
                       help="Template for the secondary component. Omit it "
                            "for a twin binary and --template1 is used for "
                            "both, which is the usual choice unless the two "
                            "stars really differ in spectral type: two "
                            "near-identical but non-identical templates "
                            "break the mirror symmetry for no physical "
                            "reason and make the component labels unstable")
    p_tod.add_argument("--teff1", type=float,
                       help="T_eff [K] if --template1 is downloaded")
    p_tod.add_argument("--teff2", type=float,
                       help="T_eff [K] if --template2 is downloaded")
    p_tod.add_argument("--logg", type=float, default=4.5)
    p_tod.add_argument("--feh", type=float, default=0.0)
    p_tod.add_argument("--vsini1", type=float,
                       help="Rotational broadening of template 1 [km/s]")
    p_tod.add_argument("--vsini2", type=float,
                       help="Rotational broadening of template 2 [km/s]")
    p_tod.add_argument("--epsilon", type=float, default=0.6,
                       help="Limb darkening for the rotational profile")
    p_tod.add_argument("--dv", type=float, default=1.0,
                       help="km/s per pixel of the log-lambda grid "
                            "(default 1.0). TODCOR works in pixel lags, so "
                            "this sets the velocity resolution")
    p_tod.add_argument("--max-rv", type=float, default=200.0,
                       help="Half-width of the velocity search [km/s] "
                            "(default 200)")
    p_tod.add_argument("--alpha", type=float,
                       help="Fix the light ratio instead of fitting it. "
                            "A fitted alpha far from the expected value is "
                            "the clearest sign of a wrong template")
    # dest is NOT min_sep: that name already belongs to the BF component
    # search (default 30 km/s) and make_args() hands the same namespace to
    # every method, so a shared name silently applies BF's 30 km/s here.
    p_tod.add_argument("--min-sep", dest="todcor_min_sep", metavar="MIN_SEP",
                       type=float, default=0.0,
                       help="Exclude |v1 - v2| < this [km/s] from the peak "
                            "search. The v1 == v2 ridge is always excluded "
                            "(two components at one velocity are "
                            "indistinguishable from one star, so the flux "
                            "ratio there is not measurable); raise this to "
                            "also skip the ill-conditioned band around it")
    p_tod.add_argument("--wave-min", type=float,
                       help="Minimum wavelength to use [nm]")
    p_tod.add_argument("--wave-max", type=float,
                       help="Maximum wavelength to use [nm]")
    p_tod.add_argument("--spectrograph",
                       help="Built-in instrument name (sets R)")
    p_tod.add_argument("--resolution", type=float,
                       help="Resolving power R; degrades both templates")
    p_tod.add_argument("--no-clean", action="store_true",
                       help="Skip the per-order spike rejection and "
                            "continuum re-flattening applied to the observed "
                            "spectrum before correlating. Only useful to see "
                            "what that step is doing - without it, pipeline "
                            "leftovers dominate the correlation")
    p_tod.add_argument("--simulate", action="store_true",
                       help="Torres-style self-calibration: build NOISELESS "
                            "composites from the two templates at known "
                            "velocities and report the bias TODCOR shows on "
                            "them. No observed spectrum needed - run this "
                            "first to check a template pair")
    p_tod.add_argument("--sim-sep", nargs="+", type=float,
                       help="Separations [km/s] to test with --simulate")
    p_tod.add_argument("--object", help="Star name (SIMBAD) for the "
                                        "barycentric correction")
    p_tod.add_argument("--ra", help="RA for the barycentric correction")
    p_tod.add_argument("--dec", help="Dec for the barycentric correction")
    p_tod.add_argument("--obstime", help="Observation time: ISOT, BJD "
                                         "number or DD/MM/YY")
    p_tod.add_argument("--site", default=None,
                       help="Observatory. " + AUTO_SITE_HELP)
    p_tod.add_argument("--no-bary", action="store_true",
                       help="Compute the barycentric correction but do NOT "
                            "apply it")
    p_tod.add_argument("--force-bary", action="store_true",
                       help="Apply the correction even when the header says "
                            "the data are already barycentric")
    p_tod.add_argument("--t0", type=float, help="Ephemeris T0 [BJD]")
    p_tod.add_argument("--period", type=float, help="Orbital period [d]")
    p_tod.add_argument("--varastro", action="store_true",
                       help="Fetch T0/P from VarAstro using --object")
    p_tod.add_argument("--t0-to-bjd", action="store_true",
                       help="Convert the ephemeris T0 from HJD to BJD_TDB")
    p_tod.add_argument("--plot", help="Figure PNG name")
    p_tod.add_argument("--output", help="Result text file name")
    p_tod.add_argument("--dpi", type=int, help="PNG resolution")
    p_tod.set_defaults(func=cmd_todcor)

    p_bf = sub.add_parser("bf", help="Broadening Function (thesis method, SVD)")
    add_common_args(p_bf)
    p_bf.add_argument("--vel-range", type=float, default=500.0,
                      help="BF window half-width [km/s]")
    p_bf.add_argument("--dv", type=float,
                      help="Velocity step [km/s] (default: the "
                           "instrumental FWHM / 4 when --spectrograph or "
                           "--resolution is given, else the reference "
                           "stepV 5 km/s; never finer than the data "
                           "pixel. Per-bin BF amplitudes scale with this "
                           "step. Setting it explicitly also switches off "
                           "the automatic refinement of an undersampled "
                           "component)")
    p_bf.add_argument("--svd-rcond", type=float, default=None,
                      help="Relative SVD cutoff; default: automatic - "
                           "the smallest cutoff (starting at 0, no "
                           "truncation) that keeps the BF wing noise "
                           "below a tenth of the peak")
    p_bf.add_argument("--smooth", type=float,
                      help="BF smoothing FWHM [km/s] (default: the "
                           "instrumental FWHM c/R when a resolution/"
                           "spectrograph is given — saphires practice — "
                           "else sigma = 2 bins, as the reference)")
    p_bf.add_argument("--components", type=int, default=1,
                      choices=[1, 2, 3],
                      help="Number of stellar components to fit: SB1=1, "
                           "SB2=2, SB3/triple=3")
    p_bf.add_argument("--min-sep", type=float, default=30.0,
                      help="Component-search scale [km/s]: the width a "
                           "residual structure must survive to seed a "
                           "new component (noise narrower than this is "
                           "ignored)")
    p_bf.add_argument("--profile", choices=["gauss", "rot"],
                      default="gauss",
                      help="Component profile fitted to the BF: 'gauss' "
                           "(default) or 'rot' — Gray (1992) rotational "
                           "profiles convolved with the instrumental "
                           "Gaussian (give --resolution), the DDO-series "
                           "practice for contact binaries; the width is "
                           "then vsini and blended nearly-touching "
                           "profiles are fit without the Gaussian's "
                           "blending bias")
    p_bf.add_argument("--guess", type=float, nargs="+",
                      help="Initial RV guesses [km/s], one per component "
                           "(BF-rvplotter gausspars practice); also fixes "
                           "the component labels: C1 = first guess, "
                           "C2 = second, ...")
    p_bf.add_argument("--guess-amp", type=float, nargs="+",
                      help="Initial amplitude guesses (BF peak heights "
                           "read off the plot), one per component, "
                           "paired with --guess: run once without "
                           "guesses, inspect the smoothed BF, then rerun "
                           "with the visual estimates — the reference-"
                           "notebook workflow")
    p_bf.add_argument("--guess-sigma", type=float, nargs="+",
                      help="Initial width guesses [km/s], one per "
                           "component, paired with --guess/--guess-amp")
    p_bf.add_argument("--gamma", type=float,
                      help="Systemic velocity [km/s] for the phase-based "
                           "component identification (default: estimated "
                           "from the BF-area-weighted mean of the pair)")
    p_bf.add_argument("--degrade-template", action="store_true",
                      help="Also degrade the template to the resolution R "
                           "before the BF (only for ultra-sharp synthetic "
                           "templates; NEVER with an observed template - "
                           "in the BF flow R normally only sets the BF "
                           "smoothing, the saphires practice, since "
                           "degrading the template too would count the "
                           "instrumental profile twice)")
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
                         help="Minimum wavelength to use [nm]")
    p_batch.add_argument("--wave-max", type=float,
                         help="Maximum wavelength to use [nm]")
    p_batch.add_argument("--spectrograph",
                         help="Built-in instrument name (sets R)")
    p_batch.add_argument("--resolution", type=float,
                         help="Resolving power R (manual, overrides "
                              "--spectrograph)")
    p_batch.add_argument("--vsini", type=float,
                         help="Template rotational broadening vsini [km/s]")
    p_batch.add_argument("--epsilon", type=float, default=0.6,
                         help="Limb-darkening coefficient (default 0.6)")
    p_batch.add_argument("--normalize", action="store_true",
                         help="Continuum-normalize each raw spectrum first")
    p_batch.add_argument("--poly-order", type=int, default=3,
                         help="Continuum polynomial order (with --normalize)")
    p_batch.add_argument("--iterations", type=int, default=20,
                         help="Clipping iterations (with --normalize)")
    p_batch.add_argument("--window", type=float, default=20.0,
                         help="Continuum window width [nm] (with "
                              "--normalize); wide segments are normalized "
                              "piecewise (default 20 nm = 200 A)")
    p_batch.add_argument("--fit-method", default="poly",
                         choices=["poly", "bspline"],
                         help="Continuum model with --normalize: windowed "
                              "polynomial (default) or cubic B-spline")
    p_batch.add_argument("--object", help="Star name for SIMBAD coordinates")
    p_batch.add_argument("--ra", help="RA [deg or sexagesimal hours]")
    p_batch.add_argument("--dec", help="Dec [deg or sexagesimal]")
    p_batch.add_argument("--obstime",
                         help="Observation time override, ISOT or BJD "
                              "(default: per-file FITS header DATE-OBS)")
    p_batch.add_argument("--bjd", type=float, nargs="+",
                         help="BJD of each spectrum, one value per file in "
                              "sorted order (as the bjd list of the "
                              "reference document)")
    p_batch.add_argument("--site", default=None,
                         help="Observatory: built-in name, 'lonE,lat,alt' "
                              "string, or astropy site name "
                              "(see ccf --site). " + AUTO_SITE_HELP)
    p_batch.add_argument("--no-bary", action="store_true",
                         help="Compute the barycentric correction but do "
                              "NOT apply it to the RVs: the value is kept "
                              "in the RV curve file for information only")
    p_batch.add_argument("--force-bary", action="store_true",
                         help="Apply the barycentric correction even when "
                              "the file headers say the wavelengths are "
                              "already barycentric (the header check "
                              "otherwise stops the run to prevent a "
                              "double correction)")
    p_batch.add_argument("--vel-range", type=float, default=500.0,
                         help="BF window half-width [km/s]")
    p_batch.add_argument("--dv", type=float,
                         help="BF velocity step [km/s] (default: the "
                              "instrumental FWHM / 4 when the resolution "
                              "is known, else the reference stepV 5 km/s; "
                              "never finer than the data pixel. Setting it "
                              "explicitly also switches off the automatic "
                              "refinement of an undersampled component)")
    p_batch.add_argument("--svd-rcond", type=float, default=None,
                         help="BF relative SVD cutoff (default: "
                              "automatic)")
    p_batch.add_argument("--smooth", type=float,
                         help="BF smoothing FWHM [km/s] (default: the "
                              "instrumental FWHM c/R when a resolution "
                              "is given, else sigma = 2 bins)")
    p_batch.add_argument("--components", type=int, default=2,
                         choices=[1, 2, 3],
                         help="Components to fit (default: 2)")
    p_batch.add_argument("--min-sep", type=float, default=30.0,
                         help="Component-search scale [km/s] (see bf "
                              "--min-sep)")
    p_batch.add_argument("--profile", choices=["gauss", "rot"],
                         default="gauss",
                         help="BF component profile (see bf --profile)")
    p_batch.add_argument("--guess", type=float, nargs="+",
                         help="Initial RV guesses [km/s], one per "
                              "component; also fixes the component labels")
    p_batch.add_argument("--guess-amp", type=float, nargs="+",
                         help="Initial amplitude guesses, one per "
                              "component, paired with --guess")
    p_batch.add_argument("--guess-sigma", type=float, nargs="+",
                         help="Initial width guesses [km/s], one per "
                              "component, paired with --guess")
    p_batch.add_argument("--gamma", type=float,
                         help="Systemic velocity [km/s] for the "
                              "phase-based component identification")
    p_batch.add_argument("--degrade-template", action="store_true",
                         help="Also degrade the template to R before the "
                              "BF (synthetic templates only; in BF mode R "
                              "normally only sets the BF smoothing)")
    p_batch.add_argument("--rv-min", type=float, default=None,
                         help="CCF scan lower limit [km/s] (default: "
                              "automatic two-stage search, see ccf "
                              "--rv-min)")
    p_batch.add_argument("--rv-max", type=float, default=None,
                         help="CCF scan upper limit [km/s] (default: "
                              "automatic two-stage search)")
    p_batch.add_argument("--rv-step", type=float, default=0.5,
                         help="Fine CCF step [km/s]")
    p_batch.add_argument("--ccf-mode", choices=["mask", "template"],
                         default="mask",
                         help="CCF construction (see ccf --ccf-mode)")
    p_batch.add_argument("--mask-depth", type=float, default=0.05,
                         help="Minimum template line depth for a mask "
                              "entry (mask mode)")
    p_batch.add_argument("--keep-balmer", action="store_true",
                         help="Keep the hydrogen lines in the CCF mask")
    p_batch.add_argument("--t0", type=float,
                         help="Ephemeris T0 [BJD] for phase-folding the plot")
    p_batch.add_argument("--period", type=float,
                         help="Orbital period [days] for phase-folding")
    p_batch.add_argument("--varastro", action="store_true",
                         help="Fetch T0/P from VarAstro using --object "
                              "(T0 in HJD, P in days)")
    p_batch.add_argument("--t0-to-bjd", action="store_true",
                         help="Convert the ephemeris T0 from HJD to "
                              "BJD_TDB before the analysis "
                              "(needs target coordinates)")
    p_batch.add_argument("--output", help="RV curve file "
                                          "(default: result_RV_curve.txt)")
    p_batch.add_argument("--plot", help="RV curve figure "
                                        "(default: result_RV_curve.png)")
    p_batch.add_argument("--dpi", type=float,
                         help="PNG figure resolution in dots per inch "
                              "(default 600)")
    p_batch.set_defaults(func=cmd_batch)

    p_head = sub.add_parser("header",
                            help="Show the header of a spectrum: FITS "
                                 "(summary + all primary cards) or "
                                 "normalized ASCII (summary + comment "
                                 "header), incl. the bary-frame verdict")
    p_head.add_argument("--spectrum", required=True, help="FITS spectrum file")
    p_head.add_argument("--site", default=None,
                        help="Observatory for the BJD computation. "
                             + AUTO_SITE_HELP)
    p_head.add_argument("--output", help="Save the header to a text file")
    p_head.set_defaults(func=cmd_header)

    p_bchk = sub.add_parser("barycheck",
                            help="Check from the file headers (FITS, or "
                                 "normalized ASCII) whether the "
                                 "wavelengths are already barycentric "
                                 "(SPECSYS / DRS BERV / BARYCORR "
                                 "cards); advises whether to use "
                                 "--no-bary")
    p_bchk.add_argument("--spectra", nargs="+", required=True,
                        help="Spectrum files or glob patterns "
                             "(e.g. 'data/ADP.*.fits')")
    p_bchk.add_argument("--from-fits", nargs="+",
                        help="Original FITS files or glob patterns: an "
                             "ASCII spectrum in --spectra that has no "
                             "frame information gets the evidence of "
                             "its matching original (same file name, "
                             "'_norm' suffix ignored) written into its "
                             "comment header, in place - retrofits "
                             "files normalized before the evidence was "
                             "propagated")
    p_bchk.add_argument("--output", help="Save the report to a text file")
    p_bchk.set_defaults(func=cmd_barycheck)

    p_fadp = sub.add_parser(
        "fetch-adp",
        help="Download the reduced (ADP phase-3) counterparts of raw ESO "
             "frames (HARPS.*/FEROS.*/... files) from the ESO archive")
    p_fadp.add_argument("--spectra", nargs="+", required=True,
                        help="Raw FITS files or glob patterns "
                             "(e.g. 'data/FEROS.*.fits'); ADP.* files "
                             "are skipped")
    p_fadp.add_argument("--dest",
                        help="Destination directory (default: the raw "
                             "file's own directory)")
    p_fadp.add_argument("--radius", type=float, default=30.0,
                        help="Position match radius [arcsec] around the "
                             "raw frame's RA/DEC (default 30)")
    p_fadp.add_argument("--tmatch", type=float, default=120.0,
                        help="Time match tolerance [s] between the raw "
                             "MJD-OBS and the product start time t_min "
                             "(default 120)")
    p_fadp.add_argument("--dry-run", action="store_true",
                        help="Only report what would be downloaded")
    p_fadp.set_defaults(func=cmd_fetch_adp)

    p_norm = sub.add_parser("normalize",
                            help="Continuum-normalize a raw spectrum "
                                 "(iterative polynomial fitting)")
    p_norm.add_argument("--spectrum", required=True, help="Raw spectrum file")
    p_norm.add_argument("--format", default="auto",
                        choices=["auto", "s2d", "text"])
    p_norm.add_argument("--poly-order", type=int, default=3,
                        help="Continuum polynomial order (default tuned "
                             "for the 500-550 nm RV region)")
    p_norm.add_argument("--iterations", type=int, default=20,
                        help="Sigma-clipping iterations (default 20)")
    p_norm.add_argument("--low-clip", type=float, default=1.0,
                        help="Rejection threshold below the fit [sigma]")
    p_norm.add_argument("--high-clip", type=float, default=4.0,
                        help="Rejection threshold above the fit [sigma]")
    p_norm.add_argument("--window", type=float, default=20.0,
                        help="Continuum window width [nm]; segments wider "
                             "than 1.5x this are normalized piecewise with "
                             "blended overlapping windows (saphires "
                             "cont_norm w_width practice, default 20 nm = "
                             "200 A). 0 = single polynomial over the whole "
                             "segment. For --fit-method bspline this is "
                             "the knot spacing")
    p_norm.add_argument("--fit-method", default="poly",
                        choices=["poly", "bspline"],
                        help="Continuum model: 'poly' (default) - "
                             "windowed polynomial; 'bspline' - cubic "
                             "B-spline with knots every --window "
                             "(saphires cont_norm model). Both clip "
                             "iteratively onto the continuum regions")
    p_norm.add_argument("--template",
                        help="Synthetic spectrum to overlay for comparison")
    p_norm.add_argument("--spectrograph",
                        help="Built-in instrument name (sets R)")
    p_norm.add_argument("--resolution", type=float,
                        help="Resolving power R (manual, overrides "
                             "--spectrograph)")
    p_norm.add_argument("--vsini", type=float,
                        help="Template rotational broadening vsini [km/s]")
    p_norm.add_argument("--epsilon", type=float, default=0.6,
                        help="Limb-darkening coefficient (default 0.6)")
    p_norm.add_argument("--teff", type=float, help="Template T_eff [K] (expecto)")
    p_norm.add_argument("--logg", type=float, default=4.5, help="Template log g")
    p_norm.add_argument("--feh", type=float, default=0.0, help="Template [Fe/H]")
    p_norm.add_argument("--object", help="Star name, written into the "
                                         "output header")
    p_norm.add_argument("--ra", help="RA (deg or sexagesimal hours), "
                                     "written into the output header for "
                                     "the later barycentric correction")
    p_norm.add_argument("--dec", help="Dec (deg or sexagesimal), written "
                                      "into the output header")
    p_norm.add_argument("--obstime", help="Observation time (ISOT/BJD/"
                                          "DD-MM-YY), written into the "
                                          "output header")
    p_norm.add_argument("--output", help="Output ASCII file "
                                         "(default: <input>_norm.txt)")
    p_norm.add_argument("--plot", help="Figure PNG file name "
                                       "(default: result_normalization.png)")
    p_norm.add_argument("--dpi", type=float,
                        help="PNG figure resolution in dots per inch "
                             "(default 600)")
    p_norm.set_defaults(func=cmd_normalize)

    p_demo = sub.add_parser("demo",
                            help="CCF vs BF comparison on synthetic SB2 data")
    p_demo.add_argument("--plot", help="Comparison figure PNG file name")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if getattr(args, "dpi", None):
        global FIGURE_DPI
        FIGURE_DPI = float(args.dpi)
    try:
        args.func(args)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")


def main():
    """Run the tool with the console output captured in the run log."""
    log_path, log_enabled = pop_log_args(sys.argv)
    state = start_logging(log_path, log_enabled)
    status, exc_info = "finished", None
    try:
        run_cli()
    except SystemExit as exc:
        # sys.exit("message") is the tool's normal way of stopping on a
        # bad input, but the interpreter prints that message only after
        # this frame is gone — after the log is closed, so it would miss
        # the log and come out after the "Log written" line. Print it
        # here instead (through the tee, so it reaches both) and exit
        # with the status a string exit code would have given anyway.
        code = exc.code
        if isinstance(code, str):
            print(code, file=sys.stderr)
            status = "stopped with an error"
            raise SystemExit(1) from None
        if code not in (None, 0):
            status = f"stopped (exit code {code})"
        raise
    except BaseException:
        status = "failed"
        exc_info = sys.exc_info()
        raise
    finally:
        finish_logging(state, status, exc_info)


if __name__ == "__main__":
    main()
