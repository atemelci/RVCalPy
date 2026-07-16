# RVCalPy — Radial Velocity from Stellar Spectra

RVCalPy is a single-file Python tool, [`rv_analysis.py`](rv_analysis.py), that
measures the radial velocity (RV) of a star by comparing an observed spectrum
with a template spectrum. Two independent methods are available:

- **CCF (Cross-Correlation Function)** — the template is Doppler-shifted over
  a grid of trial velocities and cross-correlated with the observed spectrum;
  a Gaussian fit to the correlation peak gives the RV. Fast and robust for
  single stars (SB1).
- **BF (Broadening Function, Rucinski 1992/2002)** — the observed spectrum is
  modelled as the convolution of the template with a velocity-space profile,
  solved linearly via Singular Value Decomposition. Because the method is
  linear, the components of a binary (SB2) or triple (SB3) appear as separate
  sharp peaks, so each component's RV is measured independently. Recommended
  for binary and multiple systems. The spectra enter the SVD as line depths
  (1 − flux), following Rucinski: the continuum then drops out of the linear
  problem, the BF baseline sits at zero and the wings decay cleanly. The
  velocity step defaults to the reference stepV = 5 km/s (per-bin BF
  amplitudes scale with the step, so this keeps them on the
  reference-implementation scale; `--dv` overrides).

Several practices are adopted from the
[saphires](https://github.com/tofflemire/saphires) package:

- multi-order (echelle) spectra get **one BF per order**, combined with
  1/σ² weights from each order's BF sidebands, so noisy orders are
  down-weighted automatically;
- the BF is smoothed by the **instrumental FWHM c/R** when a
  resolution/spectrograph is given (R does **not** degrade the template in
  the BF flow — that would count the instrumental profile twice; use
  `--degrade-template` only for ultra-sharp synthetic templates);
- multi-component fits are sums of Gaussians on a **shared constant
  baseline**, and the noisy BF edges are trimmed before fitting;
- unphysical pixels (normalized flux > 1.2 or < 0) are interpolated over
  before the SVD;
- the barycentric correction is applied with the relativistic cross term,
  `RV = RV_measured + v_bary + RV_measured·v_bary/c` (Wright & Eastman 2014).

## Installation

Python 3.8+ is required.

```bash
git clone https://github.com/atemelci/RVCalPy.git
cd RVCalPy
pip install -r requirements.txt
```

Optional extras:

- `pip install expecto` — automatic download of PHOENIX model templates when
  no template file is given.
- `tkinter` — needed only for the graphical interface. It ships with most
  Python installations; on Debian/Ubuntu install it with
  `sudo apt install python3-tk`. Without it the tool falls back to a
  terminal wizard with the same steps.

## Usage

### Interactive mode (recommended)

```bash
python rv_analysis.py
```

A small window opens: resolve the target via SIMBAD (or enter RA/Dec), pick
the normalized observed spectrum and the template, choose **CCF** or **BF**,
and press **Run**. The fit is shown in the window with one dashed profile per
component, labelled `C1: amp = ..., RV = ... km/s` — `amp` is the peak height
read off the BF axis (the profile integrals/areas and the light ratios stay
in the text summary). Nothing is written to disk until you press **Save**
next to Run.

For multi-component fits the **Amp/Sigma guesses** fields enable the
reference-notebook workflow: run once, read each component's approximate
peak height (y-axis), RV (x-axis) and width off the smoothed BF, type them
in, and rerun — the fit then converges from your visual estimates instead
of the automatic search.

Other GUI features:

- **Normalize raw...** — continuum-normalize a raw spectrum first; choose the
  continuum model (**Polynomial** in blended windows, or a cubic
  **B-spline** with knots every window — the polynomial order applies to the
  polynomial model only, the clipping iterations to both).
- **Header** — inspect the FITS header (object, DATE-OBS, RA/Dec, exposure
  time, any RV-related cards, mid-exposure BJD_TDB) and auto-fill the target
  fields.
- **VarAstro / HJD → BJD** — fetch the eclipsing-binary ephemeris (T0, P)
  from var.astro.cz and convert T0 to BJD_TDB for the orbital phase.
- The barycentric correction is **opt-in**: v_bary is always computed and
  reported, but only added to the RVs when the checkbox is ticked.

### Command line

```bash
# CCF — single star (SB1)
python rv_analysis.py ccf --spectrum spectrum.txt --template template.prf

# BF — binary star (SB2), two components (automatic component search)
python rv_analysis.py bf --spectrum spectrum.txt --template template.prf \
    --components 2

# BF — triple system, starting from your own estimates read off the BF plot
# (run the automatic mode first, inspect result_BF.png, then:)
python rv_analysis.py bf --spectrum spectrum.txt --template template.prf \
    --components 3 --guess 15 -250 0 --guess-amp 0.02 0.002 0.005 \
    --guess-sigma 50 25 10

# continuum-normalize a raw spectrum first
python rv_analysis.py normalize --spectrum raw.fits --output spectrum_norm.txt

# a whole time series into an RV curve
python rv_analysis.py batch --spectra 'data/*.fits' --normalize \
    --template template.prf --components 2 --object "KX Aqr" --varastro
```

Try it with the bundled example data (true values: RV1 = −70 km/s,
RV2 = +90 km/s):

```bash
python rv_analysis.py bf --spectrum examples/example_observed.obs \
    --template examples/example_synthetic.prf \
    --vel-range 300 --components 2 --smooth 10 --svd-rcond 5e-4
```

A self-test on synthetic data (no input files needed):

```bash
python rv_analysis.py demo --plot demo.png
```

Run `python rv_analysis.py <command> --help` for the full list of options
(`ccf`, `bf`, `batch`, `normalize`, `header`, `demo`, `gui`).

### Inputs

- **Observed spectrum** — ASCII tables (`.txt`, `.dat`, `.obs`, ...; first two
  numeric columns are wavelength — Å or nm, auto-detected — and flux) and the
  common FITS layouts:
  - ESPRESSO S2D (flux ext 1, wavelength ext 4),
  - phase-3 style binary tables with WAVE/LAMBDA and FLUX columns
    (FEROS, HARPS, UVES, ...),
  - IRAF echelle **multispec** images (`CTYPE1 = 'MULTISPE'`, WAT2 cards;
    linear, log and Chebyshev/Legendre dispersion solutions, per-order
    Doppler factor applied),
  - 1D images with a linear wavelength WCS (CRVAL1/CDELT1), including
    multi-extension products with one order per extension (e.g. SALT HRS).
- **Template spectrum** — an ASCII/`.prf` synthetic spectrum, a FITS spectrum
  (an observed standard used as template), or a PHOENIX model downloaded
  automatically via `--teff/--logg/--feh` (needs `expecto`). Templates must
  be continuum-normalized, like the observed spectrum.
- **Observation time** — ISOT (`2024-12-03T02:30:00`), a JD/BJD number, or
  the old FITS date convention `DD/MM/YY` (combined automatically with the
  `UT`/`UTMIDDLE` header card when only a date is present). Note that
  SOPHIE/HARPS-style `s1d` products are already in the barycentric frame —
  do not apply the correction twice.

### Normalization

`normalize` (and `batch --normalize`) fits the continuum with iterative
asymmetric sigma-clipping, so the fit converges onto the continuum regions.
Two continuum models are available (`--fit-method`):

- **poly** (default) — a low-order polynomial; segments wider than ~1.5× the
  window (default 20 nm) are fitted in 50%-overlapping windows blended with
  triangular weights;
- **bspline** — a cubic B-spline with interior knots every window
  (the saphires `cont_norm` model).

### BF components: automatic search and user estimates

Without guesses the components are found automatically by **iterative
fit-and-subtract**: one Gaussian is fitted first, the next component is
seeded on the largest coherent structure left in the residual (smoothed on
the `--min-sep` velocity scale, so single-bin noise spikes are ignored),
and the model is refitted at each step. This finds a faint secondary far
from the primary — whose own shoulder wiggles would win a naive
highest-local-maxima search — and a narrow tertiary superposed on the broad
primary at nearly the same RV.

When the automatic search is not enough, give the fit your own starting
point (the reference-notebook workflow): run once, inspect the smoothed BF
in `result_BF.png`, then rerun with per-component estimates —
`--guess RV1 RV2 ...` (x-axis positions), `--guess-amp A1 A2 ...` (peak
heights on the y-axis) and optionally `--guess-sigma S1 S2 ...` (widths in
km/s). With amplitudes given, `curve_fit` starts directly from your
[amp, RV, sigma] values, exactly like the reference notebook.

### BF component labels

By default the fitted components are labelled by BF area: **C1 is the
component with the largest light contribution** (profile integral). Two
overrides are available:

- `--guess RV1 RV2 ...` pins the labels to your initial guesses
  (C1 = first guess, C2 = second, ...);
- with an ephemeris (`--t0/--period`, or `--varastro`) the labels follow the
  orbital phase (C1 = primary; with three components the narrowest peak is
  listed last as the tertiary).

The light ratios reported with multi-component fits come from the fitted
profile integrals (the `area` values in the result summary; the figure
legend shows each component's `amp` — the real peak height on the BF
axis — and RV).

### Outputs

| File | Content |
|---|---|
| `result_CCF.txt` / `result_BF.txt` | RVs with errors, uncorrected and barycentric-corrected, amplitudes, areas, widths, light ratios |
| `result_CCF.png` / `result_BF.png` | Fit figure (BF: one dashed profile per component with `amp` and RV in the legend) |
| `result_CCF_linecheck.png` / `result_BF_linecheck.png` | Model reliability check at strong diagnostic lines |
| `result_RV_curve.txt` / `.png` | `batch` mode: BJD, phase and per-component RVs for the whole series |
| `result_BF_profiles.png` | `batch` mode: all BF profiles stacked by orbital phase |
