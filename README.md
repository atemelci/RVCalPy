# RVCalPy — Easy Way of Radial Velocity Calculation

RVCalPy is a single-file Python tool, [`rv_analysis.py`](rv_analysis.py), that
measures the radial velocity (RV) of a star by comparing an observed spectrum
with a template spectrum, by two independent methods — a weighted line-mask
**CCF** and the SVD **Broadening Function (BF)** — plus **TODCOR** for binaries
whose two peaks have merged. For a double-lined binary it carries the
velocities on to the **Wilson plot** (mass ratio and systemic velocity with no
orbit assumed) and to an **SB2 orbit fit**.

**Usage is documented in [`Guide.txt`](Guide.txt)**: preparing data, the
interactive mode, every command-line command, the barycentric-frame rules, the
BF sampling guards, the outputs, and troubleshooting. This file covers the
benchmark, setup, license, references and acknowledgements.

## Setup

Python 3.8+ is required.

```bash
git clone https://github.com/atemelci/RVCalPy.git
cd RVCalPy
pip install -r requirements.txt
```

Required packages (`requirements.txt`):

| Package | Used for |
|---|---|
| `numpy` | Arrays, the SVD linear algebra |
| `scipy` | `curve_fit` profile fitting, filtering, B-spline continua |
| `matplotlib` | All figures |
| `astropy` | FITS I/O, time scales (BJD_TDB), coordinates, the barycentric correction |
| `PyAstronomy` | `pyasl.SVD` broadening function, rotational broadening |

Optional extras:

- `pip install expecto` — automatic download of PHOENIX model templates when
  no template file is given (`--teff/--logg/--feh`).
- `tkinter` — needed only for the graphical interface. It ships with most
  Python installations; on Debian/Ubuntu install it with
  `sudo apt install python3-tk`. Without it the tool falls back to a
  terminal wizard with the same steps.

Internet access is needed only for the optional online services: SIMBAD name
resolution, the VarAstro ephemeris lookup, PHOENIX template downloads and the
ESO archive queries of `fetch-adp`. Everything else runs offline.

No installation step is required beyond the dependencies — `rv_analysis.py`
is a single self-contained script and can simply be copied next to your data.

## License

RVCalPy is released under the **GNU General Public License, version 3
(GPL-3.0)**. The full text is in [`LICENSE`](LICENSE).

In short: you are free to use, study, modify and redistribute this program,
provided that derivative works are distributed under the same license and
with their source available. The program comes with **no warranty**, to the
extent permitted by law.

The bundled example data (`examples/`) are synthetic spectra generated for
this repository from published orbital elements and are covered by the same
license.

## References

### Methods implemented

| Topic | Reference |
|---|---|
| Broadening Function, SVD formulation | Rucinski, S. M. 1992, AJ 104, 1968; Rucinski, S. M. 2002 |
| Cross-correlation with a weighted line mask | Baranne, A., et al. 1996, A&AS 119, 373; Pepe, F., et al. 2002, A&A 388, 632 |
| Mask construction and the contrast (peak) fit | Pino, L., et al. 2018, A&A 619, A3 |
| Two-stage RV search, S/N and variability warnings | Katz, D., et al. 2025, A&A 704, A294 |
| Two-dimensional correlation (`todcor`) | Zucker, S., & Mazeh, T. 1994, ApJ 420, 806; Mazeh, T., & Zucker, S. 1994, Ap&SS 212, 349; review: Zucker, S. 2012, IAU Symp. 282, 371 |
| Mass ratio from the RV1-vs-RV2 line (`wilson`) | Wilson, O. C. 1941, ApJ 93, 29 |
| The Wilson step ahead of the orbit; circular SB2 orbit fit (`orbit`) | Kovalev, M., & Straumit, I. 2022, MNRAS 510, 1515; Nachmani, G., Faigler, S., & Mazeh, T. 2026, MNRAS 546 (MESS) |
| Barycentric correction with the relativistic cross term | Wright, J. T., & Eastman, J. D. 2014, PASP 126, 838 |
| Rotational line profile (`--profile rot`) | Gray, D. F. 1992, *The Observation and Analysis of Stellar Photospheres* |
| FEROS/MIDAS barycentric accuracy caveat | Müller, A., et al. 2013, A&A 556, A3 |
| Example system V563 Lyr (orbital elements) | Alvarez et al. 2022, RMxAA 58, 223 |
| Benchmark: published velocities of LL Aqr | Graczyk, D., Smolec, R., Gazeas, K., et al. 2016, A&A 594, A92 |
| Benchmark: published velocities of AK For | Hełminiak, K. G., Graczyk, D., Konacki, M., et al. 2014, A&A 567, A64 |

### Software this tool builds on or follows

- [saphires](https://github.com/tofflemire/saphires) (B. Tofflemire) — the
  per-order BF weighting, the instrumental BF smoothing, the BF edge
  trimming, the cosmic-ray/unphysical-pixel handling, the `cont_norm`
  continuum model and the `brvc` barycentric convention are followed here;
  its [method notes](https://saphires.readthedocs.io/en/latest/intro.html)
  also set the Tonry & Davis (1979) overlap normalization used by `todcor`
  and the BF-vs-TODCOR guidance in
  the `todcor` section of [`Guide.txt`](Guide.txt).
- [Simchon/TODCOR](https://github.com/Simchon/TODCOR) (MIT) — the TODCOR
  engine, vendored unmodified at `vendor/todcor.py` with its provenance in
  `vendor/VERSION.todcor`; everything RVCalPy adds around it (the log-λ grid,
  the spectrum cleaning, the α and detection logic) lives in the wrapper.
- [PyAstronomy](https://github.com/sczesla/PyAstronomy) — `pyasl.SVD`, the
  SVD broadening-function solver used underneath, and the rotational
  broadening routines.
- [astropy](https://www.astropy.org/) — FITS, `Time`, `SkyCoord` and
  `radial_velocity_correction`.
- [expecto](https://github.com/bmorris3/expecto) — optional PHOENIX model
  template downloads.
- The full-spectrum cross-covariance mode (`--ccf-mode template`) follows the
  *Between the Lines* 2024 workshop material (E. Sedaghati).

### Data services

- [SIMBAD](https://simbad.cds.unistra.fr/simbad/) (CDS, Strasbourg) — target
  coordinates and cross-identifiers.
- [VarAstro](https://var.astro.cz/) — eclipsing-binary ephemerides (T0, P).
- [ESO Science Archive](https://archive.eso.org/) — the TAP/ObsCore service
  and the dataportal used by `fetch-adp`.

If you use RVCalPy in published work, please cite the method papers above
alongside this repository.

## Benchmark

RVCalPy's velocities were checked night by night against the published
HARPS/FEROS measurements of two double-lined eclipsing binaries, LL Aqr and
AK For. What was compared, how the statistics are defined, the night-by-night
residuals and the phased velocity figures are in the report:

**[RV benchmark report (PDF)](https://github.com/atemelci/RVCalPy/raw/HEAD/RV_benchmark_report.pdf)**

## Acknowledgements

I am grateful to **Tobias Cornelius Hinse**, **Mehmet Alperen Kul** and
**Otmar Stahl** for their guidance, discussions and feedback during the
development of this tool. Their input shaped the methods, the data handling
and the checks built into `rv_analysis.py`.

Thanks are due as well to the authors and maintainers of the open-source
packages and the archive and catalogue services listed under
[References](#references), without which this work would not be possible.
