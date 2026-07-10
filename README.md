# RVCalPy — Radial Velocity from Stellar Spectra

RVCalPy is a single-file Python tool, [`rv_analysis.py`](rv_analysis.py), that
measures the radial velocity (RV) of a star by comparing an observed spectrum
with a synthetic (template) spectrum. Two independent methods are available:

- **CCF (Cross-Correlation Function)** — the template is Doppler-shifted over
  a grid of trial velocities and cross-correlated with the observed spectrum;
  a Gaussian fit to the correlation peak gives the RV. Fast and robust for
  single stars (SB1).
- **BF (Broadening Function, Rucinski 1992/2002)** — the observed spectrum is
  modelled as the convolution of the template with a velocity-space profile,
  solved linearly via Singular Value Decomposition. Because the method is
  linear, the components of a binary (SB2) appear as separate sharp peaks, so
  each component's RV is measured independently. Recommended for binary and
  multiple systems.

Both methods report `RV ± error` and write the result to disk, with and
without the barycentric velocity correction.

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

A small window opens: select the normalized observed spectrum and the
synthetic template, choose **CCF** or **BF**, and press **Run**. The fit is
shown in the window and the result files are saved to the working directory.

### Command line

```bash
# CCF — single star (SB1)
python rv_analysis.py ccf --spectrum spectrum.txt --template template.prf

# BF — binary star (SB2), two components
python rv_analysis.py bf --spectrum spectrum.txt --template template.prf \
    --components 2
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

### Inputs

- **Observed spectrum**: any ASCII table (`.txt`, `.dat`, `.obs`, ...; first
  two numeric columns are wavelength — Å or nm, auto-detected — and
  normalized flux) or a supported FITS layout.
- **Template spectrum**: an ASCII/`.prf` synthetic spectrum, or a PHOENIX
  model downloaded automatically via `--teff/--logg/--feh` (needs `expecto`).

### Outputs

| File | Content |
|---|---|
| `result_CCF.txt` / `result_BF.txt` | RVs with errors, both uncorrected and barycentric-corrected |
| `result_CCF.png` / `result_BF.png` | Fit figure (data + Gaussian model) |
| `result_CCF_linecheck.png` / `result_BF_linecheck.png` | Model reliability check at strong diagnostic lines |

Run `python rv_analysis.py --help` (or `ccf --help` / `bf --help`) for the
full list of options.
