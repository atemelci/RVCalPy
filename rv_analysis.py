#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rv_analysis.py — Tayfsal veriden Dikine Hız (Radial Velocity, RV) hesabı
=========================================================================

Bu script iki bağımsız yöntemi bir arada sunar:

1. CCF  (Cross-Correlation Function / Çapraz Korelasyon Fonksiyonu)
   Between the Lines 2024 çalıştayındaki (E. Sedaghati) yaklaşım:
   Gözlenen tayf ile Doppler kaydırılmış şablon (model) tayfın nokta
   çarpımı bir RV ızgarası üzerinde taranır; CCF tepe noktasına Gauss
   uyumlanarak RV elde edilir. Tek yıldızlar (SB1) için hızlı ve sağlamdır.

2. BF   (Broadening Function / Çizgi Genişleme Fonksiyonu, Rucinski 1992, 2002)
   MAK_Tez Bölüm 2.2 ve 3.2'de anlatılan yöntem:
   Gözlenen tayf, şablon tayfın hız uzayında bir genişleme fonksiyonu ile
   konvolüsyonu olarak modellenir:  S(v) = B(v) * T(v).
   B(v), Tekil Değer Ayrışımı (SVD) ile doğrusal olarak çözülür.
   Yöntem doğrusal olduğu için çift/çoklu sistemlerde (SB2) bileşenler
   birbirinden bağımsız, keskin tepeler olarak ayrışır; tepe merkezleri
   dikine hızları, genişlikleri (vsini) dönme hızlarını, alanları ise
   ışık katkılarını verir. BF profillerine tek ya da çift Gauss
   uyumlanarak her bileşenin RV'si ölçülür.

Desteklenen girdiler
--------------------
- ESPRESSO S2D FITS (çok basamaklı echelle: akı ext=1, dalgaboyu ext=4)
- İki/üç sütunlu metin dosyası: dalgaboyu[Å]  akı  [akı_hatası]
  (IRAF ile indirgenmiş, normalize edilmiş, birleştirilmiş tayf — tezdeki
   5000–5500 Å bölgesi gibi)
- Şablon: metin dosyası (dalgaboyu, akı) veya `expecto` kuruluysa
  PHOENIX modelinden otomatik indirme (--teff/--logg/--feh).

Kullanım örnekleri
------------------
# Sentetik veriyle kendi kendini test (veri gerekmez):
python rv_analysis.py demo

# Çalıştaydaki gibi ESPRESSO S2D tayfından CCF ile RV:
python rv_analysis.py ccf --spectrum ESPRESSO_S2D_BLAZE_A.fits --format s2d \
    --teff 6628 --logg 4.251 --feh 0.17 --rv-min -20 --rv-max 100 --rv-step 0.5

# Tezdeki gibi normalize tayftan BF ile SB2 çift sistemin iki bileşeninin RV'si:
python rv_analysis.py bf --spectrum tayf_5000_5500.txt --template sablon.txt \
    --vel-range 500 --components 2 --wave-min 5000 --wave-max 5500

Çıktılar: terminale RV ± hata, istenirse PNG grafik (--plot) ve
sonuç metin dosyası (--output).
"""

import argparse
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Işık hızı [km/s] — astropy varsa oradan, yoksa CODATA değeri
try:
    import astropy.constants as const
    C_KMS = const.c.value / 1000.0
except ImportError:
    C_KMS = 299792.458


# ----------------------------------------------------------------------
# Ortak yardımcı fonksiyonlar
# ----------------------------------------------------------------------

def doppler_shift(wavelength, rv_kms):
    """Durgun çerçevedeki dalgaboyunu RV [km/s] için kaydırır (çalıştay, hücre-3)."""
    return wavelength * (1.0 + rv_kms / C_KMS)


def gauss(x, amp, x0, sigma, offset):
    """Tek Gauss: a·exp(-(x-x0)²/2σ²) + y0"""
    return amp * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2)) + offset


def two_gauss(x, a1, x1, s1, a2, x2, s2, offset):
    """Çift Gauss (SB2 sistemlerde iki bileşen için)."""
    return (a1 * np.exp(-((x - x1) ** 2) / (2.0 * s1 ** 2))
            + a2 * np.exp(-((x - x2) ** 2) / (2.0 * s2 ** 2))
            + offset)


def read_spectrum(path, fmt="auto"):
    """Tayfı oku. Dönen değer: (dalgaboyu, akı) listesi — echelle ise basamak başına bir çift.

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

    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: en az iki sütun (dalgaboyu, akı) bekleniyor.")
    return [(data[:, 0], data[:, 1])]


def load_template(args):
    """Şablon tayfı yükle: dosyadan ya da expecto/PHOENIX'ten."""
    if args.template:
        data = np.loadtxt(args.template)
        return data[:, 0], data[:, 1]
    if args.teff is not None:
        try:
            from expecto import get_spectrum
        except ImportError:
            sys.exit("Şablon dosyası verilmedi ve 'expecto' kurulu değil.\n"
                     "Kurulum: pip install expecto  (veya --template dosya.txt kullanın)")
        tpl = get_spectrum(T_eff=args.teff, log_g=args.logg, Z=args.feh, cache=True)
        return tpl.wavelength.value, tpl.flux.value
    sys.exit("Şablon gerekli: --template DOSYA veya --teff/--logg/--feh verin.")


def barycentric_correction(ra_deg, dec_deg, obstime_isot, site):
    """Kütle merkezli (barycentric) hız düzeltmesi [km/s].

    Tezde IDL koduyla yapılan Güneş Sistemi kütle merkezine indirgeme adımının
    astropy karşılığı. Elde edilen değer ölçülen RV'ye EKLENİR:
        RV_bary = RV_ölçülen + v_bary
    """
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy.time import Time
    import astropy.units as u

    loc = EarthLocation.of_site(site)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    t = Time(obstime_isot, format="isot", scale="utc")
    return coord.radial_velocity_correction(obstime=t, location=loc).to(u.km / u.s).value


# ----------------------------------------------------------------------
# 1) CCF yöntemi (çalıştay defterinden derlendi)
# ----------------------------------------------------------------------

def calculate_ccf(spec_wl, spec_flux, tpl_wl, tpl_flux, rv_grid):
    """Tek bir (basamak) tayf için CCF.

    Her RV için şablon Doppler kaydırılır, verinin dalgaboyu ızgarasına
    yeniden örneklenir (np.interp) ve nokta çarpımı alınır — çalıştaydaki
    'resampling şart, yoksa nokta çarpımı anlamsız' notuna dikkat.

    Çalıştay kodundan tek fark: nokta çarpımından önce her iki sinyalin
    ortalaması çıkarılır. Süreklilik (continuum) katkısı kaydırmadan neredeyse
    bağımsız olduğu için çıkarılmazsa normalize tayflarda çizgi sinyalini
    bastırıp tepe konumunu kaydırabilir; ortalama çıkarınca CCF gerçek
    çapraz-kovaryans olur ve tepe, çizgilerin hizalandığı RV'de oluşur.
    """
    spec = spec_flux - np.mean(spec_flux)
    ccf = np.empty(rv_grid.size)
    for i, rv in enumerate(rv_grid):
        shifted_wl = doppler_shift(tpl_wl, rv)
        model = np.interp(spec_wl, shifted_wl, tpl_flux)
        ccf[i] = np.dot(spec, model - np.mean(model))
    return ccf


def run_ccf(spectrum_orders, tpl_wl, tpl_flux, rv_min, rv_max, rv_step):
    """CCF'i tüm echelle basamakları üzerinde çalıştırıp toplar; Gauss uyumlar.

    Dönen: dict(rv, rv_err, rv_grid, ccf_total, ccf_orders, popt)
    """
    rv_grid = np.arange(rv_min, rv_max + rv_step, rv_step)
    tpl_norm = tpl_flux / np.nanmax(tpl_flux)

    ccf_orders = []
    for k, (wl, fx) in enumerate(spectrum_orders):
        good = np.isfinite(wl) & np.isfinite(fx)
        wl, fx = wl[good], fx[good]
        if wl.size < 10:
            continue
        # şablonun bu basamağı kapsayıp kapsamadığını denetle
        if wl.min() < tpl_wl.min() or wl.max() > tpl_wl.max():
            continue
        peak = np.nanmax(fx)
        if peak <= 0:
            continue
        ccf_orders.append(calculate_ccf(wl, fx / peak, tpl_wl, tpl_norm, rv_grid))
        print(f"  basamak {k + 1}/{len(spectrum_orders)} tamam", end="\r")
    print()
    if not ccf_orders:
        raise RuntimeError("Hiçbir basamak için CCF hesaplanamadı "
                           "(şablon ile dalgaboyu örtüşmesini denetleyin).")

    ccf_total = np.sum(ccf_orders, axis=0)
    ccf_total = ccf_total / np.nanmax(ccf_total)

    # Gauss uyumlaması: başlangıç tahmini tepe konumundan
    x0 = rv_grid[np.argmax(ccf_total)]
    amp0 = ccf_total.max() - np.median(ccf_total)
    p0 = [amp0, x0, 5.0, np.median(ccf_total)]
    popt, pcov = curve_fit(gauss, rv_grid, ccf_total, p0=p0, maxfev=20000)
    rv, rv_err = popt[1], float(np.sqrt(pcov[1, 1]))

    return dict(rv=rv, rv_err=rv_err, rv_grid=rv_grid,
                ccf_total=ccf_total, ccf_orders=np.array(ccf_orders), popt=popt)


# ----------------------------------------------------------------------
# 2) BF yöntemi (tez Bölüm 2.2/3.2: Rucinski SVD yaklaşımı)
# ----------------------------------------------------------------------

def log_wave_grid(wl_min, wl_max, dv_kms):
    """Sabit hız adımlı (log-dalgaboyu) ızgara üretir.

    Log-λ uzayında sabit adım = sabit hız adımı; Doppler kayması burada
    basit bir piksel ötelemesine dönüşür — BF'nin konvolüsyon modeli için şart.
    """
    step = np.log(1.0 + dv_kms / C_KMS)
    n = int(np.floor(np.log(wl_max / wl_min) / step)) + 1
    return wl_min * np.exp(step * np.arange(n))


def compute_bf(spec_wl, spec_flux, tpl_wl, tpl_flux,
               vel_range=400.0, dv=None, svd_rcond=1e-3, smooth_kms=None):
    """Broadening Function'ı SVD ile çözer.

    Model:  s = A · b   ;   A sütunları, şablonun hız uzayında piksel piksel
    kaydırılmış kopyaları (Toeplitz/dizayn matrisi). SVD ile en küçük kareler
    çözümünde küçük tekil değerler kesilerek (svd_rcond) gürültü bastırılır —
    tezde anılan 'Tekil Değer Ayrışımı' adımı budur.

    Parametreler
    ------------
    vel_range : BF penceresinin yarı genişliği [km/s] (±vel_range taranır)
    dv        : hız adımı [km/s]; None ise verinin medyan pikselinden alınır
    svd_rcond : s_i < rcond·s_max tekil değerleri atılır (düzenlileştirme)
    smooth_kms: BF'ye uygulanacak Gauss yumuşatma FWHM'i [km/s]
                (None -> 3·dv; Rucinski'nin önerdiği hafif yumuşatma)

    Dönen: dict(velocity, bf, bf_smooth)
    """
    # Normalize tayfları çizgi-derinliği uzayına çevir: sürekliliği ~0 yap,
    # çizgiler pozitif tepe olsun (1 - akı). Böylece BF tepeleri pozitif çıkar.
    good_s = np.isfinite(spec_wl) & np.isfinite(spec_flux)
    good_t = np.isfinite(tpl_wl) & np.isfinite(tpl_flux)
    spec_wl, spec_flux = spec_wl[good_s], spec_flux[good_s]
    tpl_wl, tpl_flux = tpl_wl[good_t], tpl_flux[good_t]

    wl_min = max(spec_wl.min(), tpl_wl.min())
    wl_max = min(spec_wl.max(), tpl_wl.max())
    if wl_max <= wl_min:
        raise ValueError("Tayf ile şablon dalgaboyu aralıkları örtüşmüyor.")

    if dv is None:
        pix = np.median(np.diff(spec_wl)) / np.median(spec_wl) * C_KMS
        dv = max(pix, 0.5)

    # BF penceresi kadar pay bırak ki kaydırılmış şablon aralık dışına taşmasın
    margin = 1.5 * vel_range / C_KMS
    grid = log_wave_grid(wl_min * (1 + margin), wl_max * (1 - margin), dv)

    s = 1.0 - np.interp(grid, spec_wl, spec_flux)   # gözlenen (çizgi derinliği)
    t = 1.0 - np.interp(grid, tpl_wl, tpl_flux)      # şablon

    m = grid.size
    half = int(np.ceil(vel_range / dv))
    nbf = 2 * half + 1
    if m <= 2 * nbf:
        raise ValueError("Tayf parçası BF penceresine göre çok kısa; "
                         "dalgaboyu aralığını genişletin veya vel-range'i küçültün.")

    # Dizayn matrisi: satır k -> s[k+half] = Σ_j b[j]·t[k+half-(j-half)]
    # (pozitif RV = kırmızıya kayma = daha büyük piksele öteleme)
    nrow = m - nbf + 1
    idx = (np.arange(nrow)[:, None] + (nbf - 1) - np.arange(nbf)[None, :])
    A = t[idx]
    rhs = s[half:m - half]

    # SVD çözümü + kesme (regularizasyon)
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
    """BF profiline tek/çift Gauss uyumlar (tez: 'BF'lere Gauss uyumlaması').

    components=2 için başlangıç tahminleri, aralarında en az min_sep [km/s]
    olan iki en yüksek yerel tepe alınarak yapılır.

    Dönen: her bileşen için (rv, rv_err, amp, sigma) sözlüğü listesi.
    """
    offset0 = np.median(bf)
    if components == 1:
        i0 = np.argmax(bf)
        p0 = [bf[i0] - offset0, velocity[i0], 20.0, offset0]
        popt, pcov = curve_fit(gauss, velocity, bf, p0=p0, maxfev=20000)
        err = np.sqrt(np.diag(pcov))
        return [dict(rv=popt[1], rv_err=float(err[1]),
                     amp=popt[0], sigma=abs(popt[2]))], popt

    # iki tepe ara
    i1 = int(np.argmax(bf))
    mask = np.abs(velocity - velocity[i1]) > min_sep
    if not mask.any():
        raise RuntimeError("İkinci tepe için yeterli hız aralığı yok; "
                           "--min-sep değerini küçültün.")
    i2 = int(np.flatnonzero(mask)[np.argmax(bf[mask])])

    p0 = [bf[i1] - offset0, velocity[i1], 20.0,
          bf[i2] - offset0, velocity[i2], 20.0, offset0]
    popt, pcov = curve_fit(two_gauss, velocity, bf, p0=p0, maxfev=40000)
    err = np.sqrt(np.diag(pcov))
    comps = [dict(rv=popt[1], rv_err=float(err[1]), amp=popt[0], sigma=abs(popt[2])),
             dict(rv=popt[4], rv_err=float(err[4]), amp=popt[3], sigma=abs(popt[5]))]
    comps.sort(key=lambda c: c["rv"])          # önce daha negatif hızlı bileşen
    return comps, popt


# ----------------------------------------------------------------------
# Grafikler
# ----------------------------------------------------------------------

def plot_ccf(result, outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ccf in result["ccf_orders"]:
        axes[0].plot(result["rv_grid"], ccf, lw=0.6, alpha=0.5)
    axes[0].set_ylabel("CCF (basamak başına)")
    axes[0].set_title("Basamak CCF'leri ve toplam CCF + Gauss uyumu")

    axes[1].plot(result["rv_grid"], result["ccf_total"], "k-", lw=1.2,
                 label="Toplam CCF (normalize)")
    axes[1].plot(result["rv_grid"], gauss(result["rv_grid"], *result["popt"]),
                 "r-", lw=2, alpha=0.7,
                 label=f"Gauss: RV = {result['rv']:.3f} ± {result['rv_err']:.3f} km/s")
    axes[1].axvline(result["rv"], color="r", ls=":", lw=1)
    axes[1].set_xlabel("RV [km/s]")
    axes[1].set_ylabel("Normalize CCF")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Grafik kaydedildi: {outfile}")


def plot_bf(bf_result, comps, popt, components, outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = bf_result["velocity"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(v, bf_result["bf"], color="0.7", lw=0.8, label="BF (ham)")
    ax.plot(v, bf_result["bf_smooth"], "b-", lw=1.5, label="BF (yumuşatılmış)")
    model = gauss(v, *popt) if components == 1 else two_gauss(v, *popt)
    ax.plot(v, model, "r--", lw=2, alpha=0.8, label="Gauss uyumu")
    for i, c in enumerate(comps, 1):
        ax.axvline(c["rv"], color="r", ls=":", lw=1)
        ax.annotate(f"B{i}: {c['rv']:.2f} km/s", (c["rv"], c["amp"]),
                    textcoords="offset points", xytext=(6, 6), color="r")
    ax.set_xlabel("Dikine hız [km/s]")
    ax.set_ylabel("Genişleme Fonksiyonu")
    ax.set_title("Broadening Function ve Gauss uyumlaması")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Grafik kaydedildi: {outfile}")


# ----------------------------------------------------------------------
# Alt komutlar
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

    print(f"{len(orders)} basamak/tayf parçası üzerinde CCF hesaplanıyor...")
    result = run_ccf(orders, tpl_wl, tpl_flux, args.rv_min, args.rv_max, args.rv_step)

    vbary = 0.0
    if args.ra is not None:
        vbary = barycentric_correction(args.ra, args.dec, args.obstime, args.site)
        print(f"Barycentric düzeltme: {vbary:+.4f} km/s")

    rv = result["rv"] + vbary
    print("\n================ CCF SONUCU ================")
    print(f"RV = {rv:.4f} ± {result['rv_err']:.4f} km/s"
          + ("  (barycentric düzeltilmiş)" if vbary else ""))
    print("============================================")

    if args.output:
        with open(args.output, "w") as f:
            f.write("# yontem  RV[km/s]  RV_hata[km/s]  bary_duzeltme[km/s]\n")
            f.write(f"CCF  {rv:.5f}  {result['rv_err']:.5f}  {vbary:.5f}\n")
        print(f"Sonuç yazıldı: {args.output}")
    if args.plot:
        plot_ccf(result, args.plot)


def cmd_bf(args):
    orders = read_spectrum(args.spectrum, args.format)
    if len(orders) > 1:
        # echelle basamaklarını tek diziye birleştir (tezde 'combine' adımı)
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

    print("BF, SVD ile çözülüyor...")
    bf_result = compute_bf(spec_wl, spec_flux, tpl_wl, tpl_flux,
                           vel_range=args.vel_range, dv=args.dv,
                           svd_rcond=args.svd_rcond, smooth_kms=args.smooth)
    print(f"  hız adımı dv = {bf_result['dv']:.3f} km/s, "
          f"tekil değer: {bf_result['n_kept_sv']}/{bf_result['n_sv']} tutuldu")

    comps, popt = fit_bf_peaks(bf_result["velocity"], bf_result["bf_smooth"],
                               components=args.components, min_sep=args.min_sep)

    vbary = 0.0
    if args.ra is not None:
        vbary = barycentric_correction(args.ra, args.dec, args.obstime, args.site)
        print(f"Barycentric düzeltme: {vbary:+.4f} km/s")

    print("\n================ BF SONUCU =================")
    for i, c in enumerate(comps, 1):
        print(f"Bileşen {i}: RV = {c['rv'] + vbary:.4f} ± {c['rv_err']:.4f} km/s   "
              f"(genlik={c['amp']:.4f}, sigma={c['sigma']:.2f} km/s)")
    if len(comps) == 2 and (comps[0]["amp"] > 0) and (comps[1]["amp"] > 0):
        # BF alan oranı ~ ışık katkısı oranı; genlik·sigma ile yaklaşıklanır
        l2_l1 = (comps[1]["amp"] * comps[1]["sigma"]) / \
                (comps[0]["amp"] * comps[0]["sigma"])
        print(f"Işık katkısı oranı (B2/B1, BF alanlarından) ≈ {l2_l1:.3f}")
    print("============================================")

    if args.output:
        with open(args.output, "w") as f:
            f.write("# bilesen  RV[km/s]  RV_hata[km/s]  genlik  sigma[km/s]  "
                    "bary_duzeltme[km/s]\n")
            for i, c in enumerate(comps, 1):
                f.write(f"{i}  {c['rv'] + vbary:.5f}  {c['rv_err']:.5f}  "
                        f"{c['amp']:.5f}  {c['sigma']:.5f}  {vbary:.5f}\n")
        print(f"Sonuç yazıldı: {args.output}")
    if args.plot:
        plot_bf(bf_result, comps, popt, args.components, args.plot)


def cmd_demo(args):
    """Sentetik SB2 tayfı üretip hem CCF hem BF ile çözer — kendi kendini test.

    Gerçek RV'ler bilindiği için iki yöntemin davranış farkı doğrudan görülür:
    CCF iki bileşeni geniş/karışık tepeler olarak, BF ise keskin ayrık
    tepeler olarak gösterir.
    """
    rng = np.random.default_rng(42)
    rv1_true, rv2_true = -80.0, 120.0     # bileşen hızları [km/s]
    light_ratio = 0.45                    # B2/B1 ışık katkısı

    # Şablon: 5000–5500 Å'da rastgele soğurma çizgileriyle normalize tayf
    wl = log_wave_grid(4950.0, 5550.0, 1.0)
    tpl = np.ones_like(wl)
    n_lines = 160
    centers = rng.uniform(4960, 5540, n_lines)
    depths = rng.uniform(0.15, 0.85, n_lines)
    widths = rng.uniform(0.08, 0.25, n_lines)  # Å
    for c0, d, w in zip(centers, depths, widths):
        tpl -= d * np.exp(-((wl - c0) ** 2) / (2 * w ** 2))
    tpl = np.clip(tpl, 0.02, None)

    # Gözlenen tayf: iki Doppler kaydırılmış + dönmece genişletilmiş bileşen
    def rot_broaden(flux, vsini_kms):
        sigma_pix = vsini_kms / 1.0 / 2.35482
        return gaussian_filter1d(flux, sigma_pix)

    f1 = np.interp(wl, doppler_shift(wl, rv1_true), tpl)
    f2 = np.interp(wl, doppler_shift(wl, rv2_true), tpl)
    f1 = rot_broaden(f1, 40.0)
    f2 = rot_broaden(f2, 25.0)
    obs = (f1 + light_ratio * f2) / (1.0 + light_ratio)
    obs += rng.normal(0, 0.004, obs.size)   # S/N ~ 250

    print("Sentetik SB2 sistemi üretildi:")
    print(f"  Gerçek RV1 = {rv1_true} km/s, RV2 = {rv2_true} km/s, "
          f"ışık oranı = {light_ratio}\n")

    # --- BF ---
    print("--- BF yöntemi ---")
    bf_result = compute_bf(wl, obs, wl, tpl, vel_range=300.0, dv=2.0,
                           svd_rcond=5e-4, smooth_kms=10.0)
    comps, popt = fit_bf_peaks(bf_result["velocity"], bf_result["bf_smooth"],
                               components=2, min_sep=50.0)
    for i, c in enumerate(comps, 1):
        print(f"  Bileşen {i}: RV = {c['rv']:8.3f} ± {c['rv_err']:.3f} km/s")

    # --- CCF ---
    print("--- CCF yöntemi (aynı veri) ---")
    rv_grid = np.arange(-200, 251, 2.0)
    ccf = calculate_ccf(wl, 1.0 - obs, wl, 1.0 - tpl, rv_grid)
    ccf /= ccf.max()
    ccomp, cpopt = fit_bf_peaks(rv_grid, ccf, components=2, min_sep=50.0)
    for i, c in enumerate(ccomp, 1):
        print(f"  Bileşen {i}: RV = {c['rv']:8.3f} ± {c['rv_err']:.3f} km/s")

    print("\nGerçek değerlerle karşılaştırma (km/s):")
    print(f"  BF : ΔRV1 = {comps[0]['rv'] - rv1_true:+.3f}, "
          f"ΔRV2 = {comps[1]['rv'] - rv2_true:+.3f}")
    print(f"  CCF: ΔRV1 = {ccomp[0]['rv'] - rv1_true:+.3f}, "
          f"ΔRV2 = {ccomp[1]['rv'] - rv2_true:+.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 11))
        sel = (wl > 5195) & (wl < 5235)
        axes[0].plot(wl[sel], obs[sel], "k-", lw=0.8, label="Gözlenen (SB2, sentetik)")
        axes[0].plot(wl[sel], tpl[sel], "C0-", lw=0.8, alpha=0.6, label="Şablon")
        axes[0].set_xlabel("Dalgaboyu [Å]")
        axes[0].set_ylabel("Normalize akı")
        axes[0].legend()
        axes[0].set_title("Sentetik SB2 tayfı (kesit)")

        v = bf_result["velocity"]
        axes[1].plot(v, bf_result["bf_smooth"], "b-", lw=1.5, label="BF")
        axes[1].plot(v, two_gauss(v, *popt), "r--", lw=1.5, label="Çift Gauss uyumu")
        for rvt in (rv1_true, rv2_true):
            axes[1].axvline(rvt, color="0.5", ls=":", lw=1)
        axes[1].set_ylabel("BF")
        axes[1].set_title("Broadening Function: bileşenler keskin ve ayrık")
        axes[1].legend()

        axes[2].plot(rv_grid, ccf, "k-", lw=1.5, label="CCF")
        axes[2].plot(rv_grid, two_gauss(rv_grid, *cpopt), "r--", lw=1.5,
                     label="Çift Gauss uyumu")
        for rvt in (rv1_true, rv2_true):
            axes[2].axvline(rvt, color="0.5", ls=":", lw=1)
        axes[2].set_xlabel("Dikine hız [km/s]")
        axes[2].set_ylabel("Normalize CCF")
        axes[2].set_title("CCF: tepeler daha geniş, harmanlanmaya yatkın")
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nKarşılaştırma grafiği kaydedildi: {args.plot}")


# ----------------------------------------------------------------------
# Komut satırı arayüzü
# ----------------------------------------------------------------------

def add_common_args(p):
    p.add_argument("--spectrum", required=True, help="Gözlenen tayf dosyası")
    p.add_argument("--format", default="auto", choices=["auto", "s2d", "text"],
                   help="Tayf formatı (auto: uzantıdan sez)")
    p.add_argument("--template", help="Şablon tayf dosyası (dalgaboyu, akı)")
    p.add_argument("--teff", type=float, help="Şablon için T_eff [K] (expecto)")
    p.add_argument("--logg", type=float, default=4.5, help="Şablon log g")
    p.add_argument("--feh", type=float, default=0.0, help="Şablon [Fe/H]")
    p.add_argument("--wave-min", type=float, help="Kullanılacak min dalgaboyu [Å]")
    p.add_argument("--wave-max", type=float, help="Kullanılacak max dalgaboyu [Å]")
    p.add_argument("--ra", type=float, help="Barycentric düzeltme için RA [derece]")
    p.add_argument("--dec", type=float, help="Barycentric düzeltme için Dec [derece]")
    p.add_argument("--obstime", help="Gözlem zamanı (ISOT, ör. 2024-12-03T02:30:00)")
    p.add_argument("--site", default="paranal",
                   help="Gözlemevi adı (astropy site adı, ör. paranal, tug)")
    p.add_argument("--plot", help="Grafik PNG dosya adı")
    p.add_argument("--output", help="Sonuçların yazılacağı metin dosyası")


def main():
    parser = argparse.ArgumentParser(
        description="Tayftan dikine hız (RV) hesabı: CCF ve BF yöntemleri",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Kullanım örnekleri")[1] if "Kullanım örnekleri" in __doc__ else "")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ccf = sub.add_parser("ccf", help="Çapraz korelasyon (çalıştay yöntemi)")
    add_common_args(p_ccf)
    p_ccf.add_argument("--rv-min", type=float, default=-200.0, help="RV tarama alt sınırı [km/s]")
    p_ccf.add_argument("--rv-max", type=float, default=200.0, help="RV tarama üst sınırı [km/s]")
    p_ccf.add_argument("--rv-step", type=float, default=0.5, help="RV adımı [km/s]")
    p_ccf.set_defaults(func=cmd_ccf)

    p_bf = sub.add_parser("bf", help="Broadening Function (tez yöntemi, SVD)")
    add_common_args(p_bf)
    p_bf.add_argument("--vel-range", type=float, default=400.0,
                      help="BF penceresi yarı genişliği [km/s]")
    p_bf.add_argument("--dv", type=float, help="Hız adımı [km/s] (varsayılan: veri pikseli)")
    p_bf.add_argument("--svd-rcond", type=float, default=1e-3,
                      help="SVD kesme oranı (küçük tekil değer eşiği)")
    p_bf.add_argument("--smooth", type=float,
                      help="BF yumuşatma FWHM [km/s] (varsayılan: 3·dv)")
    p_bf.add_argument("--components", type=int, default=1, choices=[1, 2],
                      help="Uyumlanacak bileşen (Gauss) sayısı: SB1=1, SB2=2")
    p_bf.add_argument("--min-sep", type=float, default=30.0,
                      help="İki tepe için asgari ayrıklık [km/s]")
    p_bf.set_defaults(func=cmd_bf)

    p_demo = sub.add_parser("demo", help="Sentetik SB2 verisiyle CCF-BF karşılaştırma testi")
    p_demo.add_argument("--plot", help="Karşılaştırma grafiği PNG dosya adı")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
