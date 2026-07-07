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

Çalıştırma biçimleri
--------------------
# 1) ETKİLEŞİMLİ MOD (önerilen): argümansız çalıştırın.
#    Grafik arayüz (tkinter widget) açılır: gözlemsel tayf ve sentetik/şablon
#    tayf dosyaları seçilir, CCF/BF yöntemi işaretlenir, "Hesapla" ile sonuç
#    ve uyum grafiği pencerede görünür. Ekran/tkinter yoksa otomatik olarak
#    terminal soru-cevap sihirbazına düşer.
python rv_analysis.py

# 2) Komut satırı modu (betikleme/tayf serileri için):
python rv_analysis.py ccf --spectrum tayf.fits --format s2d \
    --teff 6628 --logg 4.251 --feh 0.17 --rv-min -20 --rv-max 100
python rv_analysis.py bf --spectrum tayf.txt --template sablon.txt \
    --vel-range 500 --components 2 --wave-min 5000 --wave-max 5500

# 3) Sentetik veriyle kendi kendini test (veri gerekmez):
python rv_analysis.py demo --plot demo.png

Çıktılar
--------
Her analiz sonunda otomatik olarak (isim verilmezse):
  result_CCF.txt / result_BF.txt  — sayısal sonuçlar
  result_CCF.png / result_BF.png  — uyum (fit) grafiği

Desteklenen girdiler
--------------------
- ESPRESSO S2D FITS (çok basamaklı echelle: akı ext=1, dalgaboyu ext=4)
- İki/üç sütunlu metin dosyası: dalgaboyu[Å]  akı  [akı_hatası]
  (IRAF ile indirgenmiş, normalize edilmiş, birleştirilmiş tayf — tezdeki
   5000–5500 Å bölgesi gibi)
- Şablon/sentetik tayf: metin dosyası (dalgaboyu, akı) veya `expecto`
  kuruluysa PHOENIX modelinden otomatik indirme (--teff/--logg/--feh).
"""

import argparse
import os
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
    """Şablon/sentetik tayfı yükle: dosyadan ya da expecto/PHOENIX'ten."""
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
    raise ValueError("Şablon gerekli: şablon dosyası veya T_eff/log g/[Fe/H] verin.")


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
# Grafikler (pyplot yerine OO Figure: hem dosyaya kayıt hem GUI'ye gömme
# aynı fonksiyonla, backend çakışması olmadan yapılabilsin diye)
# ----------------------------------------------------------------------

def make_ccf_figure(result):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 8))
    ax0, ax1 = fig.subplots(2, 1, sharex=True)

    for ccf in result["ccf_orders"]:
        ax0.plot(result["rv_grid"], ccf, lw=0.6, alpha=0.5)
    ax0.set_ylabel("CCF (basamak başına)")
    ax0.set_title("Basamak CCF'leri ve toplam CCF + Gauss uyumu")

    ax1.plot(result["rv_grid"], result["ccf_total"], "k-", lw=1.2,
             label="Toplam CCF (normalize)")
    ax1.plot(result["rv_grid"], gauss(result["rv_grid"], *result["popt"]),
             "r-", lw=2, alpha=0.7,
             label=f"Gauss: RV = {result['rv']:.3f} ± {result['rv_err']:.3f} km/s")
    ax1.axvline(result["rv"], color="r", ls=":", lw=1)
    ax1.set_xlabel("RV [km/s]")
    ax1.set_ylabel("Normalize CCF")
    ax1.legend()
    fig.tight_layout()
    return fig


def make_bf_figure(bf_result, comps, popt, components):
    from matplotlib.figure import Figure
    v = bf_result["velocity"]
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
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
    return fig


def save_figure(fig, outfile):
    fig.savefig(outfile, dpi=150)
    print(f"Grafik kaydedildi: {outfile}")


# ----------------------------------------------------------------------
# Analiz komutları — hem CLI hem etkileşimli mod bunları çağırır.
# Sonuçlar her zaman result_CCF/result_BF dosyalarına (txt + png) yazılır.
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
    summary = ("================ CCF SONUCU ================\n"
               f"Gözlemsel tayf : {args.spectrum}\n"
               f"Şablon         : {args.template or f'PHOENIX T={args.teff}K'}\n"
               f"RV = {rv:.4f} ± {result['rv_err']:.4f} km/s"
               + ("  (barycentric düzeltilmiş)\n" if vbary else "\n")
               + "============================================")
    print("\n" + summary)

    outfile = args.output or "result_CCF.txt"
    with open(outfile, "w") as f:
        f.write(f"# Gozlemsel tayf : {args.spectrum}\n")
        f.write(f"# Sablon         : {args.template or f'PHOENIX T={args.teff}K'}\n")
        f.write("# yontem  RV[km/s]  RV_hata[km/s]  bary_duzeltme[km/s]\n")
        f.write(f"CCF  {rv:.5f}  {result['rv_err']:.5f}  {vbary:.5f}\n")
    print(f"Sonuç yazıldı: {outfile}")

    plotfile = args.plot or "result_CCF.png"
    fig = make_ccf_figure(result)
    save_figure(fig, plotfile)

    return dict(method="CCF", fig=fig, text=summary,
                output=outfile, plot=plotfile)


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

    lines = ["================ BF SONUCU =================",
             f"Gözlemsel tayf : {args.spectrum}",
             f"Şablon         : {args.template or f'PHOENIX T={args.teff}K'}"]
    for i, c in enumerate(comps, 1):
        lines.append(f"Bileşen {i}: RV = {c['rv'] + vbary:.4f} ± {c['rv_err']:.4f} km/s"
                     f"   (genlik={c['amp']:.4f}, sigma={c['sigma']:.2f} km/s)")
    if len(comps) == 2 and (comps[0]["amp"] > 0) and (comps[1]["amp"] > 0):
        # BF alan oranı ~ ışık katkısı oranı; genlik·sigma ile yaklaşıklanır
        l2_l1 = (comps[1]["amp"] * comps[1]["sigma"]) / \
                (comps[0]["amp"] * comps[0]["sigma"])
        lines.append(f"Işık katkısı oranı (B2/B1, BF alanlarından) ≈ {l2_l1:.3f}")
    lines.append("============================================")
    summary = "\n".join(lines)
    print("\n" + summary)

    outfile = args.output or "result_BF.txt"
    with open(outfile, "w") as f:
        f.write(f"# Gozlemsel tayf : {args.spectrum}\n")
        f.write(f"# Sablon         : {args.template or f'PHOENIX T={args.teff}K'}\n")
        f.write("# bilesen  RV[km/s]  RV_hata[km/s]  genlik  sigma[km/s]  "
                "bary_duzeltme[km/s]\n")
        for i, c in enumerate(comps, 1):
            f.write(f"{i}  {c['rv'] + vbary:.5f}  {c['rv_err']:.5f}  "
                    f"{c['amp']:.5f}  {c['sigma']:.5f}  {vbary:.5f}\n")
    print(f"Sonuç yazıldı: {outfile}")

    plotfile = args.plot or "result_BF.png"
    fig = make_bf_figure(bf_result, comps, popt, args.components)
    save_figure(fig, plotfile)

    return dict(method="BF", fig=fig, text=summary,
                output=outfile, plot=plotfile)


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
        from matplotlib.figure import Figure

        fig = Figure(figsize=(11, 11))
        axes = fig.subplots(3, 1)
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
        save_figure(fig, args.plot)


# ----------------------------------------------------------------------
# Etkileşimli mod — argümansız çalıştırınca devreye girer.
# Önce tkinter widget arayüzü denenir; ekran/tkinter yoksa terminal
# soru-cevap sihirbazına düşülür. İkisi de aynı cmd_ccf/cmd_bf çekirdeğini
# çağırır, dolayısıyla çıktılar (result_*.txt + result_*.png) özdeştir.
# ----------------------------------------------------------------------

def make_args(**overrides):
    """cmd_ccf/cmd_bf'nin beklediği tüm alanları içeren Namespace üretir."""
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
    """Terminal sihirbazı için tek soru: varsayılanlı, doğrulamalı input."""
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            if default is not None:
                return default
            if allow_empty:
                return None
            print("  Bu alan boş bırakılamaz.")
            continue
        try:
            val = cast(raw)
        except (TypeError, ValueError):
            print("  Geçersiz değer, tekrar deneyin.")
            continue
        if validate is not None and not validate(val):
            print("  Geçersiz seçim, tekrar deneyin.")
            continue
        return val


def _ask_common_inputs():
    """Her iki yöntem için ortak girdileri sor: dosyalar, dalgaboyu aralığı."""
    spectrum = ask("Gözlemsel tayf dosyası (FITS S2D veya metin)",
                   cast=str, validate=os.path.isfile)
    template = ask("Sentetik/şablon tayf dosyası (boş: PHOENIX modeli indirilecek)",
                   allow_empty=True,
                   validate=lambda p: p is None or os.path.isfile(p))
    kw = dict(spectrum=spectrum, template=template)
    if template is None:
        kw["teff"] = ask("  Şablon T_eff [K]", cast=float)
        kw["logg"] = ask("  Şablon log g", default=4.5, cast=float)
        kw["feh"] = ask("  Şablon [Fe/H]", default=0.0, cast=float)
    kw["wave_min"] = ask("Kullanılacak min dalgaboyu [Å] (boş: tümü)",
                         allow_empty=True, cast=float)
    kw["wave_max"] = ask("Kullanılacak max dalgaboyu [Å] (boş: tümü)",
                         allow_empty=True, cast=float)
    return kw


def _ask_barycentric(kw):
    yanit = ask("Barycentric düzeltme uygulansın mı? (e/h)", default="h",
                cast=str, validate=lambda s: s.lower() in ("e", "h"))
    if yanit.lower() == "e":
        kw["ra"] = ask("  RA [derece]", cast=float)
        kw["dec"] = ask("  Dec [derece]", cast=float)
        kw["obstime"] = ask("  Gözlem zamanı (ISOT, ör. 2024-12-03T02:30:00)")
        kw["site"] = ask("  Gözlemevi (astropy adı: tug, paranal, ...)",
                         default="tug")
    return kw


def run_terminal_wizard():
    """Terminal üzerinden adım adım RV analizi (GUI açılamadığında)."""
    print("=" * 60)
    print(" RV ANALİZİ — Etkileşimli Terminal Modu")
    print("=" * 60)

    kw = _ask_common_inputs()

    print("\nHangi yöntemle devam etmek istiyorsun?")
    print("  [1] CCF — Çapraz Korelasyon (tek yıldız / SB1 için pratik)")
    print("  [2] BF  — Broadening Function (çift sistem / SB2 için önerilen)")
    secim = ask("Seçim", default="2", cast=str,
                validate=lambda s: s in ("1", "2"))

    if secim == "1":
        kw["rv_min"] = ask("RV tarama alt sınırı [km/s]", default=-200.0, cast=float)
        kw["rv_max"] = ask("RV tarama üst sınırı [km/s]", default=200.0, cast=float)
        kw["rv_step"] = ask("RV adımı [km/s]", default=0.5, cast=float)
        kw = _ask_barycentric(kw)
        print()
        cmd_ccf(make_args(**kw))
    else:
        kw["vel_range"] = ask("BF penceresi yarı genişliği [km/s]",
                              default=400.0, cast=float)
        kw["components"] = ask("Bileşen sayısı (SB1=1, SB2=2)", default=2,
                               cast=int, validate=lambda n: n in (1, 2))
        kw["smooth"] = ask("BF yumuşatma FWHM [km/s] (boş: otomatik)",
                           allow_empty=True, cast=float)
        kw["svd_rcond"] = ask("SVD kesme eşiği", default=1e-3, cast=float)
        kw = _ask_barycentric(kw)
        print()
        cmd_bf(make_args(**kw))

    print("\nBitti. Sonuç dosyaları ve uyum grafiği çalışma dizinine kaydedildi.")


def run_gui():
    """tkinter widget arayüzü: dosya seçimi, yöntem seçimi, gömülü uyum grafiği.

    Analiz çekirdeği CLI ile aynı (cmd_ccf/cmd_bf); sonuç metni pencerede
    gösterilir, dosyalar (result_*.txt, result_*.png) yine diske yazılır.
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("RV Analizi — CCF / BF")

    main = ttk.Frame(root, padding=10)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # --- 1) Veri dosyaları ------------------------------------------------
    files = ttk.LabelFrame(main, text="1) Veri dosyaları", padding=8)
    files.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    files.columnconfigure(1, weight=1)

    spec_var = tk.StringVar()
    tpl_var = tk.StringVar()

    def browse(var, title):
        p = filedialog.askopenfilename(
            title=title,
            filetypes=[("Tayf dosyaları", "*.fits *.fit *.txt *.dat *.ascii"),
                       ("Tümü", "*.*")])
        if p:
            var.set(p)

    ttk.Label(files, text="Gözlemsel tayf:").grid(row=0, column=0, sticky="w")
    ttk.Entry(files, textvariable=spec_var, width=52).grid(row=0, column=1,
                                                           sticky="ew", padx=4)
    ttk.Button(files, text="Gözat...",
               command=lambda: browse(spec_var, "Gözlemsel tayf seç")
               ).grid(row=0, column=2)

    ttk.Label(files, text="Sentetik/şablon tayf:").grid(row=1, column=0, sticky="w")
    ttk.Entry(files, textvariable=tpl_var, width=52).grid(row=1, column=1,
                                                          sticky="ew", padx=4)
    ttk.Button(files, text="Gözat...",
               command=lambda: browse(tpl_var, "Şablon tayf seç")
               ).grid(row=1, column=2)

    teff_var, logg_var, feh_var = tk.StringVar(), tk.StringVar(value="4.5"), \
        tk.StringVar(value="0.0")
    phx = ttk.Frame(files)
    phx.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(phx, text="Şablon dosyası boşsa PHOENIX (expecto):  T_eff[K]"
              ).pack(side="left")
    ttk.Entry(phx, textvariable=teff_var, width=7).pack(side="left", padx=2)
    ttk.Label(phx, text="log g").pack(side="left")
    ttk.Entry(phx, textvariable=logg_var, width=5).pack(side="left", padx=2)
    ttk.Label(phx, text="[Fe/H]").pack(side="left")
    ttk.Entry(phx, textvariable=feh_var, width=5).pack(side="left", padx=2)

    wmin_var, wmax_var = tk.StringVar(), tk.StringVar()
    wrng = ttk.Frame(files)
    wrng.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(wrng, text="Dalgaboyu aralığı [Å] (boş: tümü):  min").pack(side="left")
    ttk.Entry(wrng, textvariable=wmin_var, width=8).pack(side="left", padx=2)
    ttk.Label(wrng, text="max").pack(side="left")
    ttk.Entry(wrng, textvariable=wmax_var, width=8).pack(side="left", padx=2)

    # --- 2) Yöntem seçimi -------------------------------------------------
    method_var = tk.StringVar(value="BF")
    method_box = ttk.LabelFrame(main, text="2) Hangi yöntemle devam etmek "
                                           "istiyorsun?", padding=8)
    method_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))

    ccf_frame = ttk.Frame(method_box)
    bf_frame = ttk.Frame(method_box)

    def on_method_change():
        if method_var.get() == "CCF":
            bf_frame.grid_remove()
            ccf_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        else:
            ccf_frame.grid_remove()
            bf_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    ttk.Radiobutton(method_box, text="CCF — Çapraz Korelasyon (SB1 / tek yıldız)",
                    variable=method_var, value="CCF",
                    command=on_method_change).grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(method_box, text="BF — Broadening Function (SB2 / çift sistem)",
                    variable=method_var, value="BF",
                    command=on_method_change).grid(row=0, column=1, sticky="w",
                                                   padx=12)

    rvmin_var = tk.StringVar(value="-200")
    rvmax_var = tk.StringVar(value="200")
    rvstep_var = tk.StringVar(value="0.5")
    for j, (lbl, var) in enumerate([("RV min [km/s]", rvmin_var),
                                    ("RV max [km/s]", rvmax_var),
                                    ("RV adımı [km/s]", rvstep_var)]):
        ttk.Label(ccf_frame, text=lbl).grid(row=0, column=2 * j, sticky="w")
        ttk.Entry(ccf_frame, textvariable=var, width=8).grid(row=0, column=2 * j + 1,
                                                             padx=(2, 10))

    vrange_var = tk.StringVar(value="400")
    comp_var = tk.IntVar(value=2)
    smooth_var = tk.StringVar()
    rcond_var = tk.StringVar(value="1e-3")
    ttk.Label(bf_frame, text="Pencere ±[km/s]").grid(row=0, column=0, sticky="w")
    ttk.Entry(bf_frame, textvariable=vrange_var, width=8).grid(row=0, column=1,
                                                               padx=(2, 10))
    ttk.Label(bf_frame, text="Bileşen").grid(row=0, column=2, sticky="w")
    ttk.Combobox(bf_frame, textvariable=comp_var, values=[1, 2], width=3,
                 state="readonly").grid(row=0, column=3, padx=(2, 10))
    ttk.Label(bf_frame, text="Yumuşatma FWHM [km/s]").grid(row=0, column=4,
                                                           sticky="w")
    ttk.Entry(bf_frame, textvariable=smooth_var, width=8).grid(row=0, column=5,
                                                               padx=(2, 10))
    ttk.Label(bf_frame, text="SVD eşiği").grid(row=0, column=6, sticky="w")
    ttk.Entry(bf_frame, textvariable=rcond_var, width=8).grid(row=0, column=7,
                                                              padx=2)
    on_method_change()

    # --- 3) Sonuç metni + gömülü grafik ------------------------------------
    result_text = tk.Text(main, height=8, width=100, state="disabled",
                          font=("Courier", 10))
    result_text.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 6))

    plot_frame = ttk.LabelFrame(main, text="Uyum grafiği (result_*.png olarak da "
                                           "kaydedilir)", padding=4)
    plot_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
    main.rowconfigure(4, weight=1)
    main.columnconfigure(0, weight=1)
    canvas_holder = {"canvas": None}

    def _f(var, default=None):
        """StringVar -> float (boşsa default)."""
        s = var.get().strip()
        return float(s) if s else default

    def hesapla():
        try:
            if not spec_var.get().strip():
                raise ValueError("Gözlemsel tayf dosyası seçilmedi.")
            kw = dict(spectrum=spec_var.get().strip(),
                      template=tpl_var.get().strip() or None,
                      teff=_f(teff_var), logg=_f(logg_var, 4.5),
                      feh=_f(feh_var, 0.0),
                      wave_min=_f(wmin_var), wave_max=_f(wmax_var))
            if method_var.get() == "CCF":
                kw.update(rv_min=_f(rvmin_var, -200.0),
                          rv_max=_f(rvmax_var, 200.0),
                          rv_step=_f(rvstep_var, 0.5))
                payload = cmd_ccf(make_args(**kw))
            else:
                kw.update(vel_range=_f(vrange_var, 400.0),
                          components=int(comp_var.get()),
                          smooth=_f(smooth_var),
                          svd_rcond=_f(rcond_var, 1e-3))
                payload = cmd_bf(make_args(**kw))
        except Exception as exc:
            messagebox.showerror("Hata", str(exc))
            return

        result_text.configure(state="normal")
        result_text.delete("1.0", "end")
        result_text.insert("1.0", payload["text"] + "\n"
                           f"Dosyalar: {payload['output']}, {payload['plot']}")
        result_text.configure(state="disabled")

        if canvas_holder["canvas"] is not None:
            canvas_holder["canvas"].get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(payload["fig"], master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas_holder["canvas"] = canvas

    ttk.Button(main, text="3) Hesapla", command=hesapla
               ).grid(row=2, column=0, columnspan=2, pady=4)

    root.mainloop()


def run_interactive():
    """Argümansız çalıştırma: önce widget (GUI), olmuyorsa terminal sihirbazı."""
    try:
        run_gui()
        return
    except Exception as exc:
        print(f"Grafik arayüz açılamadı ({exc.__class__.__name__}: {exc})")
        print("Terminal moduna geçiliyor...\n")
    run_terminal_wizard()


# ----------------------------------------------------------------------
# Komut satırı arayüzü
# ----------------------------------------------------------------------

def add_common_args(p):
    p.add_argument("--spectrum", required=True, help="Gözlemsel tayf dosyası")
    p.add_argument("--format", default="auto", choices=["auto", "s2d", "text"],
                   help="Tayf formatı (auto: uzantıdan sez)")
    p.add_argument("--template", help="Sentetik/şablon tayf dosyası (dalgaboyu, akı)")
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
    p.add_argument("--plot", help="Grafik PNG dosya adı "
                                  "(varsayılan: result_CCF.png / result_BF.png)")
    p.add_argument("--output", help="Sonuç metin dosyası "
                                    "(varsayılan: result_CCF.txt / result_BF.txt)")


def main():
    # Argümansız çalıştırma -> etkileşimli mod (widget veya terminal sihirbazı)
    if len(sys.argv) == 1:
        run_interactive()
        return

    parser = argparse.ArgumentParser(
        description="Tayftan dikine hız (RV) hesabı: CCF ve BF yöntemleri. "
                    "Argümansız çalıştırınca etkileşimli mod açılır.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gui = sub.add_parser("gui", help="Widget arayüzünü aç (argümansızla aynı)")
    p_gui.set_defaults(func=lambda a: run_interactive())

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
