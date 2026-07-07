# RV_testing — Tayftan Dikine Hız (Radial Velocity) Hesabı

Bu depo, tayfsal veriden dikine hız (RV) ölçümü için tek dosyalık bir Python
aracı içerir: [`rv_analysis.py`](rv_analysis.py). Script iki bağımsız yöntemi
uygular ve şu kaynaklardan derlenmiştir:

- **CCF (Cross-Correlation Function)** — *Between the Lines 2024* çalıştayı
  (E. Sedaghati, ESO): ESPRESSO echelle tayfından çapraz korelasyonla RV ölçümü.
- **BF (Broadening Function)** — MAK_Tez Bölüm 2.2 (yöntem) ve 3.2 (uygulama):
  Rucinski'nin SVD tabanlı Çizgi Genişleme Fonksiyonu yöntemi; örten çift
  sistemlerin (SB2) iki bileşeninin RV'lerinin ayrı ayrı ölçümü.

---

## CCF ile BF arasındaki fark nedir?

İkisi de gözlenen tayfı bir **şablon** (referans/model tayf) ile karşılaştırır,
ama sordukları soru farklıdır:

| | **CCF** | **BF** |
|---|---|---|
| Sorduğu soru | "Şablonu hangi hıza kaydırırsam veriyle en çok benzeşir?" | "Şablonu hız uzayında **hangi fonksiyonla konvolüve edersem** tam olarak veriyi elde ederim?" |
| Matematik | Nokta çarpımı taraması: `CCF(v) = Σ S(λ)·M(λ(1+v/c))` | Doğrusal ters problem: `S = B * T`, SVD ile `B(v)` çözülür |
| Çıktı profili | Şablonun **otokorelasyonu ile bulanmış** geniş tepe | Gerçek çizgi profili genişliğinde **keskin** tepe |
| Doğrusallık | Doğrusal değil — iki yıldızın CCF tepeleri üst üste binince birbirini **kaydırır** (blending sistematiği) | **Doğrusal** — bileşenlerin katkıları toplanır, tepeler birbirini bozmaz |
| Ek çıktılar | Sadece RV | RV + dönme hızı (**vsini**, tepe genişliğinden) + **ışık katkısı oranı** (tepe alanlarından) + varsa 3. cismin izi |
| Gürültü davranışı | Çok sağlam, ayar gerektirmez | SVD kesme/yumuşatma ayarı ister; düşük S/N'de daha nazlıdır |
| İdeal kullanım | **Tek yıldız (SB1)**, öte gezegen RV serileri, hızlı bakış | **Çift/çoklu sistemler (SB2/SB3)**, bileşenleri ayırma, W UMa gibi değen çiftler |

**Sezgisel özet:** CCF, verinin şablonla "benzerlik haritası"dır; her çizgi,
şablonun kendi çizgi genişliğiyle bir kez daha bulanır. BF ise bu bulanıklığı
**dekonvolüsyonla geri alır**: "Şablon yıldızı hangi hız dağılımıyla
gözlemlersem bu tayfı görürdüm?" sorusunun cevabı olan gerçek hız profilini verir.
Bu yüzden tezde de (Bölüm 2.2) vurgulandığı gibi BF'nin en önemli avantajı
**doğrusal** olmasıdır: yakın çift sistemlerde bileşenlerin tepeleri
birbirinden bağımsız kalır ve harmanlanma (blending) kaynaklı sistematik
kaymalar oluşmaz.

### Hangisiyle RV hesaplamak daha mantıklı?

- **Tek yıldız / öte gezegen barındıran yıldız (51 Peg gibi):** CCF yeterli ve
  daha pratiktir; ESPRESSO/HARPS boru hatları da CCF kullanır.
- **Örten/tayfsal çift sistem (tezdeki TIC adayları gibi):** **BF tercih
  edilmelidir.** Bileşenler hız uzayında yakınken (özellikle değen çiftlerde
  konjonksiyon evrelerine yaklaşırken) CCF tepeleri harmanlanır ve K
  genlikleri sistematik olarak küçük ölçülür; BF bu durumda bile bileşenleri
  ayırır. BinMag'ın basit geldiği nokta da tam budur — BF hem RV hem vsini hem
  ışık oranını tek profilden verir.

Bu depodaki `demo` komutu bu farkı sentetik bir SB2 sistemiyle sayısal olarak
gösterir (aşağıya bakın): aynı veride BF ~0.02 km/s, CCF ~0.2–0.6 km/s hata yapar.

---

## Kurulum

```bash
pip install -r requirements.txt
# İsteğe bağlı: PHOENIX model şablonlarını otomatik indirmek için
pip install expecto
```

## Kullanım

### 0) Etkileşimli mod / widget arayüzü (önerilen)

Argümansız çalıştırın:

```bash
python rv_analysis.py
```

- Sisteminizde **tkinter** varsa bir widget penceresi açılır:
  1. **Veri dosyaları** — gözlemsel tayf ve sentetik/şablon tayf dosyalarını
     "Gözat..." ile seçin (şablon boş bırakılırsa T_eff/log g/[Fe/H] alanlarıyla
     PHOENIX modeli indirilir), istenirse dalgaboyu aralığı girin.
  2. **Yöntem seçimi** — "Hangi yöntemle devam etmek istiyorsun?": CCF veya BF
     işaretlenir; seçime göre ilgili parametre alanları görünür.
  3. **Hesapla** — sonuç metni pencerede, uyum grafiği pencereye gömülü olarak
     gösterilir; aynı anda `result_CCF.txt`/`result_BF.txt` ve
     `result_CCF.png`/`result_BF.png` diske kaydedilir.
- tkinter/ekran yoksa aynı akış **terminal soru-cevap sihirbazı** olarak
  çalışır (dosya yolları, yöntem seçimi `[1] CCF / [2] BF`, parametreler,
  isteğe bağlı barycentric düzeltme) ve aynı çıktı dosyaları üretilir.

Her analiz sonunda otomatik üretilen çıktılar (CLI modunda da geçerli):

| Dosya | İçerik |
|---|---|
| `result_CCF.txt` / `result_BF.txt` | Girdi dosyaları + RV ± hata (BF'de bileşen başına, genlik/sigma ve ışık oranıyla) |
| `result_CCF.png` / `result_BF.png` | Uyum (fit) grafiği: veri + Gauss modeli |

### 1) Kendi kendini test (veri gerekmez)

```bash
python rv_analysis.py demo --plot demo.png
```

Sentetik bir SB2 tayfı üretir (RV1 = −80, RV2 = +120 km/s, bilinen ışık oranı),
aynı veriyi hem BF hem CCF ile çözer ve gerçek değerlerle karşılaştırır:

```
--- BF yöntemi ---
  Bileşen 1: RV =  -80.010 ± 0.035 km/s
  Bileşen 2: RV =  119.982 ± 0.041 km/s
--- CCF yöntemi (aynı veri) ---
  Bileşen 1: RV =  -79.804 ± 0.130 km/s
  Bileşen 2: RV =  119.426 ± 0.217 km/s
```

### 2) CCF — çalıştaydaki ESPRESSO örneği

```bash
python rv_analysis.py ccf \
    --spectrum ESPRESSO_S2D_BLAZE_A.fits --format s2d \
    --teff 6628 --logg 4.251 --feh 0.17 \
    --rv-min -20 --rv-max 100 --rv-step 0.5 \
    --plot ccf.png --output ccf_sonuc.txt
```

Tüm echelle basamakları için CCF hesaplanır, toplanır, normalize edilir ve
Gauss uyumundan `RV ± hata` verilir (çalıştay defterindeki akışın birebir
karşılığı).

### 3) BF — tezdeki SB2 uygulaması

Tezdeki akış: IRAF ile indirgenmiş, normalize edilmiş ve birleştirilmiş
5000–5500 Å tayfı + sıcaklığa göre seçilmiş referans yıldız şablonu
(Gaia FGK Benchmark Stars vb.):

```bash
python rv_analysis.py bf \
    --spectrum tayf_5000_5500.txt \
    --template sablon_referans.txt \
    --wave-min 5000 --wave-max 5500 \
    --vel-range 500 --components 2 \
    --plot bf.png --output bf_sonuc.txt
```

Çıktıda her bileşen için `RV ± hata`, Gauss genliği ve sigma (vsini göstergesi),
ayrıca BF tepe alanlarından yaklaşık **ışık katkısı oranı** verilir. Zamana
yayılmış tayf serisinde her tayfa aynı komutu uygulayıp çıktıları birleştirerek
dikine hız eğrisini (PyWD2015'e girdi olacak biçimde) oluşturabilirsiniz.

### Barycentric düzeltme

Tezdeki "Güneş Sistemi kütle merkezine indirgeme" adımı için koordinat ve
gözlem zamanı verin (her iki alt komutta da çalışır):

```bash
python rv_analysis.py bf --spectrum tayf.txt --template sablon.txt \
    --components 2 \
    --ra 123.456 --dec -12.345 --obstime 2024-12-03T02:30:00 --site tug
```

`--site` astropy gözlemevi adıdır (`tug` = TÜBİTAK Ulusal Gözlemevi,
`paranal`, `lapalma`...). Düzeltme ölçülen RV'ye **eklenir**.

## Önemli parametreler (BF)

| Parametre | Anlamı | Öneri |
|---|---|---|
| `--vel-range` | BF penceresi ±[km/s] | Beklenen K1+K2+Vγ'yi rahat kapsasın (değen çiftlerde 400–600) |
| `--dv` | Hız adımı | Boş bırakın; verinin piksel ölçeğinden alınır |
| `--svd-rcond` | SVD kesme eşiği | Gürültülü veride büyütün (1e-2), yüksek S/N'de küçültün (1e-4) |
| `--smooth` | BF yumuşatma FWHM | Tayf çözünürlüğü civarı (R=20000 için ~15 km/s) |
| `--components` | Gauss sayısı | SB1=1, SB2=2 |
| `--min-sep` | İki tepenin asgari ayrıklığı | Harmanlanmış evrelerde küçültmeyin; o tayfı atlamak daha sağlıklı |

## Kaynaklar

- Rucinski, S. M. (1992, 2002, 2012) — Broadening Function yöntemi
- Mayor & Queloz (1995), *Nature* 378, 355 — CCF ile 51 Peg b keşfi
- Between the Lines Workshop 2024, E. Sedaghati (ESO) — CCF uygulaması
- MAK_Tez, Bölüm 2.2 ve 3.2 — BF yöntemi ve TIC 142587827 / TIC 82224114 uygulaması
