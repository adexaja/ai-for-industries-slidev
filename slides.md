---
# a clean, bold look
theme: seriph
colorSchema: auto
background: https://images.unsplash.com/photo-1518779578993-ec3579fee39f?q=80&w=1974&auto=format&fit=crop
class: text-center
transition: slide-left
mdc: true
fonts:
  sans: Inter
  mono: Fira Code
title: AI for Industries — Beyond LLM
info: |
  Practical, non-LLM AI that drives business outcomes. With demos and architecture patterns.
---

# AI for Industries 


Rezki Nasrullah · AI & Software Engineering

@rezki.dev

<div class="abs-br m-6 text-xl opacity-70">
  Press → to begin
</div>

<!--
Notes
- Opening: AI jauh lebih luas dari chatbot. Fokus dampak bisnis.
- Janji: contoh konkret + demo singkat.
-->

---
layout: image-right
image: https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?q=80&w=1935&auto=format&fit=crop
class: text-left
---

# Big Idea

<span class="text-5xl font-700">AI ≠ ChatGPT</span>

Di industri, AI adalah sistem yang **belajar dari data** untuk membantu keputusan.

- Mengurangi downtime mesin
- Mengoptimalkan stok & logistik
- Mendeteksi fraud & anomali
- Mempercepat diagnosa visual
- Membuat laporan otomatis

Notes:
- Align with business outcomes first; tools later.

---
layout: two-cols
layoutClass: gap-8 items-center
transition: slide-up
---

# Agenda

1. Landscape: AI di luar LLM
2. Use cases lintas industri
3. Lightning demos (tabular, vision, forecasting)
4. LLM sebagai interface
5. Arsitektur modern + praktik implementasi
6. Mitos vs Fakta + langkah awal

---

# Landscape AI (ringkas)

```txt {monaco}
Artificial Intelligence
├─ Machine Learning (tabular, time-series)
│  ├─ Supervised (regresi/klasifikasi)
│  ├─ Unsupervised (clustering)
│  └─ Reinforcement Learning
├─ Computer Vision (deteksi objek, OCR, inspeksi)
├─ Natural Language Processing (NLP)
│  └─ LLM (ChatGPT, Claude, dsb)
└─ Generative AI (gambar, audio, video)
```

<v-click>
<div class="mt-6 text-xl">
LLM adalah subset NLP — alat bantu penjelas & interface, bukan pengganti semua model.
</div>
</v-click>

---
layout: two-cols
layoutClass: gap-10
---

# Peta Manfaat Industri

- Manufaktur: Predictive Maintenance → kurangi downtime
- Retail: Forecast Demand → efisiensi stok & promo yang tepat
- Healthcare: Image Diagnosis → triase lebih cepat
- Finance: Fraud Detection → cegah transaksi curang
- Transportasi: Route Optimization → hemat biaya & waktu

::right::

```mermaid
flowchart TB

A[Data Sensor / Transaksi / Citra] --> B[Feature Engineering]
B --> C{Model}
C -->|Prediksi| D[Keputusan]
D -->|A/B & Monitor| E[ROI]
```

---

# Klasifikasi menurut tujuan

- Descriptive: apa yang terjadi
- Predictive: apa yang akan terjadi
- Prescriptive: tindakan terbaik
- Generative: membuat konten baru



---

## Klasifikasi: Berdasarkan Metode
- **Supervised Learning** (berlabel): deteksi cacat, scoring risiko
- **Unsupervised Learning** (tanpa label): segmentasi pelanggan
- **Reinforcement Learning**: kontrol robot, optimasi kebijakan


---

## Demo 1 — Tabular: Predictive Maintenance (Non‑LLM)
**Kasus**: sensor (suhu, getaran, arus) → prediksi potensi fault.

**Goal**: klasifikasi *fault type* / *at-risk* untuk scheduling maintenance.

```python {all|1-7|9-14|16-23|all}
# Random Forest (baseline)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/faults.csv"
Xy = pd.read_csv(url)

target = 'Class' if 'Class' in Xy.columns else Xy.columns[-1]
X = Xy.drop(columns=[target])
y = Xy[target]

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
mdl = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1).fit(Xtr,ytr)
print(classification_report(yte, mdl.predict(Xte)))
```

<small class="op60">Mulai dari baseline sederhana → iterasi fitur → kalibrasi → monitoring</small>

---

## Demo 2 — Computer Vision: Inspeksi Kualitas
**Kasus**: deteksi cacat produk (retak, gores, lubang) dari citra pabrik.

**Goal**: kurangi inspeksi manual, *early reject* otomatis.

```python {all|1-6|8-15|17-22|all}
# OpenCV Edge (baseline)
import cv2, numpy as np, urllib.request
import matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Checkerboard_pattern.svg/512px-Checkerboard_pattern.svg.png"
urllib.request.urlretrieve(url, "sample.png")
img = cv2.imread("sample.png")

edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
plt.imshow(edges, cmap='gray'); plt.axis('off'); plt.title('Edge Map')
plt.show()
```

<v-click>
<div class="mt-4 op80">Naikkan tingkat: fine-tune ResNet/YOLO + data augmentasi + active learning.</div>
</v-click>

---

## Demo 3 — Time Series Forecasting: Penjualan
**Kasus**: prediksi penjualan mingguan untuk perencanaan stok & promosi.

**Goal**: kurangi kehabisan stok / stok berlebih.

```python {all|1-6|8-16|18-24|all}
# ARIMA baseline
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
series = pd.read_csv(url)['Passengers']

fit = ARIMA(series, order=(5,1,0)).fit()
fc = fit.forecast(steps=12)

plt.plot(series,label='history')
plt.plot(range(len(series), len(series)+12), fc, label='forecast')
plt.legend(); plt.title('12-month baseline forecast'); plt.show()
```

---
layout: two-cols
layoutClass: gap-10 items-center
---

# LLM sebagai Interface

- Menyusun laporan dari angka/plot
- Q&A atas hasil anomaly/forecast
- Natural language query untuk dashboard

::right::

```md
Anda adalah analis ritel. Jelaskan hasil forecast ke Head of Ops.
Soroti risiko kehabisan stok, reorder point, dan rencana promo.
Data: [120,133,140,160,180,175,190,200,210,205,220,230]
Output: 5 bullet, bahasa bisnis, langsung to-do.
```

<small class="op60">LLM = lapisan penjelas dan antarmuka — model inti tetap non‑LLM</small>

---
layout: image
image: /chatgpt.png
--- 
# Result 
---

# Arsitektur Modern AI di Industri

```mermaid
graph LR

A["Raw Data<br/>(sensor/transaksi/citra)"] --> B["ETL &amp; Feature<br/>(batch/stream)"]
B --> C["Training ML/CV/RL<br/>+ Tracking (MLflow/W&amp;B)"]
C --> D["Evaluation<br/>(offline + online)"]
D --> E["Deployment<br/>(API/Batch/Edge/Stream)"]
E --> F["Monitoring<br/>(drift, data quality, alerting)"]
F --> G["Interface<br/>(Dashboard / Chat via LLM)"]
```

<small class="op60">Governance: versi model, keamanan data, audit trail, reproducibility</small>

---
layout: two-cols
layoutClass: gap-12
---

## Praktik Implementasi (Ringkas)
- Mulai dari **masalah bisnis** → turunkan ke metrik & data
- Gunakan **baseline sederhana** dulu (rule/ARIMA/RF)
- **Iterasi cepat**: data → fitur → model → evaluasi → deploy kecil
- **Observability**: logging, tracing, model drift, data quality
- **MLOps**: reproducibility, CI/CD, registry, monitoring

::right::

## Mitos vs Fakta
| Mitos | Fakta |
|---|---|
| AI = ChatGPT | LLM hanya subset NLP |
| AI untuk big tech saja | Banyak OSS & cloud credits |
| AI menggantikan manusia | AI memperkuat keputusan manusia |
| Harus deep learning dulu | Seringnya baseline klasik sudah ROI |

---
layout: center
class: text-center
---

# Quickstart Colab


---
layout: center
---

## Takeaways
- **AI lebih luas dari LLM**: tabular, vision, time-series, RL, optimization
- Mulai dengan **baseline yang masuk akal** dan data yang ada
- Gunakan **LLM sebagai antarmuka/penjelas**, bukan satu-satunya solusi

---
layout: center
class: text-center
---

## **Terima kasih**

## Q&A
