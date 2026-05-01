# Favorita Store Sales Forecasting: Multi-Model Pipeline

Dieses Projekt befasst sich mit der Vorhersage von Verkaufszahlen für die ecuadorianische Supermarktkette **Favorita**. Die zentrale Herausforderung liegt in der Verarbeitung von volatilen Zeitreihen, die stark von externen Faktoren wie Feiertagen, Ölpreisen und lokalen Events beeinflusst werden.

## 🛠 Methodik & Umsetzung

Anstatt auf ein einzelnes Modell zu setzen, wurde eine skalierbare **Vergleichs-Pipeline** entwickelt, die verschiedene mathematische Ansätze evaluiert:

### 1. Daten-Architektur & Performance
* **High-Speed Processing:** Einsatz von **Polars** für das Data Wrangling, um eine effiziente Verarbeitung zu gewährleisten.
* **Storage:** Nutzung des **Parquet-Formats** zur optimierten Speicherung und schnellen Bereitstellung der Trainings- und Validierungsdaten.
* **Modularität:** Strikte Trennung der Logik (`03_src/`) von der Analyse (`04_notebooks/`), um die Wiederverwendbarkeit des Codes zu sichern.

### 2. Multivariate Feature Engineering
Die Zeitreihen wurden um kontextbezogene Dimensionen erweitert, um den Modellen tieferes Wissen über die Marktgegebenheiten zu vermitteln:
* **Ökonomische Indikatoren:** Integration des täglichen Ölpreises als Proxy für die Kaufkraft.
* **Kalendarische Ereignisse:** Mapping von nationalen und lokalen Feiertagen inklusive "Pre- & Post-Holiday"-Effekten.
* **Zeitreihen-Dynamik:** Generierung von rollierenden Statistiken (Rolling Windows) und Zeit-Lags, um saisonale Trends (Wochentage, Monatsenden) explizit abzubilden.

### 3. Evaluierungs-Framework
Implementierung eines hybriden Benchmarking-Systems, das drei Modell-Generationen gegenüberstellt:
* **Gradient Boosted Trees (XGBoost, LightGBM):** Optimiert für tabellarische Daten unter Ausnutzung des manuellen Feature Engineerings.
* **Deep Learning (PatchTST, NHITS):** Einsatz von Transformer-basierten Architekturen, die multivariate Abhängigkeiten über "Patches" (Zeitfenster) hinweg lernen.
* **Probabilistische & Statistische Modelle (Prophet, SARIMAX):** Einsatz als Baseline zur Validierung der Modell-Komplexität.

## 📂 Repository-Struktur

```text
├── 01_business_understanding/  # Projektdefinition und Zielsetzung
├── 02_data/                    # Strukturierte Ablage (Raw, Final, Results)
├── 03_src/                     # Kern-Logik (Features, Utilities, Config)
├── 04_notebooks/               # Workflow von EDA bis Evaluierung
├── requirements_clean.txt      # Projekt-Abhängigkeiten
└── README.md                   # Projektdokumentation