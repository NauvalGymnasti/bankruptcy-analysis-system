# Membangun Model Machine Learning

Repository:
https://github.com/NauvalGymnasti/bankruptcy-analysis-system.git

## Deskripsi
Tahap ini melatih model machine learning menggunakan dataset hasil preprocessing.  
Pelatihan dilakukan menggunakan MLflow Tracking UI untuk mencatat eksperimen dan artefak model.

## Struktur Folder
Membangun_Model/
├── modelling.py
├── modelling_tuning.py
├── namadataset_preprocessing/
├── screenshoot_dashboard.jpg
├── screenshoot_artifak.jpg
├── requirements.txt
├── DagsHub.txt

## Tools & Library
- Scikit-learn
- MLflow (autolog & manual logging)
- Pandas, NumPy

## Logging
Tracking hasil pelatihan dilakukan melalui MLflow UI (`mlflow ui`).

## Output
- Artefak model tersimpan di folder `mlruns/`
- Metrik pelatihan (accuracy, precision, recall, f1-score, dsb)
