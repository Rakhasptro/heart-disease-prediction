# Heart Disease Prediction using Logistic Regression

Proyek Machine Learning untuk memprediksi penyakit jantung menggunakan algoritma **Logistic Regression** dengan dataset UCI Heart Disease.

## 📋 Deskripsi

Proyek ini merupakan implementasi model klasifikasi biner untuk memprediksi apakah seorang pasien menderita penyakit jantung atau tidak berdasarkan berbagai fitur medis seperti:
- Umur (age)
- Jenis kelamin (sex)
- Tipe nyeri dada (cp - chest pain)
- Tekanan darah (trestbps)
- Kolesterol (chol)
- Detak jantung maksimum (thalach)
- Dan fitur medis lainnya

## 🎯 Tujuan Proyek

- Membangun model prediksi penyakit jantung dengan akurasi tinggi
- Menganalisis fitur-fitur yang paling berpengaruh terhadap penyakit jantung
- Menghasilkan evaluasi model yang komprehensif (accuracy, confusion matrix, classification report)

## 📊 Dataset

- **Sumber**: UCI Machine Learning Repository - Heart Disease Dataset
- **Total Data**: 920 records
- **Fitur**: 13 fitur medis + 1 target
- **Target**: 
  - 0 = Tidak ada penyakit jantung
  - 1 = Ada penyakit jantung (tingkat 1-4 digabung)
- **Lokasi Data**: Cleveland, Hungary, Switzerland, VA Long Beach

### Fitur Dataset:
| Fitur | Deskripsi | Tipe |
|-------|-----------|------|
| age | Usia pasien | Numerik |
| sex | Jenis kelamin | Kategorikal |
| cp | Tipe nyeri dada | Kategorikal |
| trestbps | Tekanan darah saat istirahat | Numerik |
| chol | Kolesterol serum | Numerik |
| fbs | Gula darah puasa > 120 mg/dl | Boolean |
| restecg | Hasil elektrokardiografi | Kategorikal |
| thalach | Detak jantung maksimum | Numerik |
| exang | Angina akibat olahraga | Boolean |
| oldpeak | ST depression | Numerik |
| slope | Slope dari ST segment | Kategorikal |
| ca | Jumlah pembuluh darah utama | Numerik |
| thal | Thalassemia | Kategorikal |
| num/target | Target (0=sehat, 1=sakit) | Binary |

## 🛠️ Teknologi yang Digunakan

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine Learning
  - `matplotlib` - Visualisasi
  - `seaborn` - Visualisasi statistik
  - `openpyxl` - Read Excel files

## 📦 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/Rakhasptro/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

**ATAU** gunakan requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Pastikan file `heart_disease_uci.csv` atau `heart_disease_uci.xlsx` berada di folder yang sama dengan `main.py`.

## 🚀 Cara Menjalankan

```bash
python main.py
```

### Alternatif:
```bash
python3 main.py
```

## 📈 Output Program

Program akan menghasilkan:

### 1. **Akurasi Model**
```
🎯 AKURASI MODEL: 0.8500 (85.00%)
```

### 2. **Confusion Matrix**
- Visualisasi dalam bentuk heatmap
- File: `confusion_matrix.png`

### 3. **Classification Report**
```
              precision    recall  f1-score   support

No Disease       0.87      0.84      0.85        92
   Disease       0.83      0.86      0.85        92

  accuracy                           0.85       184
```

### 4. **Interpretasi Detail**
- Penjelasan lengkap dalam Bahasa Indonesia
- Analisis kesalahan model
- Rekomendasi improvement

### 5. **Feature Importance**
- Visualisasi fitur paling berpengaruh
- File: `feature_importance.png`

## 🔍 Metodologi

### 1. **Data Preprocessing**
- Handling missing values
- Encoding variabel kategorikal (Label Encoding)
- Normalisasi fitur (StandardScaler)
- Split data (80% training, 20% testing)

### 2. **Model Training**
- Algoritma: Logistic Regression
- Parameter: `max_iter=1000`, `random_state=42`

### 3. **Model Evaluation**
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-Score
- Feature Importance Analysis

## 📊 Hasil Evaluasi

### Model Performance:
| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| Precision (Class 0) | ~87% |
| Recall (Class 0) | ~84% |
| Precision (Class 1) | ~83% |
| Recall (Class 1) | ~86% |
| F1-Score | ~85% |

*Note: Hasil dapat bervariasi tergantung random split*

## 💡 Insight Penting

1. **False Negative** sangat berbahaya dalam kasus medis karena pasien yang sakit tidak terdeteksi
2. Model ini cocok untuk **screening awal**, bukan diagnosis final
3. Semua prediksi positif harus dikonfirmasi dengan pemeriksaan medis lanjutan

## 🔮 Improvement Ideas

- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Feature engineering (kombinasi fitur)
- [ ] Coba algoritma lain (Random Forest, XGBoost, SVM)
- [ ] Handle class imbalance (SMOTE)
- [ ] Cross-validation untuk evaluasi lebih robust
- [ ] Deploy model sebagai web app

## 📚 Referensi

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Logistic Regression Theory](https://en.wikipedia.org/wiki/Logistic_regression)

## 👨‍💻 Author

**Rakha Adi Saputro**
- GitHub: [@Rakhasptro](https://github.com/Rakhasptro)
- Email: rakhaadi24@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset dari UCI Machine Learning Repository
- Dosen Mata Kuliah Machine Learning
- Komunitas Data Science Indonesia

---

⭐ **Jika project ini membantu, jangan lupa kasih star ya!** ⭐
