import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATASET
# ============================================
print("="*60)
print("HEART DISEASE PREDICTION - LOGISTIC REGRESSION")
print("="*60)

# Load dataset dari file CSV kamu
# GANTI 'heart_disease_uci.csv' sesuai dengan nama file kamu
df = pd.read_csv('heart_disease_uci.csv')

print(f"\n📊 Dataset Shape Awal: {df.shape}")
print(f"📊 Kolom: {list(df.columns)}")
print(f"\n🔍 Info Dataset:")
print(df.info())

# ============================================
# 2. DATA PREPROCESSING
# ============================================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Ubah target menjadi binary (0 = tidak sakit, 1 = sakit)
# num: 0 = tidak sakit, 1-4 = ada penyakit jantung (berbagai tingkat keparahan)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

print(f"\n✅ Target Distribution (SEBELUM cleaning):")
print(df['target'].value_counts())

# Drop kolom yang tidak diperlukan
columns_to_drop = ['id', 'dataset', 'num']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Encode kolom kategorikal
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

print(f"\n🔄 Encoding kolom kategorikal...")
le = LabelEncoder()
for col in categorical_columns:
    if col in df.columns:
        # Isi missing values dengan modus sebelum encoding
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = le.fit_transform(df[col].astype(str))

# Handle missing values di kolom numerik
print(f"\n🔍 Missing values sebelum cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Isi missing values dengan median
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Drop baris yang masih memiliki missing values (jika ada)
df = df.dropna()

print(f"\n✅ Dataset Shape Setelah Cleaning: {df.shape}")
print(f"✅ Target Distribution (SETELAH cleaning):")
print(df['target'].value_counts())
print(f"\nNo Heart Disease (0): {(df['target']==0).sum()}")
print(f"Heart Disease (1): {(df['target']==1).sum()}")

# ============================================
# 3. PREPARE DATA
# ============================================
print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📌 Training Set: {X_train.shape[0]} samples")
print(f"📌 Testing Set: {X_test.shape[0]} samples")

# Feature Scaling (penting untuk Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed")

# ============================================
# 4. TRAIN LOGISTIC REGRESSION MODEL
# ============================================
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Create and train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("✅ Logistic Regression model trained successfully!")

# ============================================
# 5. MAKE PREDICTIONS
# ============================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Predictions
y_pred = model.predict(X_test_scaled)

# ============================================
# 6. OUTPUT 1: ACCURACY
# ============================================
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 AKURASI MODEL: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================
# 7. OUTPUT 2: CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 CONFUSION MATRIX:")
print(cm)
print(f"\nTrue Negative (TN): {cm[0,0]} - Pasien TIDAK sakit, diprediksi TIDAK sakit")
print(f"False Positive (FP): {cm[0,1]} - Pasien TIDAK sakit, diprediksi SAKIT")
print(f"False Negative (FN): {cm[1,0]} - Pasien SAKIT, diprediksi TIDAK sakit")
print(f"True Positive (TP): {cm[1,1]} - Pasien SAKIT, diprediksi SAKIT")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease (0)', 'Disease (1)'],
            yticklabels=['No Disease (0)', 'Disease (1)'])
plt.title('Confusion Matrix - Heart Disease Prediction', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n💾 Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# ============================================
# 8. OUTPUT 3: CLASSIFICATION REPORT
# ============================================
print(f"\n📋 CLASSIFICATION REPORT:")
print("="*60)
print(classification_report(y_test, y_pred, 
                          target_names=['No Disease (0)', 'Disease (1)']))

# Extract metrics for detailed view
report = classification_report(y_test, y_pred, output_dict=True)

# Gunakan key string sesuai dengan target_names
try:
    precision_0 = report['No Disease (0)']['precision']
    recall_0 = report['No Disease (0)']['recall']
    f1_0 = report['No Disease (0)']['f1-score']
    
    precision_1 = report['Disease (1)']['precision']
    recall_1 = report['Disease (1)']['recall']
    f1_1 = report['Disease (1)']['f1-score']
except KeyError:
    # Fallback ke key numerik jika string tidak tersedia
    precision_0 = report['0']['precision']
    recall_0 = report['0']['recall']
    f1_0 = report['0']['f1-score']
    
    precision_1 = report['1']['precision']
    recall_1 = report['1']['recall']
    f1_1 = report['1']['f1-score']

# ============================================
# 9. OUTPUT 4: INTERPRETASI HASIL
# ============================================
print("\n" + "="*60)
print("INTERPRETASI HASIL")
print("="*60)

print(f"""
🎯 PERFORMA MODEL SECARA KESELURUHAN:
   • Akurasi: {accuracy*100:.2f}%
   • Artinya: Dari {len(y_test)} pasien dalam data testing, model berhasil 
     memprediksi dengan benar sebanyak {int(accuracy*len(y_test))} pasien 
     ({accuracy*100:.2f}%).

📊 CONFUSION MATRIX:
   • True Negative (TN) = {cm[0,0]}: Pasien TIDAK sakit, diprediksi TIDAK sakit ✅
   • False Positive (FP) = {cm[0,1]}: Pasien TIDAK sakit, diprediksi SAKIT ❌
   • False Negative (FN) = {cm[1,0]}: Pasien SAKIT, diprediksi TIDAK sakit ❌❌ (Bahaya!)
   • True Positive (TP) = {cm[1,1]}: Pasien SAKIT, diprediksi SAKIT ✅

⚠️ ANALISIS KESALAHAN:
   • Model salah memprediksi {cm[0,1]} kasus sebagai sakit (False Positive)
   • Model gagal mendeteksi {cm[1,0]} kasus penyakit jantung (False Negative)
   • False Negative sangat berbahaya dalam kasus medis karena pasien yang 
     sebenarnya sakit tidak terdeteksi dan tidak mendapat penanganan!

📈 METRIK UNTUK KELAS "TIDAK SAKIT" (0):
   • Precision: {precision_0:.2f} → Dari semua prediksi "tidak sakit", {precision_0*100:.1f}% benar
   • Recall: {recall_0:.2f} → Dari semua pasien yang tidak sakit, {recall_0*100:.1f}% terdeteksi
   • F1-Score: {f1_0:.2f} → Keseimbangan precision dan recall

📈 METRIK UNTUK KELAS "SAKIT" (1):
   • Precision: {precision_1:.2f} → Dari semua prediksi "sakit", {precision_1*100:.1f}% benar
   • Recall: {recall_1:.2f} → Dari semua pasien yang sakit, {recall_1*100:.1f}% terdeteksi
   • F1-Score: {f1_1:.2f} → Keseimbangan precision dan recall

💡 PENJELASAN METRIK PENTING:
   • PRECISION: Seberapa tepat model saat memprediksi kelas tertentu?
   • RECALL (Sensitivity): Seberapa baik model menangkap semua kasus positif?
   • F1-SCORE: Harmonic mean dari precision dan recall (semakin tinggi semakin baik)
   • Dalam kasus medis, RECALL untuk kelas positif sangat penting karena 
     kita tidak ingin melewatkan pasien yang sakit!

💡 KESIMPULAN:
   Model Logistic Regression menunjukkan performa {'BAIK' if accuracy > 0.75 else 'CUKUP'} 
   dengan akurasi {accuracy*100:.2f}%. {'Model ini dapat digunakan sebagai alat screening awal untuk deteksi penyakit jantung, namun tetap memerlukan validasi medis profesional.' if accuracy > 0.75 else 'Model perlu ditingkatkan untuk aplikasi klinis yang lebih andal.'}
   
   {'⚠️ Perhatian: Recall untuk kelas sakit cukup rendah, artinya model masih melewatkan beberapa kasus positif. Dalam aplikasi medis nyata, ini berbahaya!' if recall_1 < 0.8 else '✅ Recall tinggi menunjukkan model baik dalam mendeteksi pasien yang sakit.'}

📌 REKOMENDASI:
   1. Model ini cocok untuk SCREENING AWAL, bukan diagnosis final
   2. Semua prediksi positif harus dikonfirmasi dengan pemeriksaan medis lanjutan
   3. Untuk meningkatkan performa, bisa dicoba:
      - Feature engineering (kombinasi fitur)
      - Hyperparameter tuning
      - Algoritma lain (Random Forest, XGBoost)
      - Mengatasi class imbalance jika ada
""")

# ============================================
# 10. FEATURE IMPORTANCE
# ============================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

# Get feature importance from coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n🔍 Top 10 Fitur Paling Berpengaruh:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Top 15 Feature Importance in Logistic Regression', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n💾 Feature importance saved as 'feature_importance.png'")
plt.show()

print("\n" + "="*60)
print("✅ ANALISIS SELESAI!")
print("="*60)
print("\n📁 File yang tersimpan:")
print("   • confusion_matrix.png")
print("   • feature_importance.png")
print("\n💡 Tips untuk Laporan:")
print("   • Screenshot output ini untuk laporan kamu")
print("   • Jelaskan perbedaan Precision vs Recall")
print("   • Diskusikan kenapa False Negative berbahaya dalam kasus medis")
print("   • Bandingkan dengan hasil penelitian lain (biasanya 75-85%)")