from google_play_scraper import reviews, Sort
import pandas as pd
import re, string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datetime import datetime
import math
from collections import Counter
import matplotlib.pyplot as plt

print("=== IMPLEMENTASI ALGORITMA NAIVE BAYES UNTUK ANALISIS SENTIMEN ===")
print("=== ULASAN SHOPEE PADA GOOGLE PLAY STORE ===")
print(f"Waktu mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. WEB SCRAPING DATA
print("\n1. WEB SCRAPING DATA")
print("="*50)
print("   - Target: Aplikasi Shopee (com.shopee.id)")
print("   - Jumlah ulasan: 1000")
print("   - Bahasa: Indonesia")
print("   - Urutan: Terbaru")

result, _ = reviews(
    'com.shopee.id',
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=1000
)

# Buat DataFrame dengan lebih banyak informasi
df_raw = pd.DataFrame(result)
print(f"   ✓ Berhasil mengambil {len(df_raw)} ulasan dari Google Play Store")

# Pilih kolom yang diperlukan dan tambahkan informasi
df = df_raw[['content', 'score', 'at', 'userName', 'thumbsUpCount']].copy()
df['kategori'] = df['score'].apply(lambda x: 'Positif' if x >= 4 else 'Negatif')

print(f"   ✓ Data mentah siap diproses")
print(f"   - Ulasan Positif (score ≥4): {len(df[df['kategori'] == 'Positif'])}")
print(f"   - Ulasan Negatif (score <4): {len(df[df['kategori'] == 'Negatif'])}")

# 2. TEXT PREPROCESSING
print("\n2. TEXT PREPROCESSING")
print("="*50)

# 2a. Case folding & remove punctuation/numerik
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[\d'+string.punctuation+']+', ' ', text)
    return text.strip()

# 2b. Tokenization
def tokenize(text):
    return text.split()

# 2c. Stopword removal
sw_factory = StopWordRemoverFactory()
stopword = sw_factory.create_stop_word_remover()
def remove_stopwords(text):
    return stopword.remove(text)

# 2d. Stemming
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()
def stem(text):
    return stemmer.stem(text)

# 2e. Gabungkan preprocessing dengan tracking
def preprocess_with_steps(text):
    original = text
    step1 = clean_text(text)
    step2 = remove_stopwords(step1)
    step3 = stem(step2)
    return {
        'original': original,
        'after_cleaning': step1,
        'after_stopword_removal': step2,
        'after_stemming': step3,
        'tokens': tokenize(step3) if step3.strip() else []
    }

print("a. Case Folding")
print("   - Mengubah semua huruf menjadi huruf kecil")
print("   - Menghapus tanda baca dan karakter numerik")

print("b. Tokenization")
print("   - Memecah teks menjadi token-token kata")

print("c. Filtering (Stopword Removal)")
print("   - Menghapus kata-kata yang tidak relevan")

print("d. Stemming")
print("   - Mengubah kata berimbuhan menjadi kata dasar")

# Proses dengan tracking langkah-langkah
preprocessing_steps = []
for idx, text in enumerate(df['content']):
    steps = preprocess_with_steps(text)
    steps['index'] = idx
    preprocessing_steps.append(steps)
    if (idx + 1) % 200 == 0:
        print(f"   ✓ Diproses: {idx + 1}/1000 ulasan")

df_preprocessing = pd.DataFrame(preprocessing_steps)
df['clean'] = df_preprocessing['after_stemming']

# Filter data yang kosong setelah preprocessing
df_clean = df[df['clean'].str.strip() != ''].copy().reset_index(drop=True)
print(f"   ✓ Text preprocessing selesai")
print(f"   ✓ Data bersih: {len(df_clean)} ulasan (dari {len(df)} ulasan)")

# 3. TF-IDF (TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY)
print("\n3. TF-IDF (TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY)")
print("="*60)

# 3a. Manual TF-IDF calculation untuk contoh (mengikuti jurnal)
def calculate_manual_tfidf(documents, max_examples=5):
    """
    Menghitung TF-IDF secara manual untuk beberapa dokumen contoh
    mengikuti metodologi dari jurnal
    """
    # Tokenize semua dokumen
    all_tokens = []
    doc_tokens = []
    
    for doc in documents[:max_examples]:
        tokens = doc.split()
        doc_tokens.append(tokens)
        all_tokens.extend(tokens)
    
    # Hitung vocabulary
    vocab = list(set(all_tokens))
    vocab.sort()
    
    # Hitung document frequency untuk setiap term
    df_dict = {}
    for term in vocab:
        df_dict[term] = sum(1 for doc_token in doc_tokens if term in doc_token)
    
    # Hitung TF-IDF untuk setiap dokumen
    tfidf_results = []
    N = len(doc_tokens)  # Total dokumen
    
    for i, tokens in enumerate(doc_tokens):
        doc_tfidf = {}
        term_freq = Counter(tokens)
        max_tf = max(term_freq.values()) if term_freq else 1
        
        for term in vocab:
            # TF calculation: 0.5 + 0.5 * (tf / max_tf)
            tf = term_freq.get(term, 0)
            tf_normalized = 0.5 + 0.5 * (tf / max_tf) if tf > 0 else 0
            
            # IDF calculation: ln(N / df) + 1
            df = df_dict[term]
            idf = math.log(N / df) + 1 if df > 0 else 0
            
            # TF-IDF
            tfidf = tf_normalized * idf
            
            if tf > 0:  # Hanya simpan term yang ada di dokumen
                doc_tfidf[term] = {
                    'tf': tf,
                    'tf_normalized': tf_normalized,
                    'df': df,
                    'idf': idf,
                    'tfidf': tfidf
                }
        
        tfidf_results.append({
            'document_index': i,
            'original_text': documents[i],
            'tokens': tokens,
            'tfidf_scores': doc_tfidf
        })
    
    return tfidf_results, vocab, df_dict

# Hitung manual TF-IDF untuk 5 dokumen pertama
manual_tfidf_results, vocab_sample, df_dict = calculate_manual_tfidf(df_clean['clean'].tolist())

print("   - Menghitung Term Frequency (TF) dengan rumus: tf = 0.5 + 0.5 * (tf/max(tf))")
print("   - Menghitung Inverse Document Frequency (IDF) dengan rumus: idf = ln(N/df) + 1")
print("   - Menghitung TF-IDF = TF × IDF")
print(f"   ✓ Manual TF-IDF calculation selesai untuk {len(manual_tfidf_results)} dokumen contoh")

# 3b. Sklearn TF-IDF untuk seluruh dataset
vectorizer = TfidfVectorizer(max_features=1000, min_df=2)
X = vectorizer.fit_transform(df_clean['clean'])
y = df_clean['kategori']

# Dapatkan feature names untuk analisis
feature_names = vectorizer.get_feature_names_out()
tfidf_matrix = X.toarray()

print(f"   ✓ TF-IDF vectorization selesai: {X.shape[0]} dokumen, {X.shape[1]} fitur")

# 4. SPLITTING DATA
print("\n4. SPLITTING DATA")
print("="*30)

# 4a. Hold-Out (80:20)
print("a. Hold-Out Method")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   - Data training: {X_train.shape[0]} sampel (80%)")
print(f"   - Data testing: {X_test.shape[0]} sampel (20%)")

# 4b. K-Fold Cross Validation setup
print("b. K-Fold Cross Validation")
print(f"   - K = 10 fold")
print(f"   - Stratified sampling untuk menjaga proporsi kelas")

# 5. MULTINOMIAL NAIVE BAYES
print("\n5. MULTINOMIAL NAIVE BAYES")
print("="*40)
print("   - Algoritma klasifikasi untuk data tekstual")
print("   - Menggunakan asumsi independensi antar fitur")
print("   - Rumus: P(c|d) ∝ P(c) ∏ P(wi|c)")

# Training model Hold-Out
model_holdout = MultinomialNB()
model_holdout.fit(X_train, y_train)
y_pred_holdout = model_holdout.predict(X_test)

# Hitung metrik evaluasi Hold-Out
holdout_accuracy = accuracy_score(y_test, y_pred_holdout)
holdout_precision = precision_score(y_test, y_pred_holdout, pos_label='Positif', average='binary')
holdout_recall = recall_score(y_test, y_pred_holdout, pos_label='Positif', average='binary')
holdout_f1 = f1_score(y_test, y_pred_holdout, pos_label='Positif', average='binary')

# Confusion Matrix Hold-Out
cm_holdout = confusion_matrix(y_test, y_pred_holdout)
cr_holdout = classification_report(y_test, y_pred_holdout, digits=4, output_dict=True)

print("   ✓ Model Naive Bayes berhasil dilatih")

# K-Fold Cross Validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_accuracies = []
cv_precisions = []
cv_recalls = []
cv_f1_scores = []
cv_confusion_matrices = []
fold_results = []

print("   ✓ Memproses 10-fold cross validation...")
for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    
    model_fold = MultinomialNB()
    model_fold.fit(X_train_fold, y_train_fold)
    y_pred_fold = model_fold.predict(X_test_fold)
    
    acc = accuracy_score(y_test_fold, y_pred_fold)
    prec = precision_score(y_test_fold, y_pred_fold, pos_label='Positif', average='binary')
    rec = recall_score(y_test_fold, y_pred_fold, pos_label='Positif', average='binary')
    f1 = f1_score(y_test_fold, y_pred_fold, pos_label='Positif', average='binary')
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    
    cv_accuracies.append(acc)
    cv_precisions.append(prec)
    cv_recalls.append(rec)
    cv_f1_scores.append(f1)
    cv_confusion_matrices.append(cm)
    
    fold_results.append({
        'fold': fold + 1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'train_size': len(train_idx),
        'test_size': len(test_idx)
    })

# Hitung rata-rata dan standar deviasi
cv_acc_mean = np.mean(cv_accuracies)
cv_acc_std = np.std(cv_accuracies)
cv_prec_mean = np.mean(cv_precisions)
cv_prec_std = np.std(cv_precisions)
cv_rec_mean = np.mean(cv_recalls)
cv_rec_std = np.std(cv_recalls)
cv_f1_mean = np.mean(cv_f1_scores)
cv_f1_std = np.std(cv_f1_scores)

print("   ✓ 10-fold cross validation selesai")

# 6. HASIL DAN ANALISIS
print("\n6. HASIL DAN ANALISIS")
print("="*30)

print("\na. Hasil Hold-Out Validation (80:20)")
print("-" * 40)
print(f"Accuracy  : {holdout_accuracy:.4f} ({holdout_accuracy*100:.0f}%)")
print(f"Precision : {holdout_precision:.4f}")
print(f"Recall    : {holdout_recall:.4f}")
print(f"F1-Score  : {holdout_f1:.4f}")

print("\nConfusion Matrix:")
print("                 Prediksi")
print("Aktual    Negatif  Positif")
print(f"Negatif      {cm_holdout[0][0]:3d}      {cm_holdout[0][1]:3d}")
print(f"Positif      {cm_holdout[1][0]:3d}      {cm_holdout[1][1]:3d}")

print("\nb. Hasil 10-Fold Cross Validation")
print("-" * 40)
print(f"Accuracy  : {cv_acc_mean:.4f} ± {cv_acc_std:.4f} ({cv_acc_mean*100:.0f}% ± {cv_acc_std*100:.1f}%)")
print(f"Precision : {cv_prec_mean:.4f} ± {cv_prec_std:.4f}")
print(f"Recall    : {cv_rec_mean:.4f} ± {cv_rec_std:.4f}")
print(f"F1-Score  : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

# Tentukan metode terbaik dan tampilkan classification report
print("\nc. Perbandingan Metode")
print("-" * 25)
if holdout_accuracy > cv_acc_mean:
    best_method = "Hold-Out"
    print(f"✓ Hold-Out validation menghasilkan akurasi terbaik: {holdout_accuracy*100:.0f}%")
    print(f"  Lebih unggul {(holdout_accuracy - cv_acc_mean)*100:.0f}% dari K-Fold Cross Validation ({cv_acc_mean*100:.0f}%)")
    
    print(f"\nClassification Report (Hold-Out - Metode Terbaik):")
    print(classification_report(y_test, y_pred_holdout, digits=2))
else:
    best_method = "K-Fold Cross Validation"
    print(f"✓ K-Fold Cross Validation menghasilkan akurasi terbaik: {cv_acc_mean*100:.0f}%")
    print(f"  Lebih unggul {(cv_acc_mean - holdout_accuracy)*100:.0f}% dari Hold-Out ({holdout_accuracy*100:.0f}%)")
    
    # Untuk classification report K-fold, gabungkan semua prediksi
    all_y_true = []
    all_y_pred = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        model_fold = MultinomialNB()
        model_fold.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_fold.predict(X_test_fold)
        
        all_y_true.extend(y_test_fold.tolist())
        all_y_pred.extend(y_pred_fold.tolist())
    
    print(f"\nClassification Report (K-Fold - Metode Terbaik):")
    print(classification_report(all_y_true, all_y_pred, digits=2))

# TAMBAHAN: VISUALISASI GRAFIK PERBANDINGAN METRIK
print("\n Jika popup grafik sudah muncul, silahkan di close supaya bisa lanjut ke proses selanjutnya !")
print("\n   📊 Membuat grafik perbandingan metrik evaluasi...")

# Set up the figure and axis
plt.figure(figsize=(12, 8))

# Data untuk grafik
metrics = ['Accuracy', 'Precision', 'Recall', 'f1-score']
holdout_values = [holdout_accuracy, holdout_precision, holdout_recall, holdout_f1]
kfold_values = [cv_acc_mean, cv_prec_mean, cv_rec_mean, cv_f1_mean]

# Posisi bar
x = np.arange(len(metrics))
width = 0.35

# Membuat bar chart dengan pattern yang sesuai dengan gambar
fig, ax = plt.subplots(figsize=(12, 8))

# Bar untuk Hold-Out dengan pattern dots
bars1 = ax.bar(x - width/2, holdout_values, width, label='Hold-Out', 
               color='lightgray', edgecolor='black', hatch='...')

# Bar untuk 10-Fold CV dengan pattern diagonal lines
bars2 = ax.bar(x + width/2, kfold_values, width, label='10-Fold Cross Validation',
               color='lightgray', edgecolor='black', hatch='///')

# Menambahkan nilai di atas setiap bar
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    
    # Nilai untuk Hold-Out
    ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
            f'{holdout_values[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Nilai untuk K-Fold
    ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
            f'{kfold_values[i]:.2f}', ha='center', va='bottom', fontweight='bold')

# Kustomisasi grafik
ax.set_xlabel('Metrik Evaluasi', fontsize=12, fontweight='bold')
ax.set_ylabel('Nilai', fontsize=12, fontweight='bold')
ax.set_title('Perbandingan Akurasi, Precision, Recall, dan f1-score\nHold-Out vs 10-Fold Cross Validation', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 1.2)

# Menambahkan grid untuk kemudahan membaca
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

# Menyimpan grafik
chart_filename = f'comparison_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
plt.tight_layout()
plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"   ✓ Grafik perbandingan berhasil dibuat dan disimpan: {chart_filename}")

# Analisis sentimen
print("\nd. Analisis Sentimen")
print("-" * 20)
pos_count = len(df_clean[df_clean['kategori'] == 'Positif'])
neg_count = len(df_clean[df_clean['kategori'] == 'Negatif'])
pos_percentage = (pos_count / len(df_clean)) * 100

if pos_count > neg_count:
    sentiment_tendency = "POSITIF"
    print(f"✓ Sentimen pengguna aplikasi Shopee cenderung POSITIF")
    print(f"  - Ulasan Positif: {pos_count} ({pos_percentage:.1f}%)")
    print(f"  - Ulasan Negatif: {neg_count} ({100-pos_percentage:.1f}%)")
else:
    sentiment_tendency = "NEGATIF"
    print(f"✓ Sentimen pengguna aplikasi Shopee cenderung NEGATIF")
    print(f"  - Ulasan Negatif: {neg_count} ({100-pos_percentage:.1f}%)")
    print(f"  - Ulasan Positif: {pos_count} ({pos_percentage:.1f}%)")

# Analisis fitur penting
log_prob_pos = model_holdout.feature_log_prob_[1]  # Positif
log_prob_neg = model_holdout.feature_log_prob_[0]  # Negatif

feature_importance = log_prob_pos - log_prob_neg
feature_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance,
    'log_prob_positive': log_prob_pos,
    'log_prob_negative': log_prob_neg
})
feature_df = feature_df.sort_values('importance', ascending=False)

# 7. MENYIMPAN HASIL KE EXCEL
print(f"\n7. MENYIMPAN HASIL")
print("="*20)
excel_filename = f'sentiment_analysis_shopee_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Sheet 1: Data Mentah
    df_raw_export = df_raw[['content', 'score', 'at', 'userName', 'thumbsUpCount']].copy()
    df_raw_export['kategori_sentimen'] = df_raw_export['score'].apply(lambda x: 'Positif' if x >= 4 else 'Negatif')
    df_raw_export.to_excel(writer, sheet_name='Data_Mentah', index=False)
    
    # Sheet 2: Preprocessing Steps
    df_preprocessing_export = df_preprocessing.copy()
    if len(df_preprocessing_export) > 0:
        df_preprocessing_export['score'] = df['score'][:len(df_preprocessing_export)]
        df_preprocessing_export['kategori'] = df['kategori'][:len(df_preprocessing_export)]
    df_preprocessing_export.to_excel(writer, sheet_name='Preprocessing_Steps', index=False)
    
    # Sheet 3: Manual TF-IDF Calculation (Contoh 5 Dokumen)
    manual_tfidf_data = []
    for result in manual_tfidf_results:
        for term, scores in result['tfidf_scores'].items():
            manual_tfidf_data.append({
                'document_index': result['document_index'],
                'original_text': result['original_text'][:100] + '...' if len(result['original_text']) > 100 else result['original_text'],
                'term': term,
                'tf': scores['tf'],
                'tf_normalized': scores['tf_normalized'],
                'df': scores['df'],
                'idf': scores['idf'],
                'tfidf': scores['tfidf']
            })
    
    manual_tfidf_df = pd.DataFrame(manual_tfidf_data)
    manual_tfidf_df.to_excel(writer, sheet_name='Manual_TF-IDF_Calculation', index=False)
    
    # Sheet 4: TF-IDF Matrix (Sample)
    tfidf_sample_df = pd.DataFrame(
        tfidf_matrix[:20],  # 20 dokumen pertama
        columns=feature_names
    )
    tfidf_sample_df.insert(0, 'document_text', df_clean['clean'][:20].values)
    tfidf_sample_df.insert(1, 'kategori', df_clean['kategori'][:20].values)
    tfidf_sample_df.to_excel(writer, sheet_name='TF-IDF_Matrix_Sample', index=False)
    
    # Sheet 5: Hold-Out Results
    holdout_results_df = pd.DataFrame({
        'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Nilai': [holdout_accuracy, holdout_precision, holdout_recall, holdout_f1],
        'Persentase': [f'{holdout_accuracy*100:.2f}%', f'{holdout_precision*100:.2f}%', 
                      f'{holdout_recall*100:.2f}%', f'{holdout_f1*100:.2f}%']
    })
    
    # Confusion Matrix Hold-Out
    cm_holdout_df = pd.DataFrame(cm_holdout, 
                                index=['Actual_Negatif', 'Actual_Positif'],
                                columns=['Predicted_Negatif', 'Predicted_Positif'])
    
    # Classification Report Hold-Out
    cr_holdout_df = pd.DataFrame(cr_holdout).transpose()
    
    holdout_results_df.to_excel(writer, sheet_name='Hold-Out_Results', index=False, startrow=0)
    cm_holdout_df.to_excel(writer, sheet_name='Hold-Out_Results', startrow=7)
    cr_holdout_df.to_excel(writer, sheet_name='Hold-Out_Results', startrow=12)
    
    # Sheet 6: K-Fold Results
    kfold_summary_df = pd.DataFrame({
        'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Mean': [cv_acc_mean, cv_prec_mean, cv_rec_mean, cv_f1_mean],
        'Std': [cv_acc_std, cv_prec_std, cv_rec_std, cv_f1_std],
        'Mean_Percentage': [f'{cv_acc_mean*100:.2f}%', f'{cv_prec_mean*100:.2f}%', 
                           f'{cv_rec_mean*100:.2f}%', f'{cv_f1_mean*100:.2f}%'],
        'Std_Percentage': [f'{cv_acc_std*100:.2f}%', f'{cv_prec_std*100:.2f}%', 
                          f'{cv_rec_std*100:.2f}%', f'{cv_f1_std*100:.2f}%']
    })
    
    kfold_detail_df = pd.DataFrame(fold_results)
    
    kfold_summary_df.to_excel(writer, sheet_name='K-Fold_Results', index=False, startrow=0)
    kfold_detail_df.to_excel(writer, sheet_name='K-Fold_Results', index=False, startrow=8)
    
    # Sheet 7: Comparison Hold-Out vs K-Fold
    comparison_df = pd.DataFrame({
        'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Hold-Out': [holdout_accuracy, holdout_precision, holdout_recall, holdout_f1],
        'K-Fold_Mean': [cv_acc_mean, cv_prec_mean, cv_rec_mean, cv_f1_mean],
        'K-Fold_Std': [cv_acc_std, cv_prec_std, cv_rec_std, cv_f1_std],
        'Hold-Out_Percentage': [f'{holdout_accuracy*100:.2f}%', f'{holdout_precision*100:.2f}%', 
                               f'{holdout_recall*100:.2f}%', f'{holdout_f1*100:.2f}%'],
        'K-Fold_Percentage': [f'{cv_acc_mean*100:.2f}% ± {cv_acc_std*100:.2f}%', 
                             f'{cv_prec_mean*100:.2f}% ± {cv_prec_std*100:.2f}%',
                             f'{cv_rec_mean*100:.2f}% ± {cv_rec_std*100:.2f}%', 
                             f'{cv_f1_mean*100:.2f}% ± {cv_f1_std*100:.2f}%']
    })
    comparison_df.to_excel(writer, sheet_name='Comparison_Results', index=False)
    
    # Sheet 8: Data Bersih untuk Training
    df_clean_export = df_clean[['content', 'clean', 'score', 'kategori']].copy()
    df_clean_export.to_excel(writer, sheet_name='Data_Bersih', index=False)
    
    # Sheet 9: Fitur Penting
    feature_df.head(50).to_excel(writer, sheet_name='Fitur_Penting', index=False)
    
    # Sheet 10: Statistik Dataset
    stats_data = {
        'Informasi': [
            'Total Ulasan Awal',
            'Total Ulasan Setelah Preprocessing',
            'Ulasan Positif',
            'Ulasan Negatif',
            'Persentase Positif',
            'Persentase Negatif',
            'Rata-rata Score',
            'Score Tertinggi',
            'Score Terendah',
            'Total Fitur TF-IDF',
            'Data Training (Hold-Out)',
            'Data Testing (Hold-Out)',
            'Vocabulary Size (Manual TF-IDF Sample)',
            'Metode Terbaik',
            'Akurasi Terbaik',
            'Kecenderungan Sentimen'
        ],
        'Nilai': [
            len(df),
            len(df_clean),
            len(df_clean[df_clean['kategori'] == 'Positif']),
            len(df_clean[df_clean['kategori'] == 'Negatif']),
            f"{len(df_clean[df_clean['kategori'] == 'Positif'])/len(df_clean)*100:.1f}%",
            f"{len(df_clean[df_clean['kategori'] == 'Negatif'])/len(df_clean)*100:.1f}%",
            f"{df_clean['score'].mean():.2f}",
            df_clean['score'].max(),
            df_clean['score'].min(),
            X.shape[1],
            X_train.shape[0],
            X_test.shape[0],
            len(vocab_sample),
            best_method,
            f"{max(holdout_accuracy, cv_acc_mean)*100:.0f}%",
            sentiment_tendency
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_excel(writer, sheet_name='Statistik_Dataset', index=False)

print(f"   ✓ File Excel berhasil disimpan: {excel_filename}")

# KESIMPULAN
print("\n" + "="*80)
print("KESIMPULAN")
print("="*80)
print(f"1. Proses web scraping berhasil mengumpulkan {len(df)} ulasan aplikasi Shopee")
print(f"   dari Google Play Store yang kemudian diproses menjadi {len(df_clean)} data bersih.")

print(f"\n2. Performa algoritma Naive Bayes:")
print(f"   - Hold-Out (80:20): Akurasi {holdout_accuracy*100:.0f}%")
print(f"   - 10-Fold CV: Akurasi {cv_acc_mean*100:.0f}% ± {cv_acc_std*100:.1f}%")

if holdout_accuracy > cv_acc_mean:
    print(f"\n3. Metode Hold-Out menghasilkan akurasi lebih baik ({holdout_accuracy*100:.0f}%)")
    print(f"   dibandingkan 10-Fold Cross Validation ({cv_acc_mean*100:.0f}%).")
    print(f"   Selisih: {(holdout_accuracy - cv_acc_mean)*100:.0f}%")
else:
    print(f"\n3. Metode 10-Fold Cross Validation menghasilkan akurasi lebih baik ({cv_acc_mean*100:.0f}%)")
    print(f"   dibandingkan Hold-Out ({holdout_accuracy*100:.0f}%).")
    print(f"   Selisih: {(cv_acc_mean - holdout_accuracy)*100:.0f}%")

print(f"\n4. Berdasarkan hasil pengujian, sentimen pengguna Shopee yang memberikan")
print(f"   ulasan pada Google Play Store cenderung {sentiment_tendency}.")

print(f"\n5. Grafik perbandingan metrik evaluasi telah dibuat untuk visualisasi hasil.")

print(f"\nWaktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n✅ PROSES SELESAI!")
print("="*80)
