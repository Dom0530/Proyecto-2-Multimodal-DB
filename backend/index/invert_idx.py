import os
import glob
import csv
import numpy as np
import librosa
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import heapq
import math
import time
from collections import defaultdict
import joblib


class BoAWRetriever:
    def __init__(self, n_clusters=100, n_mfcc=12, Na=1):
        self.n_clusters = n_clusters
        self.n_mfcc = n_mfcc
        self.Na = Na
        self.codebook = None
        self.idf = None
        self.tfidf_matrix = []
        self.inverted_index = defaultdict(list)
        self.audio_paths = []
        self.labels = []

    def extract_llds(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # Calcular parámetros seguros
        hop_length = int(0.01 * sr)
        win_length = int(0.025 * sr)
        n_fft = max(512, 2**int(np.ceil(np.log2(win_length))))  # asegura que n_fft >= win_length y es potencia de 2
        
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        energy = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)
        
        mfcc = mfcc[:, :energy.shape[1]]  # asegurar mismo número de frames
        return np.vstack((mfcc, energy)).T


    def build_codebook(self, audio_paths):
        all_descriptors = []
        valid_audio_paths = []
        for path in audio_paths:
            try:
                llds = self.extract_llds(path)
                all_descriptors.extend(llds)
                valid_audio_paths.append(path)
            except Exception as e:
                print(f"[WARNING] Error procesando {path}: {e}")
        
        if not all_descriptors:
            raise RuntimeError("No se pudo procesar ningún audio válido.")

        self.audio_paths = valid_audio_paths  # actualizar solo con los que sí cargaron
        print(f"[INFO] Entrenando KMeans con {len(all_descriptors)} vectores de {len(valid_audio_paths)} archivos válidos...")
        self.codebook = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.codebook.fit(np.array(all_descriptors))


    def compute_boaw(self, llds):
        distances = cdist(llds, self.codebook.cluster_centers_)
        histogram = np.zeros(self.n_clusters)
        top_indices = np.argsort(distances, axis=1)[:, :self.Na]
        for row in top_indices:
            for idx in row:
                histogram[idx] += 1
        return np.log1p(histogram)

    def compute_df(self, histograms):
        histograms = np.array(histograms)
        df = np.sum(histograms > 0, axis=0)
        return df

    def compute_tf_idf(self, histograms, df):
        N = len(histograms)
        tf = histograms / (np.sum(histograms, axis=1, keepdims=True) + 1e-6)
        idf = np.log(1 + N / (df + 1e-6))
        self.idf = idf
        return tf * idf

    def cosine_similarity(self, a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def fit(self, audio_paths, labels=None):
        self.labels = []
        histograms = []
        valid_labels = []

        self.build_codebook(audio_paths)
        
        for path, label in zip(self.audio_paths, labels or []):
            try:
                llds = self.extract_llds(path)
                hist = self.compute_boaw(llds)
                histograms.append(hist)
                self.labels.append(label)
            except Exception as e:
                print(f"[WARNING] Error en histogramas para {path}: {e}")
                continue

        if not histograms:
            raise RuntimeError("No se pudieron calcular histogramas para ningún audio.")

        df = self.compute_df(histograms)
        self.tfidf_matrix = self.compute_tf_idf(histograms, df)

        for doc_id, hist in enumerate(histograms):
            for word_id, tf in enumerate(hist):
                if tf > 0:
                    self.inverted_index[word_id].append((doc_id, tf))



    def query(self, query_path, k=3):
        llds = self.extract_llds(query_path)
        bow = self.compute_boaw(llds)
        t0 = time.time()
        tf = bow / (np.sum(bow) + 1e-6)
        tfidf_query = tf * self.idf

        heap = []
        for i, tfidf_vec in enumerate(self.tfidf_matrix):
            sim = self.cosine_similarity(tfidf_query, tfidf_vec)
            heapq.heappush(heap, (-sim, i))  # max-heap por similitud

        top_k = heapq.nsmallest(k, heap)

        return [(self.audio_paths[i], -sim) for sim, i in top_k], time.time() - t0
    
    def query_inverted_knn(self, query_path, k=5):
        
        if self.codebook is None or self.inverted_index is None:
            raise ValueError("El modelo debe tener codebook e índice cargado.")

        N = len(self.audio_paths)

        llds = self.extract_llds(query_path)
        bow_query = self.compute_boaw(llds)

        tfidf_query = {}
        norm_q = 0
        for word_id, tf_q in enumerate(bow_query):
            if tf_q == 0 or word_id not in self.inverted_index:
                continue
            df = len(self.inverted_index[word_id])
            idf = math.log(1 + N / df)
            tfidf = math.log(1 + tf_q) * idf
            tfidf_query[word_id] = tfidf
            norm_q += tfidf**2
        norm_q = math.sqrt(norm_q)

        scores = defaultdict(float)
        norms_d = defaultdict(float)

        for word_id, tfidf_q in tfidf_query.items():
            for doc_id, tf_d in self.inverted_index[word_id]:
                df = len(self.inverted_index[word_id])
                idf = math.log(1 + N / df)
                tfidf_d = math.log(1 + tf_d) * idf
                scores[doc_id] += tfidf_q * tfidf_d
                norms_d[doc_id] += tfidf_d**2

        results = []
        for doc_id, dot in scores.items():
            norm_d = math.sqrt(norms_d[doc_id])
            sim = dot / (norm_d * norm_q) if norm_d > 0 and norm_q > 0 else 0
            results.append((self.audio_paths[doc_id], sim))

        return sorted(results, key=lambda x: -x[1])[:k]


    def export_features_csv(self, output_path="features.csv"):
        with open(output_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = [f"cw_{i}" for i in range(self.n_clusters)] + ["label", "filepath"]
            writer.writerow(header)
            for vec, label, path in zip(self.tfidf_matrix, self.labels, self.audio_paths):
                row = list(vec) + [label, path]
                writer.writerow(row)



    @staticmethod
    def load_dataset(root='dataset'):
        audio_paths = []
        labels = []
        for folder in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                for audio_file in glob.glob(os.path.join(folder_path, "*.wav")):
                    audio_paths.append(audio_file)
                    labels.append(folder)
        return audio_paths, labels
    
    def load_model_from_disk(self, codebook_path="codebook.pkl", idf_path="idf.npy", features_csv="features.csv", inverted_index_path="inverted_index.pkl"):

        self.codebook = joblib.load(codebook_path)
        self.idf = np.load(idf_path)
        self.inverted_index = joblib.load(inverted_index_path)
        tfidf_matrix, labels, paths = load_features_from_csv(features_csv)

        self.tfidf_matrix = tfidf_matrix
        self.labels = labels
        self.audio_paths = paths
        

    def fit_and_export(self, dataset_root="dataset", output_csv="features.csv"):
        audio_paths, labels = self.load_dataset(dataset_root)
        self.fit(audio_paths, labels)
        self.export_features_csv(output_csv)
        joblib.dump(self.codebook, "codebook.pkl")
        np.save("idf.npy", self.idf)
        joblib.dump(self.inverted_index, 'inverted_index.pkl')


def load_features_from_csv(csv_path):
    tfidf_matrix = []
    labels = []
    paths = []

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            *features, label, path = row
            tfidf_matrix.append([float(val) for val in features])
            labels.append(label)
            paths.append(path)

    return np.array(tfidf_matrix), labels, paths


retriever = BoAWRetriever(n_clusters=100, Na=1)

dataset_root = "C:\\Users\\niki_\\Downloads\\fma_small_wav" 

retriever.fit_and_export(dataset_root=dataset_root, output_csv="features.csv")
#codebook_path = 'C:\\Users\\niki_\\OneDrive\\Dom\\utec\\11 ciclo\\BD2\\Proyecto 2\\codebook.pkl'
#idf_path = 'C:\\Users\\niki_\\OneDrive\\Dom\\utec\\11 ciclo\\BD2\\Proyecto 2\\idf.npy'
#features_path = 'C:\\Users\\niki_\\OneDrive\\Dom\\utec\\11 ciclo\\BD2\\Proyecto 2\\features.csv'
#inverted_index_path = 'C:\\Users\\niki_\\OneDrive\\Dom\\utec\\11 ciclo\\BD2\\Proyecto 2\\inverted_index.pkl'
#retriever.load_model_from_disk(codebook_path, idf_path, features_path, inverted_index_path)

query = "C:\\Users\\niki_\\Downloads\\fma_small_wav\\083\\083898.wav"

result1, time_1 = retriever.query(query_path=query, k=5)
result2, time_2 = retriever.query_inverted_index(query_path=query, k=5) 

#print(result1)
#print(result2)
#print(time_1)
#print(time_2)
