from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np 
import os 

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app) # Mengizinkan Cross-Origin Resource Sharing

# --- TAHAP 1: MUAT SEMUA MODEL ---
# Kita sekarang memuat SEMUA 4 file .joblib
try:
    # 1. Model utama untuk prediksi
    model = joblib.load("best_model.joblib")
    
    # 2. Scaler untuk fitur numerik (umur, tinggi, berat)
    scaler = joblib.load("scaler.joblib")
    
    # 3. Encoder untuk fitur 'Jenis Kelamin' (input)
    # PERHATIKAN: Nama file pakai spasi, harus sama persis
    jk_encoder = joblib.load("Jenis Kelamin_encoder.joblib")
    
    # 4. Encoder untuk target 'Stunting' (output)
    stunting_encoder = joblib.load("Stunting_encoder.joblib")
    
    print("Semua 4 model (model, scaler, 2 encoder) berhasil dimuat!")

except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None
    scaler = None
    jk_encoder = None
    stunting_encoder = None

# --- TAHAP 2: ENDPOINT PREDIKSI DENGAN PREPROCESSING ---
# Ini adalah inti dari API Anda.
@app.route('/predict', methods=['POST'])
def predict():    
    # Cek apakah semua model sudah siap
    if not all([model, scaler, jk_encoder, stunting_encoder]):
        return jsonify({"error": "Satu atau lebih file model gagal dimuat di server"}), 500

    try:
        # 1. Ambil data JSON yang dikirim oleh React
        data = request.get_json()

        # React HARUS mengirim JSON dengan key yang SAMA PERSIS seperti ini:
        # {
        #   "jenis_kelamin": "Laki-laki",  // (atau "Perempuan")
        #   "umur": 24,                  // (angka)
        #   "tinggi": 85.5,              // (angka)
        #   "berat": 10.1                // (angka)
        # }

        # --- MULAI PREPROCESSING ---
        # Kita harus mengulang langkah yang sama persis seperti saat training

        # 2. Ambil data mentah dari JSON
        jk_string = data['jenis_kelamin']
        umur = data['umur']
        tinggi = data['tinggi']
        berat = data['berat']

        # 3. ENCODE 'Jenis Kelamin' (String -> Angka)
        # Input: ["Laki-laki"] -> Output: [0]
        # Kita ambil elemen pertama [0] untuk mendapatkan angkanya saja
        jk_encoded = jk_encoder.transform([jk_string])[0]

        # 4. SCALE Fitur Numerik (Angka -> Angka Skala 0-1)
        # Scaler dilatih dengan urutan: ['Umur (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)']
        # Kita harus memasukkan data dengan urutan yang SAMA PERSIS.
        # Input harus 2D array, contoh: [[24, 85.5, 10.1]]
        numerical_features = [[umur, tinggi, berat]]
        scaled_features = scaler.transform(numerical_features)
        
        # Outputnya adalah 2D array, contoh: [[0.25, 0.4, 0.3]]
        # Kita ambil baris pertama [0]
        umur_scaled = scaled_features[0][0]
        tinggi_scaled = scaled_features[0][1]
        berat_scaled = scaled_features[0][2]

        # 5. GABUNGKAN SEMUA FITUR
        # Model utama dilatih dengan urutan: 
        # ['Jenis Kelamin', 'Umur (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)']
        # Kita gabungkan fitur yg sudah di-encode dan di-scale sesuai urutan itu
        
        final_features_list = [jk_encoded, umur_scaled, tinggi_scaled, berat_scaled]
        
        # 6. Ubah jadi 2D Numpy Array
        # Sesuai format yang diterima model .predict()
        final_features = [np.array(final_features_list)]
        
        # --- SELESAI PREPROCESSING ---

        # 7. Lakukan Prediksi
        # Input: [[0, 0.25, 0.4, 0.3]]
        prediction_encoded = model.predict(final_features)
        # Output: [1] (Contoh)

        # 8. UBAH PREDIKSI (Angka -> String)
        # Kita pakai stunting_encoder untuk mengubah angka hasil prediksi
        # kembali menjadi label string yang bisa dibaca manusia.
        # Input: [1] -> Output: ["Stunted"]
        prediction_string = stunting_encoder.inverse_transform(prediction_encoded)

        # 9. Ambil hasil string-nya
        output = prediction_string[0] # "Stunted"

        # 10. Kirim hasil akhir ke React
        return jsonify({'prediction': output})

    except KeyError as e:
        # Ini terjadi jika React salah kirim key JSON
        # (misal: "nama" bukan "jenis_kelamin")
        return jsonify({"error": f"Key JSON tidak ditemukan: {str(e)}. Pastikan key adalah 'jenis_kelamin', 'umur', 'tinggi', dan 'berat'."}), 400
    
    except Exception as e:
        # Tangkap error lainnya (misal: user kirim "Pria" bukan "Laki-laki")
        return jsonify({"error": f"Terjadi error saat prediksi: {str(e)}"}), 400

# --- INI BAGIAN PENTING UNTUK HOSTING vs LOCALHOST ---
# (Tidak perlu diubah, sudah benar)

if __name__ == '__main__':
    # Ambil port dari environment variable, atau pakai 5000 jika tidak ada
    port = int(os.environ.get('PORT', 5000))
    
    # Jalankan HANYA untuk development di localhost
    app.run(debug=True, port=port)