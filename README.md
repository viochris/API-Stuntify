# 👶 Stuntify API: Intelligent Inference Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-RESTful-orange)
![Status](https://img.shields.io/badge/Status-Operational-success)

## 📌 Overview
**Stuntify API** is a specialized backend service designed to bridge the gap between raw health data and actionable stunting diagnostics.

It isn't just a script; it's the **intelligent engine** working behind the scenes. While the frontend handles the looks, this API handles the **thinking**. Unlike simple model wrappers, this project implements a complete **MLOps Inference Pipeline**. It intelligently orchestrates distinct serialized artifacts (Encoders, Scalers, Models) to ensure that every prediction request undergoes the exact same rigorous preprocessing as the training phase.

## ✨ Key Features

### 🧠 Multi-Artifact Orchestration (The "Invisible" Brain)
The system acts as a centralized inference unit. It doesn't just "guess"; it reconstructs the mathematical environment by synchronizing 4 key artifacts:
* **Gender Encoder:** Translates human categories (`Laki-laki`) into machine vectors.
* **Standard Scaler:** Normalizes anthropometric data (`Age`, `Height`, `Weight`) to the model's expected distribution.
* **Classifier Model:** The core logic engine trained for high accuracy.
* **Target Decoder:** Translates the math result back to a human-readable label (e.g., `Severely Stunted`).

### 🛡️ Defensive Architecture
* **Schema Validation:** Strictly enforces JSON structure to prevent garbage-in-garbage-out.
* **Smart Error Handling:** Catches edge cases (missing keys, wrong types) and returns meaningful HTTP 400 errors instead of crashing the server.
* **Cross-Origin Ready:** Fully configured with **Flask-CORS** to serve requests from any device—be it a Web App, Mobile Phone, or IoT Device.

## 🛠️ Tech Stack
* **Core:** Python 3.9+
* **Service:** Flask (Microframework)
* **Computation:** NumPy, Scikit-Learn, Joblib
* **Security:** Flask-CORS (Cross-Origin Resource Sharing enabled)

## 🚀 The Inference Pipeline (How It Works)
Most tutorials just load a model. This project ensures data consistency through a strict lifecycle for every request:

1.  **Ingestion:** The API receives raw JSON data: `{"jenis_kelamin", "umur", "tinggi", "berat"}`.
2.  **Context Reconstruction:** It loads the synchronized artifacts (Encoder, Scaler, Classifier, Decoder).
3.  **Processing:** The raw data flows through the pipeline:
    > *Input Validated -> Gender Encoded -> Metrics Scaled -> Predicted -> Label Decoded*
4.  **Response:** The final output is packaged as clean JSON for the client.

## 🔌 Integration Guide (API Contract)

To talk to Stuntify, send a **POST** request to `/predict`.

**1. The Request (What you send):**
```json
{
    "jenis_kelamin": "Laki-laki",
    "umur": 24,
    "tinggi": 85.5,
    "berat": 10.1
}
````

**2. The Response (What you get):**

```json
{
    "prediction": "Severely Stunted"
}
```

## 📦 Installation & Usage

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/viochris/API-Stuntify.git
    cd API-Stuntify
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**
    Execute the main script directly:

    ```bash
    python api_predict.py
    ```

    *Output: `Running on http://127.0.0.1:5000`*

4.  **Consume Endpoint**
    Send the POST request described in the **Integration Guide** above to `http://127.0.0.1:5000/predict`.

-----

**Author:** [Silvio Christian, Joe](https://www.linkedin.com/in/silvio-christian-joe)
*"Code that speaks JSON, Logic that saves lives."*
