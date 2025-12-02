# 🚀 Optimisation Pipeline: GPU Preprocessing & NVJPEG

Ce document détaille les optimisations apportées au pipeline de pré-traitement d'images de `depth-anything-3`, visant à réduire la latence d'inférence, en particulier pour les hautes résolutions (4K) et les flux vidéo.

## 📊 Résultats du Benchmark (NVIDIA L4)

Les tests ont comparé quatre stratégies différentes sur un lot de 4 images avec 8 workers.

| Résolution | Méthode CPU (Ref) | Méthode Full GPU (Kornia) | Méthode Hybride | **Méthode GPU Decode (NVJPEG)** | **Gain vs CPU** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGA** (640x480) | 35.0 ms | 20.6 ms | 17.4 ms | **10.8 ms** | **x 3.2** |
| **HD** (1280x720) | 33.8 ms | 44.8 ms | 28.2 ms | **17.6 ms** | **x 1.9** |
| **FHD** (1920x1080) | 53.2 ms | 97.2 ms | 46.5 ms | **18.3 ms** | **x 2.9** |
| **4K** (3840x2160) | 222.5 ms | 436.3 ms | 209.5 ms | **54.0 ms** | **x 4.1** 🚀 |

### Analyse
1.  **GPU Decode (NVJPEG) est dominant :** En lisant directement le fichier JPEG et en le décodant via l'accélération matérielle du GPU, on élimine le goulot d'étranglement du décodage CPU et le transfert coûteux de bitmaps non compressés sur le bus PCI-e.
2.  **Limites du "Full GPU" classique :** L'approche naïve (charger sur CPU -> transférer -> resize GPU) devient **2x plus lente** que le CPU pour la 4K à cause de la latence de transfert mémoire.
3.  **Efficacité de la Méthode Hybride :** Pour les images déjà en mémoire (ex: flux vidéo décodé ailleurs), l'approche Hybride (Resize CPU -> Transfert uint8 -> Norm GPU) offre un gain constant sans overhead.

---

## 🛠️ Stratégies Implémentées

Le système sélectionne automatiquement la stratégie optimale en fonction du matériel et du type d'entrée.

### 1. 🟢 GPU Decode (Fichiers + CUDA/MPS)
*   **Cible :** Inférence CLI, Traitement par lots depuis le disque.
*   **Technique CUDA :** Utilise `nvjpeg` (via `torchvision`) pour décoder le JPEG directement dans la mémoire GPU.
*   **Technique MPS :** Utilise les API natives optimisées (ImageIO/Accelerate) pour décoder et transférer immédiatement.
*   **Flux :** `File Bytes` → `Decoder (HW/Opt)` → `GPU Memory` → `Kornia Resize/Norm`.

### 2. 🟡 Mode Hybride (Objets Mémoire + GPU)
*   **Cible :** API Python, Webcams, Flux vidéo où l'image est déjà un array numpy/PIL.
*   **Technique :** Effectue le redimensionnement sur CPU (rapide et parallèle), mais retarde la normalisation et la conversion float.
*   **Avantage :** Transfère des données `uint8` (4x plus légères que `float32`) vers le GPU, réduisant la saturation de la bande passante.

### 3. 🔴 Mode CPU Standard (Fallback)
*   **Cible :** Machines sans GPU dédié.
*   **Technique :** Pipeline classique utilisant `PIL` et `numpy` avec parallélisation multiprocessing.

---

## 💻 Architecture du Code

### `src/depth_anything_3/api.py`
*   **Détection Auto :** Configure automatiquement `GPUInputProcessor` si un GPU (NVIDIA ou Apple Silicon) est disponible.
*   **Pipeline Intelligent :** Ajuste dynamiquement les étapes de normalisation selon que les données arrivent du CPU ou sont déjà sur le GPU.

### `src/depth_anything_3/utils/io/gpu_input_processor.py`
*   **Support NVJPEG :** Intégration de `torchvision.io.decode_jpeg`.
*   **Support MPS :** Compatibilité assurée pour les puces M1/M2/M3.
*   **Kornia :** Utilisation de Kornia pour les opérations géométriques (Resize, CenterCrop) directement sur les tenseurs GPU.

### `benchmarks/gpu_preprocessing_benchmark.py`
*   Nouveau script de benchmark inclus pour valider les performances sur votre matériel spécifique.
*   Test : `uv run python benchmarks/gpu_preprocessing_benchmark.py`
