import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Generator

def get_sorted_indices_by_aspect_ratio(
    images: List[Union[str, np.ndarray, Image.Image]]
) -> Tuple[np.ndarray, List[float]]:
    """
    Retourne les indices triés par ratio d'aspect (W / H).
    Gère les chemins de fichiers (lazy loading), arrays numpy et objets PIL.
    """
    ratios = []
    
    for img in images:
        if isinstance(img, str):
            # Lazy loading: on ne lit que le header pour avoir la taille
            try:
                with Image.open(img) as i:
                    w, h = i.size
                    ratios.append(w / h)
            except Exception:
                ratios.append(1.0) # Fallback
        elif isinstance(img, np.ndarray):
            # H, W, C ou H, W
            h, w = img.shape[:2]
            ratios.append(w / h)
        elif isinstance(img, Image.Image):
            w, h = img.size
            ratios.append(w / h)
        else:
            # Fallback
            ratios.append(1.0)

    # Argsort pour obtenir les indices triés
    sorted_indices = np.argsort(ratios)
    return sorted_indices, [ratios[i] for i in sorted_indices]

def chunk_indices(indices: np.ndarray, batch_size: int) -> Generator[np.ndarray, None, None]:
    """Générateur pour découper les indices en batches"""
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]