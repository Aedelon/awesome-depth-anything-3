# src/depth_anything_3/utils/async_exporter.py

import concurrent.futures
from typing import Callable, Any
from depth_anything_3.utils.logger import logger


class AsyncExporter:
    """
    Gère les tâches d'exportation de manière asynchrone dans un thread séparé.
    Permet d'écrire sur le disque pendant que le GPU calcule le batch suivant.
    """

    def __init__(self, max_workers: int = 1):
        # Un seul worker suffit généralement pour l'écriture disque (I/O bound)
        # et évite les conflits d'écriture sur les mêmes fichiers.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def submit(self, fn: Callable, *args, **kwargs):
        """Soumet une tâche d'exportation."""
        future = self.executor.submit(fn, *args, **kwargs)
        self.futures.append(future)

    def shutdown(self, wait: bool = True):
        """Attend la fin des tâches et ferme l'exécuteur."""
        if wait:
            concurrent.futures.wait(self.futures)

        # Vérification des erreurs après coup
        for future in self.futures:
            if future.done() and future.exception():
                logger.error(f"Async export failed with error: {future.exception()}")
                raise future.exception()

        self.executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)