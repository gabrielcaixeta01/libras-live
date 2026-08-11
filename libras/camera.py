"""Captura da webcam. Fino de propósito — é a parte que não dá para testar."""

from __future__ import annotations

import cv2
import numpy as np

from . import config


class CameraIndisponivel(RuntimeError):
    pass


class Camera:
    """Webcam como context manager.

    Entrega o frame já em BGR (para desenhar/mostrar) e em RGB (para o
    MediaPipe), evitando a conversão duplicada espalhada pelo loop principal.
    """

    def __init__(
        self,
        indice: int = config.INDICE_CAMERA,
        largura: int = config.LARGURA_CAMERA,
        altura: int = config.ALTURA_CAMERA,
        espelhar: bool = config.ESPELHAR_VIDEO,
    ):
        self.indice = indice
        self.espelhar = espelhar

        self._captura = cv2.VideoCapture(indice)
        if not self._captura.isOpened():
            raise CameraIndisponivel(
                f"Nao consegui abrir a camera no indice {indice}.\n"
                "No macOS, verifique se o terminal tem permissao de camera em\n"
                "Ajustes do Sistema > Privacidade e Seguranca > Camera."
            )

        self._captura.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        self._captura.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

    def ler(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Retorna (frame_bgr, frame_rgb), ou None se a captura falhou."""
        ok, frame = self._captura.read()
        if not ok:
            return None

        if self.espelhar:
            frame = cv2.flip(frame, 1)

        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def fechar(self) -> None:
        self._captura.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()
