"""Extração e normalização dos 21 pontos da mão.

Duas partes bem separadas de propósito:

- `normalizar` é função pura (numpy entra, numpy sai). É onde mora a lógica que
  faz o modelo funcionar longe da mesa onde foi treinado, e é testável sem
  câmera.
- `DetectorMaos` envolve o MediaPipe e é fino: só traduz o resultado da
  biblioteca para arrays.

MediaPipe 1.0.0 removeu `mp.solutions.hands`; aqui usamos a Tasks API.
"""

from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from . import config

# Pares de landmarks ligados por um "osso". Usado só para desenhar — em
# mediapipe 1.0.0 não existe mais `mp.solutions.drawing_utils`.
CONEXOES_MAO: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),           # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),           # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),      # médio
    (9, 13), (13, 14), (14, 15), (15, 16),    # anelar
    (13, 17), (17, 18), (18, 19), (19, 20),   # mínimo
    (0, 17),                                   # base da palma
)

NUM_PONTOS = 21
TAMANHO_VETOR = NUM_PONTOS * 3


def normalizar(pontos: np.ndarray, mao_esquerda: bool = False) -> np.ndarray:
    """Converte 21 pontos brutos num vetor de 63 floats comparável entre frames.

    Sem isso o classificador aprende a sua distância da câmera em vez da forma
    da mão. Três passos, nesta ordem (ver D3 da spec):

    1. Espelha o eixo X se for mão esquerda, para o modelo servir às duas mãos.
    2. Move o pulso para a origem — mata a posição na tela.
    3. Divide pela maior distância ao pulso — mata a distância da câmera.

    Args:
        pontos: array (21, 3) com as coordenadas cruas do MediaPipe.
        mao_esquerda: True se o MediaPipe classificou a mão como esquerda.

    Returns:
        Vetor (63,) float32.
    """
    if pontos.shape != (NUM_PONTOS, 3):
        raise ValueError(f"esperado shape (21, 3), recebido {pontos.shape}")

    pontos = np.asarray(pontos, dtype=np.float32).copy()

    if mao_esquerda:
        pontos[:, 0] *= -1.0

    pontos -= pontos[0]

    escala = float(np.max(np.linalg.norm(pontos, axis=1)))
    if escala > 1e-8:
        pontos /= escala

    return pontos.reshape(-1)


@dataclass(frozen=True)
class Deteccao:
    """Uma mão encontrada num frame."""

    pontos: np.ndarray       # (21, 3) cru, para desenhar na tela
    vetor: np.ndarray        # (63,) normalizado, para o classificador
    mao_esquerda: bool


class DetectorMaos:
    """Envolve o HandLandmarker do MediaPipe.

    Use como context manager — o landmarker segura recursos nativos:

        with DetectorMaos() as detector:
            deteccao = detector.detectar(frame_rgb, timestamp_ms)
    """

    def __init__(self, caminho_modelo=None, confianca_minima: float = 0.5):
        caminho = caminho_modelo or config.MODELO_MAOS
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo do MediaPipe não encontrado em {caminho}.\n"
                "Rode: bash scripts/download_model.sh"
            )

        opcoes = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(caminho)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=confianca_minima,
            min_hand_presence_confidence=confianca_minima,
            min_tracking_confidence=confianca_minima,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opcoes)

    def detectar(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Deteccao | None:
        """Detecta uma mão. Retorna None quando não há mão no frame."""
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = self._landmarker.detect_for_video(imagem, timestamp_ms)

        if not resultado.hand_landmarks:
            return None

        marcos = resultado.hand_landmarks[0]
        pontos = np.array([[m.x, m.y, m.z] for m in marcos], dtype=np.float32)

        mao_esquerda = False
        if resultado.handedness and resultado.handedness[0]:
            mao_esquerda = resultado.handedness[0][0].category_name == "Left"

        return Deteccao(
            pontos=pontos,
            vetor=normalizar(pontos, mao_esquerda),
            mao_esquerda=mao_esquerda,
        )

    def fechar(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "DetectorMaos":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()


class DetectorImagem:
    """Mesma coisa, mas para imagens soltas (preparação do dataset).

    Modo IMAGE em vez de VIDEO: sem tracking entre frames, que não faz sentido
    para um diretório de fotos independentes.
    """

    def __init__(self, caminho_modelo=None, confianca_minima: float = 0.3):
        caminho = caminho_modelo or config.MODELO_MAOS
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo do MediaPipe não encontrado em {caminho}.\n"
                "Rode: bash scripts/download_model.sh"
            )

        opcoes = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(caminho)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=confianca_minima,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opcoes)

    def detectar(self, imagem_rgb: np.ndarray) -> Deteccao | None:
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
        resultado = self._landmarker.detect(imagem)

        if not resultado.hand_landmarks:
            return None

        marcos = resultado.hand_landmarks[0]
        pontos = np.array([[m.x, m.y, m.z] for m in marcos], dtype=np.float32)

        mao_esquerda = False
        if resultado.handedness and resultado.handedness[0]:
            mao_esquerda = resultado.handedness[0][0].category_name == "Left"

        return Deteccao(
            pontos=pontos,
            vetor=normalizar(pontos, mao_esquerda),
            mao_esquerda=mao_esquerda,
        )

    def fechar(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "DetectorImagem":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()
