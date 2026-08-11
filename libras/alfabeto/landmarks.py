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

import numpy as np

from .. import config, mediapipe_io

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


def _primeira_mao(resultado) -> Deteccao | None:
    """O resultado do HandLandmarker → `Deteccao`, ou None se não achou mão.

    Compartilhado pelos dois detectores abaixo: eles diferem só em pedir vídeo
    ou imagem ao MediaPipe, e a leitura do que volta é a mesma.
    """
    if not resultado.hand_landmarks:
        return None

    pontos = mediapipe_io.pontos(resultado.hand_landmarks[0])
    mao_esquerda = mediapipe_io.e_mao_esquerda(resultado.handedness)

    return Deteccao(
        pontos=pontos,
        vetor=normalizar(pontos, mao_esquerda),
        mao_esquerda=mao_esquerda,
    )


class DetectorMaos(mediapipe_io.Landmarker):
    """Uma mão, em vídeo. É o detector do alfabeto.

        with DetectorMaos() as detector:
            deteccao = detector.detectar(frame_rgb, timestamp_ms)
    """

    def __init__(self, caminho_modelo=None, confianca_minima: float = 0.5):
        self._landmarker = mediapipe_io.landmarker_de_maos(
            caminho_modelo or config.MODELO_MAOS,
            num_maos=1,
            confianca=confianca_minima,
            em_video=True,
        )
        self._recursos = (self._landmarker,)

    def detectar(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Deteccao | None:
        """Detecta uma mão. Retorna None quando não há mão no frame."""
        resultado = self._landmarker.detect_for_video(
            mediapipe_io.imagem(frame_rgb), timestamp_ms
        )
        return _primeira_mao(resultado)


class DetectorImagem(mediapipe_io.Landmarker):
    """Uma mão, em foto solta (preparação da base pública).

    Modo IMAGE em vez de VIDEO: sem rastreamento entre frames, que não faz
    sentido para um diretório de fotos independentes. A confiança mínima é mais
    baixa de propósito — as fotos difíceis são justamente o que a cascata de
    recuperação existe para resgatar.
    """

    def __init__(self, caminho_modelo=None, confianca_minima: float = 0.3):
        self._landmarker = mediapipe_io.landmarker_de_maos(
            caminho_modelo or config.MODELO_MAOS,
            num_maos=1,
            confianca=confianca_minima,
            em_video=False,
        )
        self._recursos = (self._landmarker,)

    def detectar(self, imagem_rgb: np.ndarray) -> Deteccao | None:
        resultado = self._landmarker.detect(mediapipe_io.imagem(imagem_rgb))
        return _primeira_mao(resultado)
