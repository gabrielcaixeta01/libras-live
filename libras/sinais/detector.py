"""Envolve os dois landmarkers do MediaPipe num frame de 49 pontos.

Deliberadamente fino, pela mesma razão que `camera.py` e `ui.py` são finos: é a
parte que não dá para testar sem hardware, então ela não decide nada. Traduz o
resultado de duas bibliotecas para um array e devolve.

**Por que dois modelos e não o Holistic.** O `mp.solutions.holistic` que fazia
mãos + pose + rosto de uma vez foi removido no MediaPipe 1.0 junto com o resto
das Solutions antigas. Na API Tasks, o equivalente é rodar HandLandmarker e
PoseLandmarker lado a lado — o que, de quebra, permite pedir só a pose leve e
ignorar os 468 pontos de rosto que não usamos.

**Sobre a lateralidade.** O HandLandmarker reporta "Left"/"Right" já corrigindo
o espelhamento da imagem, mas o app espelha o vídeo para dar efeito de espelho.
Quem chama informa se o frame está espelhado, e a lateralidade é invertida junto
— senão as duas mãos trocariam de slot e todo sinal assimétrico sairia ao
contrário.
"""

from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .. import config
from . import pose


@dataclass(frozen=True)
class DeteccaoSinal:
    """O que o MediaPipe achou num frame."""

    frame: np.ndarray                    # (49, 3) cru, NaN no que faltou
    maos_cruas: list[np.ndarray]         # (21, 3) cada, para desenhar
    corpo_cru: np.ndarray | None         # (33, 3), para desenhar

    @property
    def tem_mao(self) -> bool:
        return len(self.maos_cruas) > 0

    @property
    def tem_corpo(self) -> bool:
        return self.corpo_cru is not None


class DetectorSinais:
    """HandLandmarker (duas mãos) + PoseLandmarker, no modo vídeo.

        with DetectorSinais() as detector:
            deteccao = detector.detectar(frame_rgb, timestamp_ms)
    """

    def __init__(
        self,
        caminho_maos=None,
        caminho_pose=None,
        confianca_minima: float = 0.5,
    ):
        maos = caminho_maos or config.MODELO_MAOS
        corpo = caminho_pose or config.MODELO_POSE

        for caminho in (maos, corpo):
            if not caminho.exists():
                raise FileNotFoundError(
                    f"Modelo do MediaPipe não encontrado em {caminho}.\n"
                    "Rode: bash scripts/download_model.sh"
                )

        self._maos = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(maos)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=confianca_minima,
                min_hand_presence_confidence=confianca_minima,
                min_tracking_confidence=confianca_minima,
            )
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(corpo)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=confianca_minima,
                min_pose_presence_confidence=confianca_minima,
                min_tracking_confidence=confianca_minima,
            )
        )

    def detectar(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
        video_espelhado: bool = False,
    ) -> DeteccaoSinal:
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        resultado_maos = self._maos.detect_for_video(imagem, timestamp_ms)
        resultado_pose = self._pose.detect_for_video(imagem, timestamp_ms)

        esquerda, direita, cruas = self._separar_maos(resultado_maos, video_espelhado)
        corpo = self._extrair_corpo(resultado_pose)

        return DeteccaoSinal(
            frame=pose.montar_frame(
                mao_esquerda=esquerda, mao_direita=direita, corpo=corpo
            ),
            maos_cruas=cruas,
            corpo_cru=corpo,
        )

    @staticmethod
    def _separar_maos(resultado, video_espelhado: bool):
        """Põe cada mão no seu slot anatômico. Ordem de detecção não serve.

        O MediaPipe entrega as mãos na ordem em que as achou, que varia de frame
        para frame. Se elas entrassem nessa ordem, um sinal assimétrico viraria
        outro no meio da gravação.
        """
        esquerda = direita = None
        cruas: list[np.ndarray] = []

        for i, marcos in enumerate(resultado.hand_landmarks):
            pontos = np.array([[m.x, m.y, m.z] for m in marcos], dtype=np.float32)
            cruas.append(pontos)

            categorias = resultado.handedness[i] if resultado.handedness else []
            if not categorias:
                continue

            e_esquerda = categorias[0].category_name == "Left"
            if video_espelhado:
                e_esquerda = not e_esquerda

            if e_esquerda and esquerda is None:
                esquerda = pontos
            elif not e_esquerda and direita is None:
                direita = pontos

        return esquerda, direita, cruas

    @staticmethod
    def _extrair_corpo(resultado) -> np.ndarray | None:
        if not resultado.pose_landmarks:
            return None
        marcos = resultado.pose_landmarks[0]
        return np.array([[m.x, m.y, m.z] for m in marcos], dtype=np.float32)

    def fechar(self) -> None:
        self._maos.close()
        self._pose.close()

    def __enter__(self) -> "DetectorSinais":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()
