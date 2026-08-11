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

import numpy as np

from . import config, mediapipe_io
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


class DetectorSinais(mediapipe_io.Landmarker):
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
        self._maos = mediapipe_io.landmarker_de_maos(
            caminho_maos or config.MODELO_MAOS,
            num_maos=2,
            confianca=confianca_minima,
            em_video=True,
        )
        self._pose = mediapipe_io.landmarker_de_pose(
            caminho_pose or config.MODELO_POSE,
            confianca=confianca_minima,
        )
        self._recursos = (self._maos, self._pose)

    def detectar(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
        video_espelhado: bool = False,
    ) -> DeteccaoSinal:
        imagem = mediapipe_io.imagem(frame_rgb)

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
            pontos = mediapipe_io.pontos(marcos)
            cruas.append(pontos)

            if not resultado.handedness:
                continue

            if mediapipe_io.e_mao_esquerda(resultado.handedness, i, video_espelhado):
                esquerda = pontos if esquerda is None else esquerda
            else:
                direita = pontos if direita is None else direita

        return esquerda, direita, cruas

    @staticmethod
    def _extrair_corpo(resultado) -> np.ndarray | None:
        if not resultado.pose_landmarks:
            return None
        return mediapipe_io.pontos(resultado.pose_landmarks[0])
