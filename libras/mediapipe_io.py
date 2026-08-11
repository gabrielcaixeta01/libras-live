"""As primitivas do MediaPipe, num lugar só.

Três detectores diferentes vivem neste projeto — uma mão em vídeo (alfabeto),
uma mão em imagem solta (preparação da base pública) e duas mãos mais o corpo
(sinais). Eles diferem no que pedem ao MediaPipe, mas repetiam exatamente as
mesmas quatro coisas: conferir se o modelo existe, montar o array de pontos a
partir dos marcos, ler a lateralidade e fechar recursos nativos.

Repetição em código que envolve biblioteca externa envelhece mal: quando a API
do MediaPipe mudar de novo — e ela já mudou uma vez, na 1.0, levando junto o
`mp.solutions` inteiro —, a correção precisa acontecer num lugar, não em três.

Nada aqui decide nada. É tradução: o que a biblioteca devolve vira numpy.
"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


def opcoes_base(caminho: Path) -> BaseOptions:
    """Confere que o modelo existe antes de entregá-lo ao MediaPipe.

    Sem isso o erro só aparece lá dentro, como uma falha de leitura de arquivo
    sem nome nem sugestão. Um clone novo do repositório sempre passa por aqui.
    """
    if not Path(caminho).exists():
        raise FileNotFoundError(
            f"Modelo do MediaPipe não encontrado em {caminho}.\n"
            "Rode: bash scripts/download_model.sh"
        )
    return BaseOptions(model_asset_path=str(caminho))


def imagem(frame_rgb: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def pontos(marcos) -> np.ndarray:
    """Marcos do MediaPipe → array (N, 3) float32."""
    return np.array([[m.x, m.y, m.z] for m in marcos], dtype=np.float32)


def e_mao_esquerda(handedness, indice: int = 0, video_espelhado: bool = False) -> bool:
    """Lê a lateralidade que o MediaPipe reportou para a mão `indice`.

    O detector já corrige o espelhamento da câmera, mas o app espelha o vídeo
    para dar efeito de espelho — e aí a correção dele fica invertida. Quem chama
    informa se o frame está espelhado; sem isso, as duas mãos trocariam de lado
    e todo sinal assimétrico sairia ao contrário.

    Devolve False quando não há lateralidade: é o mesmo que o código anterior
    fazia, e um chute para "direita" erra menos que travar o app.
    """
    if not handedness or indice >= len(handedness) or not handedness[indice]:
        return False

    esquerda = handedness[indice][0].category_name == "Left"
    return not esquerda if video_espelhado else esquerda


class Landmarker:
    """Base dos detectores: segura recursos nativos e some com eles no fim.

        with MeuDetector() as detector:
            ...

    As subclasses preenchem `_recursos` com o que precisa ser fechado.
    """

    _recursos: tuple = ()

    def fechar(self) -> None:
        for recurso in self._recursos:
            recurso.close()

    def __enter__(self):
        return self

    def __exit__(self, *_) -> None:
        self.fechar()


def landmarker_de_maos(
    caminho: Path,
    num_maos: int,
    confianca: float,
    em_video: bool,
):
    """HandLandmarker configurado. `em_video` liga o rastreamento entre frames.

    Modo IMAGE existe para diretórios de fotos independentes, onde rastrear o
    frame anterior não só não ajuda como atrapalha.
    """
    if em_video:
        opcoes = vision.HandLandmarkerOptions(
            base_options=opcoes_base(caminho),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_maos,
            min_hand_detection_confidence=confianca,
            min_hand_presence_confidence=confianca,
            min_tracking_confidence=confianca,
        )
    else:
        opcoes = vision.HandLandmarkerOptions(
            base_options=opcoes_base(caminho),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_maos,
            min_hand_detection_confidence=confianca,
        )

    return vision.HandLandmarker.create_from_options(opcoes)


def landmarker_de_pose(caminho: Path, confianca: float):
    """PoseLandmarker de uma pessoa, em vídeo.

    Só os sinais usam. O alfabeto não carrega este modelo — pose custa tempo de
    frame e ele não tem o que fazer com ela.
    """
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=opcoes_base(caminho),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=confianca,
            min_pose_presence_confidence=confianca,
            min_tracking_confidence=confianca,
        )
    )
