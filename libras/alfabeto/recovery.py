"""Segunda chance para imagens onde o MediaPipe não achou mão.

Na base pública o detector falha em 76% das imagens de **N** e 47% das de **M**:
são poses de mão fechada, onde os dedos se escondem atrás da palma. Essas
imagens são descartadas em silêncio, e N termina o `prepare_dataset` com 37
amostras contra 568 de B. O modelo aprende B muito bem e N quase nada.

A imagem não é ruim — o detector é que é sensível a enquadramento, contraste e
inclinação. Então em vez de desistir na primeira falha, tentamos a mesma imagem
de outros jeitos: espelhada, ampliada, com contraste equalizado e girada. Se uma
variante encontra a mão, desfazemos a transformação nos pontos e ficamos com os
landmarks na geometria da imagem original.

Isso é recuperação, não aumento de dados: o resultado é a mão que estava lá,
medida por outro caminho. O aumento é assunto do `augment.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import cv2
import numpy as np

from .landmarks import NUM_PONTOS, Deteccao, normalizar

Forma = tuple[int, int]  # (altura, largura)

IDENTIDADE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class DetectorDeImagem(Protocol):
    """O contrato que `landmarks.DetectorImagem` cumpre — aqui só para os testes."""

    def detectar(self, imagem_rgb: np.ndarray) -> Deteccao | None: ...


# --- geometria ---


def aplicar_afim(
    pontos: np.ndarray, matriz: np.ndarray, origem: Forma, destino: Forma
) -> np.ndarray:
    """Leva pontos normalizados de uma imagem para outra por uma matriz 2x3.

    As coordenadas do MediaPipe são frações da largura e da altura, não pixels;
    a matriz é em pixels. Então converte para pixel, aplica, converte de volta —
    usando formas diferentes na ida e na volta, que é o que faz a conta fechar
    quando a tela muda de tamanho.

    O eixo z passa intacto: é uma profundidade relativa, e giro no plano da
    imagem não mexe nela.
    """
    altura_origem, largura_origem = origem
    altura_destino, largura_destino = destino

    pixels = np.column_stack(
        [pontos[:, 0] * largura_origem, pontos[:, 1] * altura_origem]
    )
    homogeneo = np.column_stack([pixels, np.ones(len(pixels))])
    movidos = homogeneo @ np.asarray(matriz, dtype=np.float64).T

    return np.column_stack(
        [
            movidos[:, 0] / largura_destino,
            movidos[:, 1] / altura_destino,
            pontos[:, 2],
        ]
    ).astype(np.float32)


def girar(imagem: np.ndarray, graus: float) -> tuple[np.ndarray, np.ndarray]:
    """Gira a imagem expandindo a tela para não cortar os cantos.

    Returns:
        (imagem girada, matriz 2x3 que leva pixel original -> pixel girado).
    """
    altura, largura = imagem.shape[:2]
    centro = (largura / 2.0, altura / 2.0)

    matriz = cv2.getRotationMatrix2D(centro, graus, 1.0)
    cosseno, seno = abs(matriz[0, 0]), abs(matriz[0, 1])

    nova_largura = int(altura * seno + largura * cosseno)
    nova_altura = int(altura * cosseno + largura * seno)

    matriz[0, 2] += nova_largura / 2.0 - centro[0]
    matriz[1, 2] += nova_altura / 2.0 - centro[1]

    girada = cv2.warpAffine(
        imagem, matriz, (nova_largura, nova_altura), flags=cv2.INTER_LINEAR
    )
    return girada, matriz


def desespelhar(pontos: np.ndarray) -> np.ndarray:
    """Desfaz um espelhamento horizontal em coordenadas normalizadas."""
    pontos = np.asarray(pontos, dtype=np.float32).copy()
    pontos[:, 0] = 1.0 - pontos[:, 0]
    return pontos


def _equalizar(imagem_rgb: np.ndarray) -> np.ndarray:
    """CLAHE só na luminância — recupera dedos perdidos em sombra ou estouro."""
    lab = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# --- variantes ---


@dataclass(frozen=True)
class Variante:
    """Um jeito de reapresentar a imagem ao detector, com como desfazê-lo.

    `preparar` devolve (imagem, matriz), onde a matriz leva pixels da imagem
    original para pixels da variante — ou None quando a transformação não mexe
    nas coordenadas normalizadas (é o caso de ampliar e de mexer no contraste).
    """

    nome: str
    preparar: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]]
    espelha: bool = field(default=False)

    def desfazer(
        self, pontos: np.ndarray, matriz: np.ndarray | None, variante: Forma, original: Forma
    ) -> np.ndarray:
        if self.espelha:
            pontos = desespelhar(pontos)
        if matriz is not None:
            pontos = aplicar_afim(
                pontos, cv2.invertAffineTransform(matriz), variante, original
            )
        return pontos


def _giro(graus: float) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    return lambda imagem: girar(imagem, graus)


# Ordem por custo e por quanto distorcem: o barato e fiel primeiro. A grande
# maioria das imagens resolve na primeira e nunca chega ao resto.
VARIANTES: tuple[Variante, ...] = (
    Variante("original", lambda imagem: (imagem, None)),
    Variante("espelhada", lambda imagem: (cv2.flip(imagem, 1), None), espelha=True),
    Variante(
        "ampliada",
        lambda imagem: (
            cv2.resize(imagem, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
            None,
        ),
    ),
    Variante("contraste", lambda imagem: (_equalizar(imagem), None)),
    Variante(
        "ampliada+contraste",
        lambda imagem: (
            cv2.resize(
                _equalizar(imagem), None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            ),
            None,
        ),
    ),
    Variante("girada+20", _giro(20.0)),
    Variante("girada-20", _giro(-20.0)),
    Variante("girada+40", _giro(40.0)),
    Variante("girada-40", _giro(-40.0)),
)


def detectar_com_tentativas(
    detector: DetectorDeImagem,
    imagem_rgb: np.ndarray,
    variantes: tuple[Variante, ...] = VARIANTES,
) -> tuple[Deteccao | None, str | None]:
    """Tenta cada variante até uma achar a mão.

    Returns:
        (detecção nas coordenadas da imagem original, nome da variante que
        funcionou) — ou (None, None) se nenhuma funcionou.
    """
    forma_original: Forma = imagem_rgb.shape[:2]

    for variante in variantes:
        imagem, matriz = variante.preparar(imagem_rgb)
        deteccao = detector.detectar(np.ascontiguousarray(imagem))

        if deteccao is None:
            continue

        pontos = variante.desfazer(
            deteccao.pontos, matriz, imagem.shape[:2], forma_original
        )

        # Numa imagem espelhada o MediaPipe reporta a mão trocada.
        mao_esquerda = deteccao.mao_esquerda != variante.espelha

        return (
            Deteccao(
                pontos=pontos.reshape(NUM_PONTOS, 3),
                vetor=normalizar(pontos, mao_esquerda),
                mao_esquerda=mao_esquerda,
            ),
            variante.nome,
        )

    return None, None
