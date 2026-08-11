"""Cores, fonte e as duas ou três primitivas que todo overlay usa.

Os dois modos desenham coisas completamente diferentes — o alfabeto tem faixa de
letras e texto em construção, o dicionário tem lista de candidatos e barras de
similaridade —, mas ambos escurecem faixas para o texto ficar legível, medem
largura de texto e mostram os fps. Isso mora aqui para que os dois combinem sem
que um dependa do outro.

As strings desenhadas são ASCII porque as fontes Hershey do OpenCV não têm
glifos acentuados — acento vira "?" na tela.
"""

from __future__ import annotations

import cv2
import numpy as np

FONTE = cv2.FONT_HERSHEY_SIMPLEX

VERDE = (80, 220, 100)
CINZA = (150, 150, 150)
CINZA_ESCURO = (90, 90, 90)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
AZUL = (240, 180, 60)
AMARELO = (60, 200, 240)
VERMELHO = (70, 70, 235)


def faixa(frame: np.ndarray, y0: int, y1: int, opacidade: float = 0.55) -> None:
    """Escurece uma faixa horizontal para o texto ficar legível sobre a imagem."""
    y0, y1 = max(0, y0), min(frame.shape[0], y1)
    recorte = frame[y0:y1]
    escura = np.zeros_like(recorte)
    frame[y0:y1] = cv2.addWeighted(recorte, 1 - opacidade, escura, opacidade, 0)


def largura_texto(texto: str, escala: float, espessura: int) -> int:
    (largura, _), _ = cv2.getTextSize(texto, FONTE, escala, espessura)
    return largura


def desenhar_fps(frame: np.ndarray, fps: float) -> None:
    largura = frame.shape[1]
    cv2.putText(
        frame, f"{fps:.0f} fps", (largura - 110, 40), FONTE, 0.7, CINZA, 2, cv2.LINE_AA
    )
