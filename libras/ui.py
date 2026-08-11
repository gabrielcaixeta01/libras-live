"""Desenho do overlay sobre o frame.

Fino de propósito: só desenha o que recebe pronto, não decide nada.

As strings desenhadas são ASCII porque as fontes Hershey do OpenCV não têm
glifos acentuados — acento vira "?" na tela.
"""

from __future__ import annotations

import cv2
import numpy as np

from .landmarks import CONEXOES_MAO

FONTE = cv2.FONT_HERSHEY_SIMPLEX

VERDE = (80, 220, 100)
CINZA = (150, 150, 150)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
AZUL = (240, 180, 60)


def desenhar_mao(frame: np.ndarray, pontos: np.ndarray) -> None:
    """Desenha o esqueleto da mão. Os pontos vêm normalizados em 0..1."""
    altura, largura = frame.shape[:2]
    pixels = [(int(x * largura), int(y * altura)) for x, y, _ in pontos]

    for a, b in CONEXOES_MAO:
        cv2.line(frame, pixels[a], pixels[b], VERDE, 2, cv2.LINE_AA)

    for px in pixels:
        cv2.circle(frame, px, 4, BRANCO, -1, cv2.LINE_AA)


def _faixa(frame: np.ndarray, y0: int, y1: int, opacidade: float = 0.55) -> None:
    """Escurece uma faixa horizontal para o texto ficar legível sobre a imagem."""
    recorte = frame[y0:y1]
    escura = np.zeros_like(recorte)
    frame[y0:y1] = cv2.addWeighted(recorte, 1 - opacidade, escura, opacidade, 0)


def desenhar_predicao(
    frame: np.ndarray,
    letra: str | None,
    confianca: float,
    confirmada: bool,
    preenchimento: float,
) -> None:
    """Canto superior esquerdo: letra atual, confiança e progresso do buffer."""
    _faixa(frame, 0, 110)

    if letra is None:
        cv2.putText(frame, "sem mao", (20, 70), FONTE, 1.2, CINZA, 2, cv2.LINE_AA)
        return

    cor = VERDE if confirmada else CINZA
    cv2.putText(frame, letra, (20, 85), FONTE, 2.6, cor, 5, cv2.LINE_AA)
    cv2.putText(
        frame, f"{confianca * 100:.0f}%", (110, 85), FONTE, 1.0, cor, 2, cv2.LINE_AA
    )

    # Barra de progresso do buffer de estabilização.
    x0, y0, w, h = 20, 95, 200, 8
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), CINZA, 1)
    cv2.rectangle(
        frame, (x0, y0), (x0 + int(w * preenchimento), y0 + h), AZUL, -1
    )


def desenhar_texto(frame: np.ndarray, texto: str) -> None:
    """Rodapé: o texto soletrado até agora."""
    altura, largura = frame.shape[:2]
    _faixa(frame, altura - 110, altura)

    # Mostra só o fim, para não estourar a largura da janela.
    visivel = texto[-32:]
    cv2.putText(
        frame, visivel or "...", (20, altura - 55), FONTE, 1.4, BRANCO, 3, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "ESC sair   BACKSPACE apagar   C limpar",
        (20, altura - 18),
        FONTE,
        0.55,
        CINZA,
        1,
        cv2.LINE_AA,
    )


def desenhar_fps(frame: np.ndarray, fps: float) -> None:
    largura = frame.shape[1]
    cv2.putText(
        frame, f"{fps:.0f} fps", (largura - 110, 40), FONTE, 0.7, CINZA, 2, cv2.LINE_AA
    )
