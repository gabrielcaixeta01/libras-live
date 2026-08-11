"""Desenho do overlay sobre o frame.

Fino de propósito: só desenha o que recebe pronto, não decide nada.

As strings desenhadas são ASCII porque as fontes Hershey do OpenCV não têm
glifos acentuados — acento vira "?" na tela.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..desenho import (  # noqa: F401  (reexportados: training/collect.py usa)
    AMARELO,
    AZUL,
    BRANCO,
    CINZA,
    CINZA_ESCURO,
    FONTE,
    PRETO,
    VERDE,
    VERMELHO,
    desenhar_fps,
    faixa as _faixa,
    largura_texto as _largura_texto,
)
from .classifier import Predicao
from .landmarks import CONEXOES_MAO
from .practice import Pratica


def desenhar_mao(frame: np.ndarray, pontos: np.ndarray) -> None:
    """Desenha o esqueleto da mão. Os pontos vêm normalizados em 0..1."""
    altura, largura = frame.shape[:2]
    pixels = [(int(x * largura), int(y * altura)) for x, y, _ in pontos]

    for a, b in CONEXOES_MAO:
        cv2.line(frame, pixels[a], pixels[b], VERDE, 2, cv2.LINE_AA)

    for px in pixels:
        cv2.circle(frame, px, 4, BRANCO, -1, cv2.LINE_AA)



def desenhar_predicao(
    frame: np.ndarray,
    predicao: Predicao | None,
    confirmada: bool,
    preenchimento: float,
    mao_esquerda: bool | None = None,
) -> None:
    """Canto superior esquerdo: letra atual, confiança e progresso do buffer.

    Uma predição rejeitada aparece como "?" em amarelo, com a letra que o modelo
    chutou logo ao lado — útil para entender por que ele hesitou.
    """
    _faixa(frame, 0, 110)

    if predicao is None:
        cv2.putText(frame, "sem mao", (20, 70), FONTE, 1.2, CINZA, 2, cv2.LINE_AA)
        return

    if not predicao.aceita:
        cor = AMARELO
    elif confirmada:
        cor = VERDE
    else:
        cor = CINZA

    cv2.putText(frame, predicao.rotulo, (20, 85), FONTE, 2.6, cor, 5, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{predicao.confianca * 100:.0f}%",
        (110, 85),
        FONTE,
        1.0,
        cor,
        2,
        cv2.LINE_AA,
    )

    if not predicao.aceita:
        cv2.putText(
            frame,
            f"talvez {predicao.letra}",
            (110, 55),
            FONTE,
            0.6,
            CINZA,
            1,
            cv2.LINE_AA,
        )

    if mao_esquerda is not None:
        cv2.putText(
            frame,
            "mao esq" if mao_esquerda else "mao dir",
            (240, 85),
            FONTE,
            0.6,
            CINZA,
            1,
            cv2.LINE_AA,
        )

    # Barra de progresso do buffer de estabilização.
    x0, y0, w, h = 20, 95, 200, 8
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), CINZA, 1)
    cv2.rectangle(frame, (x0, y0), (x0 + int(w * preenchimento), y0 + h), AZUL, -1)


def desenhar_alfabeto(
    frame: np.ndarray, conhecidas: list[str], ausentes: list[str]
) -> None:
    """Faixa com as 26 letras: apagadas são as que o modelo não sabe prever.

    Sem isto o app mente por omissão — você sinaliza um F, sai um E, e não há
    nada na tela dizendo que F nunca esteve disponível.
    """
    if not ausentes:
        return

    largura = frame.shape[1]
    letras = sorted(set(conhecidas) | set(ausentes))
    passo = min(38, (largura - 60) // max(len(letras), 1))
    x0 = 20
    y = 138

    _faixa(frame, 112, 152, opacidade=0.45)

    for indice, letra in enumerate(letras):
        conhecida = letra in conhecidas
        cv2.putText(
            frame,
            letra,
            (x0 + indice * passo, y),
            FONTE,
            0.65,
            BRANCO if conhecida else CINZA_ESCURO,
            2 if conhecida else 1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"apagadas: sem dados ({len(ausentes)})",
        (x0 + len(letras) * passo + 10, y),
        FONTE,
        0.45,
        CINZA_ESCURO,
        1,
        cv2.LINE_AA,
    )


def desenhar_texto(frame: np.ndarray, texto: str, ajuda: str) -> None:
    """Rodapé: o texto soletrado, com a palavra em andamento destacada."""
    altura, largura = frame.shape[:2]
    _faixa(frame, altura - 110, altura)

    # Mostra só o fim, para não estourar a largura da janela.
    visivel = texto[-32:]

    if not visivel:
        cv2.putText(
            frame, "...", (20, altura - 55), FONTE, 1.4, CINZA, 3, cv2.LINE_AA
        )
    else:
        # A palavra atual sai em verde: é a que ainda dá para corrigir.
        corte = visivel.rfind(" ") + 1
        anterior, palavra = visivel[:corte], visivel[corte:]

        x = 20
        if anterior:
            cv2.putText(
                frame, anterior, (x, altura - 55), FONTE, 1.4, BRANCO, 3, cv2.LINE_AA
            )
            x += _largura_texto(anterior, 1.4, 3)
        if palavra:
            cv2.putText(
                frame, palavra, (x, altura - 55), FONTE, 1.4, VERDE, 3, cv2.LINE_AA
            )

    cv2.putText(frame, ajuda, (20, altura - 18), FONTE, 0.55, CINZA, 1, cv2.LINE_AA)


def desenhar_pratica(
    frame: np.ndarray, sessao: Pratica, erro: str | None, ajuda: str
) -> None:
    """Painel do modo prática: a letra pedida, o placar e o último erro."""
    altura, largura = frame.shape[:2]
    _faixa(frame, altura - 150, altura)

    if sessao.concluida:
        cv2.putText(
            frame, "fim!", (20, altura - 85), FONTE, 1.6, VERDE, 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, sessao.resumo(), (20, altura - 45), FONTE, 0.7, BRANCO, 2, cv2.LINE_AA
        )
        cv2.putText(frame, ajuda, (20, altura - 15), FONTE, 0.55, CINZA, 1, cv2.LINE_AA)
        return

    cv2.putText(
        frame, "faca a letra", (20, altura - 100), FONTE, 0.7, CINZA, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, str(sessao.alvo), (20, altura - 30), FONTE, 2.8, VERDE, 6, cv2.LINE_AA
    )

    placar = (
        f"rodada {sessao.rodada}/{sessao.total}   "
        f"acertos {sessao.acertos}   erros {sessao.erros}"
    )
    cv2.putText(frame, placar, (140, altura - 75), FONTE, 0.7, BRANCO, 2, cv2.LINE_AA)

    if erro:
        cv2.putText(
            frame,
            f"saiu {erro}",
            (140, altura - 40),
            FONTE,
            0.8,
            VERMELHO,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame, ajuda, (largura - 430, altura - 15), FONTE, 0.55, CINZA, 1, cv2.LINE_AA
    )

