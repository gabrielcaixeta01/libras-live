"""O desenho do modo dicionário. Fino, como `ui.py` — não decide nada.

A tela tem dois estados e eles não se sobrepõem: ou você está sinalizando (e o
app mostra que está gravando), ou o resultado está no ar. Um dicionário que
mostra o candidato anterior enquanto você faz o próximo sinal é um dicionário
que você não consegue ler.
"""

from __future__ import annotations

import cv2
import numpy as np

from .dicionario import Candidato
from .segmenter import Estado
from .desenho import AMARELO, AZUL, BRANCO, CINZA, CINZA_ESCURO, FONTE, PRETO, VERDE, faixa

ALTURA_LINHA = 34
LARGURA_BARRA = 180


def desenhar_esqueleto(
    frame: np.ndarray,
    maos: list[np.ndarray],
    corpo: np.ndarray | None,
) -> None:
    """As mãos e a linha dos ombros — a âncora que a normalização usa.

    Desenhar os ombros não é enfeite: quando o resultado sai errado, é o
    primeiro lugar onde se vê o motivo. Sem ombros detectados não existe
    normalização, e o app fica surdo sem explicar por quê.
    """
    altura, largura = frame.shape[:2]

    for pontos in maos:
        for x, y, _ in pontos:
            cv2.circle(frame, (int(x * largura), int(y * altura)), 3, VERDE, -1)

    if corpo is None:
        return

    from . import pose

    ombros = [corpo[pose.POSE_OMBRO_ESQUERDO], corpo[pose.POSE_OMBRO_DIREITO]]
    pixels = [(int(p[0] * largura), int(p[1] * altura)) for p in ombros]
    cv2.line(frame, pixels[0], pixels[1], AZUL, 2)
    for ponto in pixels:
        cv2.circle(frame, ponto, 5, AZUL, -1)


def desenhar_estado(
    frame: np.ndarray,
    estado: Estado,
    progresso: float,
    velocidade: float,
) -> None:
    """Canto superior esquerdo: gravando ou esperando, e quanto já foi gravado."""
    faixa(frame, 0, 66)

    sinalizando = estado is Estado.SINALIZANDO
    texto = "GRAVANDO" if sinalizando else "faça um sinal"
    cor = VERDE if sinalizando else CINZA

    cv2.putText(frame, texto, (16, 32), FONTE, 0.7, cor, 2, cv2.LINE_AA)

    x0, y0 = 16, 44
    cv2.rectangle(frame, (x0, y0), (x0 + LARGURA_BARRA, y0 + 10), CINZA_ESCURO, -1)
    if sinalizando:
        preenchido = int(LARGURA_BARRA * min(progresso, 1.0))
        cv2.rectangle(frame, (x0, y0), (x0 + preenchido, y0 + 10), VERDE, -1)

    cv2.putText(
        frame,
        f"{velocidade:.2f}",
        (x0 + LARGURA_BARRA + 12, y0 + 10),
        FONTE,
        0.45,
        CINZA,
        1,
        cv2.LINE_AA,
    )


def desenhar_candidatos(
    frame: np.ndarray,
    candidatos: list[Candidato],
    ajuda: str,
) -> None:
    """Os cinco candidatos, com a barra de similaridade de cada um.

    A barra é uma medida relativa e nada mais — está escrito na tela. 0,81 não
    é "81% de chance de ser CASA", é "foi o mais perto, e por esta margem".
    Chamar isso de probabilidade seria mentir com um número bonito.

    Lista vazia não passa por aqui: quem não tem o que mostrar chama
    `desenhar_aviso`, porque "não reconheci" é uma resposta e merece o meio da
    tela, não uma faixa vazia no rodapé.
    """
    if not candidatos:
        return

    altura, largura = frame.shape[:2]
    topo = altura - (len(candidatos) * ALTURA_LINHA) - 58
    faixa(frame, topo, altura, opacidade=0.7)

    for i, candidato in enumerate(candidatos):
        y = topo + 28 + i * ALTURA_LINHA
        _linha_de_candidato(frame, candidato, y, largura, primeiro=i == 0)

    cv2.putText(
        frame, ajuda, (16, altura - 14), FONTE, 0.45, CINZA, 1, cv2.LINE_AA
    )


def _linha_de_candidato(
    frame: np.ndarray,
    candidato: Candidato,
    y: int,
    largura: int,
    primeiro: bool,
) -> None:
    cor = VERDE if primeiro else BRANCO
    escala = 0.85 if primeiro else 0.65
    espessura = 2 if primeiro else 1

    cv2.putText(frame, candidato.rotulo, (16, y), FONTE, escala, cor, espessura,
                cv2.LINE_AA)

    x0 = largura - LARGURA_BARRA - 90
    cv2.rectangle(frame, (x0, y - 12), (x0 + LARGURA_BARRA, y), CINZA_ESCURO, -1)
    preenchido = int(LARGURA_BARRA * min(max(candidato.similaridade, 0.0), 1.0))
    cv2.rectangle(frame, (x0, y - 12), (x0 + preenchido, y), cor, -1)

    cv2.putText(
        frame,
        f"{candidato.similaridade:.2f}",
        (x0 + LARGURA_BARRA + 10, y),
        FONTE,
        0.5,
        CINZA,
        1,
        cv2.LINE_AA,
    )


def desenhar_aviso(frame: np.ndarray, texto: str) -> None:
    """Mensagem centrada — usada quando falta pose ou dicionário."""
    altura, largura = frame.shape[:2]
    escala, espessura = 0.7, 2
    (w, h), _ = cv2.getTextSize(texto, FONTE, escala, espessura)
    x, y = (largura - w) // 2, altura // 2

    cv2.rectangle(frame, (x - 12, y - h - 12), (x + w + 12, y + 12), PRETO, -1)
    cv2.putText(frame, texto, (x, y), FONTE, escala, AMARELO, espessura, cv2.LINE_AA)
