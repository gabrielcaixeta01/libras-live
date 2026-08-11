"""Os 49 pontos de um sinal, e a normalização ancorada no corpo.

Esta é a inversão em relação ao alfabeto. `landmarks.normalizar` põe o **pulso**
na origem, o que apaga onde a mão está — e para uma letra isso é exatamente o
certo, porque a letra é só a forma da mão.

Para um sinal é errado. Em Libras a *localização* é fonema: PAI e MÃE têm a
mesma configuração de mão em lugares diferentes do rosto. Uma normalização que
apaga a localização apaga a diferença entre as duas palavras.

Então aqui a âncora é o corpo:

| | alfabeto | sinal |
|---|---|---|
| origem | pulso | ponto médio dos ombros |
| escala | maior distância ao pulso | distância entre os ombros |
| preserva | forma da mão | forma + localização |

A escala usa só x e y. O z do PoseLandmarker é uma estimativa fraca de
profundidade, e deixá-lo entrar na escala faria a normalização inteira tremer
junto com ele.

**Por que 49 pontos e não 543.** MediaPipe entrega 468 pontos de face + 33 de
pose + 21 por mão. Usar tudo piora: sobre três amostras por sinal, cada
coordenada a mais é uma chance a mais de decorar ruído. O subconjunto aqui —
duas mãos inteiras, mais nariz, ombros, cotovelos e punhos — segue o resultado
de arXiv:2510.24887, que bate o SOTA em Libras isolada com 5x menos tempo
justamente por selecionar landmarks.
"""

from __future__ import annotations

import numpy as np

NUM_PONTOS_MAO = 21

# Índices no esqueleto de 33 pontos do PoseLandmarker.
POSE_NARIZ = 0
POSE_OMBRO_ESQUERDO = 11
POSE_OMBRO_DIREITO = 12
POSE_COTOVELO_ESQUERDO = 13
POSE_COTOVELO_DIREITO = 14
POSE_PUNHO_ESQUERDO = 15
POSE_PUNHO_DIREITO = 16

INDICES_POSE: tuple[int, ...] = (
    POSE_NARIZ,
    POSE_OMBRO_ESQUERDO,
    POSE_OMBRO_DIREITO,
    POSE_COTOVELO_ESQUERDO,
    POSE_COTOVELO_DIREITO,
    POSE_PUNHO_ESQUERDO,
    POSE_PUNHO_DIREITO,
)
NUM_PONTOS_POSE = len(INDICES_POSE)
NUM_PONTOS = 2 * NUM_PONTOS_MAO + NUM_PONTOS_POSE  # 49
TAMANHO_VETOR = NUM_PONTOS * 3                     # 147

# Fatias e índices dentro do frame montado (49, 3).
MAO_ESQUERDA = slice(0, NUM_PONTOS_MAO)
MAO_DIREITA = slice(NUM_PONTOS_MAO, 2 * NUM_PONTOS_MAO)
POSE = slice(2 * NUM_PONTOS_MAO, NUM_PONTOS)

_BASE_POSE = 2 * NUM_PONTOS_MAO
NARIZ = _BASE_POSE + 0
OMBRO_ESQUERDO = _BASE_POSE + 1
OMBRO_DIREITO = _BASE_POSE + 2
COTOVELO_ESQUERDO = _BASE_POSE + 3
COTOVELO_DIREITO = _BASE_POSE + 4
PUNHO_ESQUERDO = _BASE_POSE + 5
PUNHO_DIREITO = _BASE_POSE + 6

# Para onde cada ponto vai quando o frame é espelhado: as mãos trocam de slot e
# os pares do corpo trocam de lado. Fica como permutação (e não como uma série
# de swaps) porque a máscara de validade em `sequencia` precisa sofrer
# exatamente a mesma troca — se ela não acompanhar, o modelo passa a achar que
# mediu a mão que na verdade faltou.
PERMUTACAO_ESPELHO: np.ndarray = np.concatenate(
    [
        np.arange(NUM_PONTOS_MAO) + NUM_PONTOS_MAO,  # slot esquerdo ← mão direita
        np.arange(NUM_PONTOS_MAO),                   # slot direito ← mão esquerda
        [
            NARIZ,
            OMBRO_DIREITO,
            OMBRO_ESQUERDO,
            COTOVELO_DIREITO,
            COTOVELO_ESQUERDO,
            PUNHO_DIREITO,
            PUNHO_ESQUERDO,
        ],
    ]
).astype(np.intp)

ESCALA_MINIMA = 1e-4  # abaixo disso os "ombros" são ruído, não um corpo


def montar_frame(
    mao_esquerda: np.ndarray | None = None,
    mao_direita: np.ndarray | None = None,
    corpo: np.ndarray | None = None,
) -> np.ndarray:
    """Junta o que o MediaPipe achou num frame (49, 3), com NaN no que faltou.

    NaN é a representação de ausência em todo o pacote: ele se propaga sozinho
    pelas contas em vez de virar um zero que o modelo leria como "mão colada na
    origem". Quem resolve a ausência é `sequencia.imputar`, que tem os frames
    vizinhos para isso — este módulo só registra.

    Args:
        mao_esquerda: (21, 3) do HandLandmarker, ou None.
        mao_direita: (21, 3) do HandLandmarker, ou None.
        corpo: (33, 3) do PoseLandmarker, ou None. É reduzido a `INDICES_POSE`.
    """
    frame = np.full((NUM_PONTOS, 3), np.nan, dtype=np.float32)

    for pontos, destino in ((mao_esquerda, MAO_ESQUERDA), (mao_direita, MAO_DIREITA)):
        if pontos is None:
            continue
        pontos = np.asarray(pontos, dtype=np.float32)
        if pontos.shape != (NUM_PONTOS_MAO, 3):
            raise ValueError(f"esperado shape (21, 3), recebido {pontos.shape}")
        frame[destino] = pontos

    if corpo is not None:
        corpo = np.asarray(corpo, dtype=np.float32)
        if corpo.ndim != 2 or corpo.shape[1] != 3 or corpo.shape[0] <= max(INDICES_POSE):
            raise ValueError(f"esperado shape (33, 3), recebido {corpo.shape}")
        frame[POSE] = corpo[list(INDICES_POSE)]

    return frame


def normalizar_frame(frame: np.ndarray, espelhar: bool = False) -> np.ndarray:
    """Ancora um frame no corpo: ombros na origem, largura dos ombros = 1.

    Sem ombros não há âncora, e um frame sem âncora não é comparável a nenhum
    outro — então ele volta inteiro como NaN, e a imputação decide o que fazer
    com ele. Fingir uma escala seria pior que admitir a falta.

    Args:
        frame: (49, 3) montado por `montar_frame`.
        espelhar: canoniza um sinalizante canhoto como destro.

    Returns:
        (49, 3) normalizado, com os NaN de ausência preservados.
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (NUM_PONTOS, 3):
        raise ValueError(f"esperado shape (49, 3), recebido {frame.shape}")

    esquerdo = frame[OMBRO_ESQUERDO]
    direito = frame[OMBRO_DIREITO]
    if np.isnan(esquerdo).any() or np.isnan(direito).any():
        return np.full_like(frame, np.nan)

    escala = float(np.linalg.norm(direito[:2] - esquerdo[:2]))
    if escala < ESCALA_MINIMA:
        return np.full_like(frame, np.nan)

    origem = (esquerdo + direito) / 2.0
    normalizado = (frame - origem) / escala

    return espelhar_frame(normalizado) if espelhar else normalizado


def normalizar_sequencia(sequencia: np.ndarray, espelhar: bool = False) -> np.ndarray:
    """`normalizar_frame` aplicado a (T, 49, 3).

    Frame a frame de propósito: a escala é recalculada em cada um, então o
    sinalizante pode se aproximar da câmera no meio do sinal sem que a
    trajetória se deforme.
    """
    sequencia = np.asarray(sequencia, dtype=np.float32)
    if sequencia.ndim != 3 or sequencia.shape[1:] != (NUM_PONTOS, 3):
        raise ValueError(f"esperado shape (T, 49, 3), recebido {sequencia.shape}")

    return np.stack([normalizar_frame(f, espelhar) for f in sequencia])


def espelhar_frame(frame: np.ndarray) -> np.ndarray:
    """Reflete o frame no plano sagital: inverte x e troca os lados do corpo.

    Só faz sentido depois da normalização, quando a origem já está no meio dos
    ombros — é esse ponto que define o plano do espelho.

    Serve para duas coisas: canonizar um sinalizante canhoto como destro, e
    dobrar o dataset no aumento sem inventar geometria nenhuma.
    """
    espelhado = np.asarray(frame, dtype=np.float32)[PERMUTACAO_ESPELHO].copy()
    espelhado[:, 0] *= -1.0
    return espelhado


def espelhar_sequencia(sequencia: np.ndarray) -> np.ndarray:
    return np.stack([espelhar_frame(f) for f in sequencia])


def vetores(sequencia: np.ndarray) -> np.ndarray:
    """(T, 49, 3) → (T, 147), o formato que a métrica de busca consome."""
    sequencia = np.asarray(sequencia, dtype=np.float32)
    return sequencia.reshape(sequencia.shape[0], TAMANHO_VETOR)
