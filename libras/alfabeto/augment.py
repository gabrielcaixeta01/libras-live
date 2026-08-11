"""Aumento de dados no espaço dos landmarks.

O modelo é treinado com fotos de estúdio e usado na sua webcam. As duas
distribuições não são a mesma: na base pública a mão está de frente, centrada e
parada; na sua mesa ela chega torta, meio de lado e tremendo. Este módulo
fabrica essa variação a partir do que já existe, sem gravar nada.

Aumentar aqui, e não na imagem, é o que torna isso barato: são 63 floats por
amostra, então gerar dez mil variações custa milissegundos e nenhum disco. É
também o único ponto onde dá para aumentar N, que tem 37 amostras na base
pública contra 568 de B.

Três transformações, todas aplicadas **depois** da normalização e seguidas de
uma renormalização, para que a amostra sintética obedeça às mesmas invariantes
que uma real (pulso na origem, raio máximo 1):

- `rotacionar`: pulso torto e mão fora do eixo da câmera.
- `perturbar`: o jitter que o MediaPipe tem frame a frame.
- `escalar_profundidade`: o z do MediaPipe é uma estimativa relativa e varia
  bastante com a distância; comprimi-lo ou esticá-lo simula esse erro.

Nada aqui simula *escala* ou *translação*: `normalizar` já as elimina, então
seriam operações nulas.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .landmarks import NUM_PONTOS, TAMANHO_VETOR, normalizar


def _como_pontos(vetor: np.ndarray) -> np.ndarray:
    """Aceita (63,) ou (21, 3) e devolve sempre (21, 3) float32."""
    pontos = np.asarray(vetor, dtype=np.float32)
    if pontos.shape == (TAMANHO_VETOR,):
        pontos = pontos.reshape(NUM_PONTOS, 3)
    if pontos.shape != (NUM_PONTOS, 3):
        raise ValueError(f"esperado (63,) ou (21, 3), recebido {pontos.shape}")
    return pontos.copy()


def _matriz_rotacao(graus_x: float, graus_y: float, graus_z: float) -> np.ndarray:
    """Rotação extrínseca X→Y→Z."""
    ax, ay, az = np.radians([graus_x, graus_y, graus_z])

    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return rz @ ry @ rx


def rotacionar(
    vetor: np.ndarray, graus_x: float, graus_y: float, graus_z: float
) -> np.ndarray:
    """Gira a mão em torno do pulso. Devolve um vetor (63,) renormalizado."""
    pontos = _como_pontos(vetor)
    girados = pontos @ _matriz_rotacao(graus_x, graus_y, graus_z).T
    return normalizar(girados.astype(np.float32))


def perturbar(
    vetor: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Soma ruído gaussiano independente por coordenada.

    `sigma` está na escala do vetor normalizado, onde a distância do pulso ao
    dedo mais longe vale 1 — então 0,02 é 2% do tamanho da mão.
    """
    pontos = _como_pontos(vetor)
    if sigma > 0:
        pontos += rng.normal(0.0, sigma, size=pontos.shape).astype(np.float32)
    return normalizar(pontos)


def escalar_profundidade(vetor: np.ndarray, fator: float) -> np.ndarray:
    """Comprime (fator < 1) ou estica (fator > 1) o eixo z."""
    pontos = _como_pontos(vetor)
    pontos[:, 2] *= fator
    return normalizar(pontos)


def variar(
    vetor: np.ndarray,
    rng: np.random.Generator,
    rotacao_graus: float = config.AUG_ROTACAO_GRAUS,
    ruido: float = config.AUG_RUIDO,
    profundidade: float = config.AUG_PROFUNDIDADE,
) -> np.ndarray:
    """Uma variação aleatória de uma amostra: as três transformações em sequência.

    Os limites vêm de `config` e são deliberadamente modestos. Rotação demais
    transforma um M num W de verdade — o aumento tem que gerar a mesma letra
    vista de outro jeito, não outra letra.
    """
    girado = rotacionar(
        vetor,
        rng.uniform(-rotacao_graus, rotacao_graus),
        rng.uniform(-rotacao_graus, rotacao_graus),
        rng.uniform(-rotacao_graus, rotacao_graus),
    )
    fundo = escalar_profundidade(girado, rng.uniform(1 - profundidade, 1 + profundidade))
    return perturbar(fundo, ruido, rng)


def equilibrar(
    X: np.ndarray,
    y: np.ndarray,
    alvo: int = config.AUG_ALVO_POR_CLASSE,
    semente: int = 42,
    **limites: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Completa cada classe até `alvo` amostras com variações sintéticas.

    Classes que já passam do alvo ficam intactas — o objetivo é levantar o piso,
    não rebaixar o teto. As amostras originais são sempre preservadas.

    Chame isto **só no conjunto de treino**. Aumentar antes de separar treino e
    teste vaza a mesma mão para os dois lados e infla a métrica.

    Returns:
        (X, y) com as originais primeiro e as sintéticas em seguida.
    """
    if alvo < 1:
        raise ValueError("alvo deve ser >= 1")

    rng = np.random.default_rng(semente)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    novos: list[np.ndarray] = []
    rotulos: list[str] = []

    for letra in sorted(set(y.tolist())):
        indices = np.flatnonzero(y == letra)
        faltam = alvo - len(indices)
        if faltam <= 0:
            continue

        escolhidos = rng.choice(indices, size=faltam, replace=True)
        for indice in escolhidos:
            novos.append(variar(X[indice], rng, **limites))
            rotulos.append(letra)

    if not novos:
        return X, y

    return (
        np.vstack([X, np.array(novos, dtype=np.float32)]),
        np.concatenate([y, np.array(rotulos)]),
    )
