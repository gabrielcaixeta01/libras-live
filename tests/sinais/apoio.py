"""Corpos e mãos sintéticos para os testes de sinais.

Tudo aqui é geometria inventada, mas plausível: as medidas do corpo são
múltiplos da largura dos ombros, então afastar da câmera encolhe o corpo
inteiro em vez de deformá-lo.
"""

from __future__ import annotations

import numpy as np

from libras.sinais import pose


def mao(valor: float = 0.5) -> np.ndarray:
    return np.full((pose.NUM_PONTOS_MAO, 3), valor, dtype=np.float32)


def corpo(largura_ombros: float = 0.2, centro=(0.5, 0.5, 0.0)) -> np.ndarray:
    """Uma pose (33, 3) plausível: ombros simétricos em torno de `centro`."""
    p = np.zeros((33, 3), dtype=np.float32)
    cx, cy, cz = centro
    L = largura_ombros
    p[pose.POSE_NARIZ] = (cx, cy - 0.75 * L, cz)
    p[pose.POSE_OMBRO_ESQUERDO] = (cx - L / 2, cy, cz)
    p[pose.POSE_OMBRO_DIREITO] = (cx + L / 2, cy, cz)
    p[pose.POSE_COTOVELO_ESQUERDO] = (cx - L, cy + 0.75 * L, cz)
    p[pose.POSE_COTOVELO_DIREITO] = (cx + L, cy + 0.75 * L, cz)
    p[pose.POSE_PUNHO_ESQUERDO] = (cx - L, cy + 1.5 * L, cz)
    p[pose.POSE_PUNHO_DIREITO] = (cx + L, cy + 1.5 * L, cz)
    return p


def frame(**kwargs) -> np.ndarray:
    base = {"mao_esquerda": mao(0.4), "mao_direita": mao(0.6), "corpo": corpo()}
    return pose.montar_frame(**{**base, **kwargs})


def gravacao(n_frames: int = 10, **kwargs) -> np.ndarray:
    """(T, 49, 3) parada no tempo — para estragar de propósito nos testes."""
    return np.stack([frame(**kwargs) for _ in range(n_frames)])
