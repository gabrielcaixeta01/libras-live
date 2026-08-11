"""Distância entre sinais com alinhamento temporal — a baseline da fase 2.

O mesmo sinal feito devagar e feito rápido é a mesma palavra. Uma distância
frame a frame diria que não: ela compara o frame 5 de um com o frame 5 do outro,
e se um deles se atrasou, tudo depois disso desalinha. DTW procura o
emparelhamento entre os dois eixos de tempo que minimiza o custo total, então
ritmo diferente sai barato e trajetória diferente sai caro — que é exatamente a
distinção que interessa.

**Por que uma baseline antes do encoder.** O encoder neural traz torch para um
projeto que hoje é scikit-learn puro com modelo de menos de 1MB. Antes de pagar
essa conta é preciso saber quanto ela compra. Este módulo produz o número que o
encoder tem que bater — e se ele não bater por margem clara em
leave-one-articulator-out, o torch não entra. A decisão fica registrada em
`models/relatorio_sinais.txt`.

**Como isto roda em 4.092 candidatos sem ser lento.** A recorrência do DTW é
sequencial em (i, j) e não dá para vetorizar no tempo. Mas dá para vetorizar
**nos candidatos**: o laço Python roda 32×32 = 1.024 vezes, e cada iteração
opera num array de 4.092 posições. Medido em 87 ms para a busca inteira — e é
uma consulta por sinal, não por frame, então ela não entra no orçamento dos
33 ms do loop de vídeo.
"""

from __future__ import annotations

import numpy as np

BANDA_PADRAO = 0.2  # fração do comprimento; ±20% de desvio da diagonal


def distancias(
    consulta: np.ndarray,
    base: np.ndarray,
    banda: float = BANDA_PADRAO,
) -> np.ndarray:
    """Distância DTW de uma consulta para cada candidato, de uma vez só.

    Args:
        consulta: (Tq, D) — a sequência sinalizada agora.
        base: (N, Tb, D) — os protótipos do dicionário.
        banda: largura da banda de Sakoe-Chiba, em fração do comprimento. Ela
            impede alinhamentos absurdos (o começo de um sinal casando com o fim
            do outro) e corta o custo. 1.0 desliga a restrição.

    Returns:
        (N,) float32, normalizado pelo comprimento do caminho para que
        sequências de tamanhos diferentes sejam comparáveis entre si.
    """
    consulta = np.asarray(consulta, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)

    if consulta.ndim != 2:
        raise ValueError(f"consulta deve ser (T, D), recebido {consulta.shape}")
    if base.ndim != 3:
        raise ValueError(f"base deve ser (N, T, D), recebido {base.shape}")
    if consulta.shape[0] == 0 or base.shape[1] == 0:
        raise ValueError("sequência vazia")
    if consulta.shape[1] != base.shape[2]:
        raise ValueError(
            f"dimensões incompatíveis: {consulta.shape[1]} vs {base.shape[2]}"
        )

    custo = _matriz_de_custo(consulta, base)
    _aplicar_banda(custo, banda)

    n_candidatos, t_consulta, t_base = custo.shape
    acumulado = _programacao_dinamica(custo)

    return (acumulado / (t_consulta + t_base)).astype(np.float32)


def distancia(a: np.ndarray, b: np.ndarray, banda: float = BANDA_PADRAO) -> float:
    """Conveniência para um par. Chama `distancias` com um candidato só."""
    b = np.asarray(b, dtype=np.float32)
    if b.ndim != 2:
        raise ValueError(f"b deve ser (T, D), recebido {b.shape}")
    return float(distancias(a, b[None], banda)[0])


def _matriz_de_custo(consulta: np.ndarray, base: np.ndarray) -> np.ndarray:
    """(N, Tq, Tb) com a distância euclidiana de cada par de frames.

    Pela identidade ||a-b||² = ||a||² + ||b||² - 2a·b, para que o trabalho pesado
    caia num produto de matrizes e vá para o BLAS em vez de um laço Python.
    """
    produto = np.einsum("qd,ntd->nqt", consulta, base)
    norma_consulta = (consulta**2).sum(axis=1)   # (Tq,)
    norma_base = (base**2).sum(axis=2)           # (N, Tb)

    quadrado = norma_consulta[None, :, None] + norma_base[:, None, :] - 2.0 * produto
    return np.sqrt(np.maximum(quadrado, 0.0))


def _aplicar_banda(custo: np.ndarray, banda: float) -> None:
    """Marca como infinito o que está longe demais da diagonal. Modifica no lugar.

    A diagonal sempre passa, por menor que seja a banda — senão não haveria
    caminho nenhum e a distância viraria infinito para tudo.
    """
    if banda >= 1.0:
        return

    _, t_consulta, t_base = custo.shape
    largura = max(1, int(round(banda * max(t_consulta, t_base))))

    i = np.arange(t_consulta)[:, None]
    j = np.arange(t_base)[None, :]
    diagonal = i * (t_base - 1) / max(t_consulta - 1, 1)

    custo[:, np.abs(diagonal - j) > largura] = np.inf


def _programacao_dinamica(custo: np.ndarray) -> np.ndarray:
    """Custo mínimo acumulado do canto (0,0) ao canto (Tq-1, Tb-1).

    O laço é sequencial em (i, j) porque cada célula depende da vizinha à
    esquerda — mas roda sobre todos os candidatos ao mesmo tempo.
    """
    n_candidatos, t_consulta, t_base = custo.shape

    anterior = np.empty((n_candidatos, t_base), dtype=np.float64)
    atual = np.empty_like(anterior)

    anterior[:] = np.cumsum(custo[:, 0, :], axis=1)

    for i in range(1, t_consulta):
        atual[:, 0] = anterior[:, 0] + custo[:, i, 0]
        for j in range(1, t_base):
            melhor = np.minimum(
                np.minimum(anterior[:, j], anterior[:, j - 1]), atual[:, j - 1]
            )
            atual[:, j] = custo[:, i, j] + melhor
        anterior, atual = atual, anterior

    return anterior[:, -1]
