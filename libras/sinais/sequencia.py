"""De uma gravação crua para a sequência comparável que a busca consome.

Quatro problemas, nesta ordem, e todos vêm do mesmo lugar: um sinal é medido ao
longo do tempo, e a medição falha em pedaços.

1. **Buracos.** O MediaPipe perde a mão quando ela cruza o corpo, sai de quadro
   ou fecha. Num frame isolado isso é fatal; numa sequência não, porque os
   vizinhos no tempo sabem onde a mão estava — daí a imputação por spline.
2. **O que foi medido e o que foi inventado.** A imputação não pode se disfarçar
   de medição. A máscara de validade acompanha a sequência inteira dizendo, para
   cada ponto de cada frame, se aquilo saiu da câmera ou da interpolação.
3. **Durações diferentes.** O mesmo sinal sai em 0,8s ou 2,1s. Tudo é reamostrado
   para `T_PADRAO` frames — esticar e encolher, nunca cortar, porque cortar
   perde o fim dos sinais longos, que é onde muitos deles se distinguem.
4. **Frames sem âncora.** Sem os dois ombros não há normalização possível. Esses
   frames voltam como NaN de `pose.normalizar_sequencia` e são reconstruídos
   pelos vizinhos como qualquer outro buraco.

A imputação por spline não é enfeite: o paper que embasa o subconjunto de
landmarks (arXiv:2510.24887) mede ganho substancial de acurácia só com ela.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from . import pose

T_PADRAO = 32  # frames por sinal depois da reamostragem; ~1s a 30fps
MINIMO_PARA_SPLINE = 4  # abaixo disso a cúbica não tem apoio e vira linear


@dataclass(frozen=True)
class Sequencia:
    """Um sinal pronto para comparação.

    Attributes:
        pontos: (T, 49, 3) normalizado no corpo, sem NaN.
        validade: (T, 49) em [0, 1] — 1 é medido, 0 é inventado. Valores
            fracionários aparecem depois da reamostragem, quando um frame novo
            cai entre um medido e um inventado.
    """

    pontos: np.ndarray
    validade: np.ndarray

    @property
    def vetores(self) -> np.ndarray:
        """(T, 147) — o formato que as métricas de busca consomem."""
        return pose.vetores(self.pontos)

    def __len__(self) -> int:
        return len(self.pontos)


def validade(bruta: np.ndarray) -> np.ndarray:
    """(T, 49, 3) → (T, 49) booleano. Um ponto vale se as três coordenadas valem."""
    return np.isfinite(np.asarray(bruta, dtype=np.float32)).all(axis=2)


def imputar(bruta: np.ndarray) -> np.ndarray:
    """Preenche os NaN interpolando cada coordenada ao longo do tempo.

    Spline cúbica quando há apoio suficiente, linear quando não há, e o valor
    da borda quando o buraco está nas pontas — **nunca extrapolação**. Spline
    cúbica extrapolada dispara para longe em poucos frames, e uma mão inventada
    a dois metros do corpo estraga toda distância que ela tocar.

    Canal sem nenhum valor válido vira zero: não há de onde interpolar, e o
    importante passa a ser que a ausência tenha sempre a *mesma* forma, para que
    duas gravações sem a mão esquerda concordem nesses canais em vez de
    discordarem por ruído.
    """
    bruta = np.asarray(bruta, dtype=np.float32)
    if bruta.ndim != 3 or bruta.shape[1:] != (pose.NUM_PONTOS, 3):
        raise ValueError(f"esperado shape (T, 49, 3), recebido {bruta.shape}")

    n_frames = len(bruta)
    if n_frames == 0:
        raise ValueError("sequência vazia")

    canais = bruta.reshape(n_frames, -1).copy()
    tempo = np.arange(n_frames)

    for c in range(canais.shape[1]):
        coluna = canais[:, c]
        ok = np.isfinite(coluna)

        if ok.all():
            continue
        if not ok.any():
            canais[:, c] = 0.0
            continue

        indices = np.flatnonzero(ok)
        if len(indices) >= MINIMO_PARA_SPLINE:
            spline = CubicSpline(indices, coluna[indices], extrapolate=False)
            preenchido = spline(tempo)
            # Fora do intervalo medido a spline devolve NaN; seguramos a borda.
            preenchido[: indices[0]] = coluna[indices[0]]
            preenchido[indices[-1] + 1 :] = coluna[indices[-1]]
        else:
            preenchido = np.interp(tempo, indices, coluna[indices])

        canais[:, c] = np.where(ok, coluna, preenchido)

    return canais.reshape(bruta.shape)


def reamostrar(sequencia: np.ndarray, t_alvo: int) -> np.ndarray:
    """Estica ou encolhe no tempo para `t_alvo` frames, por interpolação linear.

    As pontas são preservadas: o primeiro e o último frame do resultado são o
    primeiro e o último da entrada. Um sinal é delimitado pelo seu começo e pelo
    seu fim, e perder qualquer um dos dois muda a palavra.
    """
    sequencia = np.asarray(sequencia, dtype=np.float32)
    n_frames = sequencia.shape[0]
    if n_frames == 0:
        raise ValueError("sequência vazia")
    if t_alvo < 1:
        raise ValueError("t_alvo deve ser >= 1")
    if n_frames == t_alvo:
        return sequencia.copy()

    canais = sequencia.reshape(n_frames, -1)
    origem = np.linspace(0.0, 1.0, n_frames)
    destino = np.linspace(0.0, 1.0, t_alvo)

    saida = np.stack(
        [np.interp(destino, origem, canais[:, c]) for c in range(canais.shape[1])],
        axis=1,
    )
    return saida.reshape(t_alvo, *sequencia.shape[1:]).astype(np.float32)


def preparar(
    bruta: np.ndarray,
    t_alvo: int = T_PADRAO,
    espelhar: bool = False,
) -> Sequencia:
    """Gravação crua → `Sequencia` comparável. É o pipeline inteiro num lugar.

    Args:
        bruta: (T, 49, 3) de `pose.montar_frame`, com NaN onde faltou medição.
        t_alvo: frames do resultado.
        espelhar: canoniza um sinalizante canhoto como destro. A máscara de
            validade sofre a mesma troca de lados — sem isso o modelo passaria a
            acreditar que mediu a mão que faltou.

    Raises:
        ValueError: se nenhum frame tem os dois ombros. Sem âncora em nenhum
            momento, a gravação não é comparável a nada e é melhor recusá-la do
            que devolver uma sequência que parece boa.
    """
    bruta = np.asarray(bruta, dtype=np.float32)
    medido = validade(bruta)

    normalizada = pose.normalizar_sequencia(imputar(bruta))

    ancorado = np.isfinite(normalizada).all(axis=(1, 2))  # (T,) tem os dois ombros
    if not ancorado.any():
        raise ValueError(
            "nenhum frame da gravação tem os dois ombros — sem âncora não há "
            "normalização possível"
        )

    normalizada = imputar(normalizada)
    medido = medido & ancorado[:, None]

    # Ponto que não foi visto em frame nenhum não tem de onde interpolar: vai
    # para a origem, que é o meio dos ombros. O zero cru cairia no canto da
    # imagem, e viraria uma mão a várias larguras de ombro do corpo.
    nunca_visto = ~medido.any(axis=0)
    normalizada[:, nunca_visto, :] = 0.0

    if espelhar:
        normalizada = pose.espelhar_sequencia(normalizada)
        medido = medido[:, pose.PERMUTACAO_ESPELHO]

    return Sequencia(
        pontos=reamostrar(normalizada, t_alvo),
        validade=np.clip(reamostrar(medido.astype(np.float32), t_alvo), 0.0, 1.0),
    )
