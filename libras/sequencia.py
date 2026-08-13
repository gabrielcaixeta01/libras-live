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
from scipy.interpolate import PchipInterpolator

from . import pose

T_PADRAO = 32  # frames por sinal depois da reamostragem; ~1s a 30fps
MINIMO_PARA_SPLINE = 4  # abaixo disso a cúbica não tem apoio e vira linear

# Em larguras de ombro: um braço aberto chega a ~1,5 e a mão mais distante que o
# corpo alcança fica perto de 2. Valor acima disto não é gesto, é imputação ruim
# — e o dicionário tem amostras chegando a ±400. Cortar aí não perde gesto
# nenhum e evita que um ponto inventado domine a distância ou a rede.
LIMITE_PLAUSIVEL = 5.0


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
        """(T, 147) — só a geometria."""
        return pose.vetores(self.pontos)

    @property
    def vetores_com_validade(self) -> np.ndarray:
        """(T, 196) — a geometria com a máscara grudada atrás dela.

        É o formato **guardado em disco** e o que circula entre o extrator, o
        dicionário e o encoder. A máscara viaja junto porque ela não é
        recuperável depois: uma vez imputado, um ponto inventado é
        indistinguível de um medido, e reconstruí-la custaria reprocessar o
        vídeo.

        Quem só quer a geometria chama `separar_validade` e ignora a segunda
        metade — é o que a busca DTW faz, para continuar comparável à baseline
        que foi medida antes desta mudança.
        """
        return np.concatenate(
            [self.vetores, self.validade.astype(np.float32)], axis=-1, dtype=np.float32
        )

    def __len__(self) -> int:
        return len(self.pontos)


def validade(bruta: np.ndarray) -> np.ndarray:
    """(T, 49, 3) → (T, 49) booleano. Um ponto vale se as três coordenadas valem."""
    return np.isfinite(np.asarray(bruta, dtype=np.float32)).all(axis=2)


# Largura da máscara e do vetor que a carrega. A máscara é guardada **ponto a
# ponto**, e não já resumida: resumir é barato e reversível, re-extrair 4.086
# vídeos para recuperar o que foi jogado fora não é. Quem consome decide o
# resumo — hoje `validade_agrupada`, amanhã outro, sem tocar no disco.
TAMANHO_VALIDADE = pose.NUM_PONTOS                              # 49
TAMANHO_COM_VALIDADE = pose.TAMANHO_VETOR + TAMANHO_VALIDADE    # 196

# Os três grupos que a máscara realmente distingue. O MediaPipe entrega uma mão
# inteira ou nenhuma, então os 21 pontos de cada mão sobem e descem juntos; o
# corpo cai ponto a ponto, mas quase nunca cai. Três canais carregam o que 49
# diriam, e numa entrada de 414 canais com 2 gravações por classe a largura que
# não informa é largura que atrapalha.
NUM_GRUPOS_VALIDADE = 3


def separar_validade(vetores: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(..., T, 147 ou 196) → (geometria, máscara ou None).

    Aceita os dois formatos de propósito: o dicionário em disco mudou de largura
    quando a máscara passou a ser guardada, e um `.npz` antigo tem que continuar
    carregando em vez de estourar. Sem máscara, o segundo elemento é None e quem
    chamou decide se isso é um problema.
    """
    v = np.asarray(vetores, dtype=np.float32)
    largura = v.shape[-1]

    if largura == pose.TAMANHO_VETOR:
        return v, None
    if largura == TAMANHO_COM_VALIDADE:
        return v[..., : pose.TAMANHO_VETOR], v[..., pose.TAMANHO_VETOR :]

    raise ValueError(
        f"esperado largura {pose.TAMANHO_VETOR} ou {TAMANHO_COM_VALIDADE}, "
        f"recebido {largura}"
    )


def validade_agrupada(mascara: np.ndarray) -> np.ndarray:
    """(..., T, 49) → (..., T, 3): fração medida da mão esquerda, da direita e do corpo.

    Média e não mínimo: depois da reamostragem a máscara já é fracionária, e um
    mínimo jogaria o grupo inteiro para zero por causa de um ponto. A média diz
    *quanto* daquele grupo foi medido, que é a pergunta que o encoder tem
    condição de usar.
    """
    m = np.asarray(mascara, dtype=np.float32)
    if m.shape[-1] != TAMANHO_VALIDADE:
        raise ValueError(f"esperado (..., T, {TAMANHO_VALIDADE}), recebido {m.shape}")

    grupos = [
        m[..., pose.MAO_ESQUERDA].mean(axis=-1),
        m[..., pose.MAO_DIREITA].mean(axis=-1),
        m[..., pose.POSE].mean(axis=-1),
    ]
    return np.stack(grupos, axis=-1).astype(np.float32)


def imputar(bruta: np.ndarray) -> np.ndarray:
    """Preenche os NaN interpolando cada coordenada ao longo do tempo.

    Cúbica de Hermite (PCHIP) quando há apoio suficiente, linear quando não há,
    e o valor da borda quando o buraco está nas pontas — **nunca extrapolação**.

    **PCHIP e não `CubicSpline`, e a diferença não é estética.** A cúbica natural
    é global: ela casa a segunda derivada entre os trechos, e para conseguir isso
    através de um buraco longo ela dispara para fora do intervalo medido. O
    dicionário extraído com ela tinha pontos a **218 larguras de ombro** do
    corpo, quando um braço aberto chega a 1,5 — a mão sumia atrás do tronco por
    meio segundo e a spline preenchia o buraco com uma parábola gigante. PCHIP é
    a mesma família de cúbicas, mas preserva a forma: dentro de um buraco ela
    fica entre os dois valores que o delimitam, por construção. Não há
    ultrapassagem possível.

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
            spline = PchipInterpolator(indices, coluna[indices], extrapolate=False)
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


def velocidade(vetores: np.ndarray) -> np.ndarray:
    """(..., T, D) → (..., T, D): quanto cada coordenada andou desde o frame anterior.

    O primeiro frame vem zerado, e não descartado: o resultado tem que continuar
    alinhado com a posição frame a frame para que os dois possam ser
    concatenados como canais da mesma sequência.

    Duas coisas diferentes em Libras podem ter a mesma posição média e
    trajetórias opostas — a posição diz *onde*, a velocidade diz *para onde*.
    """
    v = np.asarray(vetores, dtype=np.float32)
    if v.ndim < 2:
        raise ValueError(f"esperado (T, D) ou (N, T, D), recebido {v.shape}")
    return np.diff(v, axis=-2, prepend=v[..., :1, :])


def normalizar_z(vetores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """z-score de cada canal ao longo do tempo, por gravação.

    Tira de cada coordenada a sua própria média e o seu próprio desvio dentro
    daquela gravação, o que absorve diferenças de proporção corporal e de
    enquadramento entre pessoas. Medido sobre o DTW, é o único ganho barato que
    apareceu: recall@5 de 16,4% para 19,5% num vocabulário de 250.

    **O preço é alto e por isso isto não é o padrão.** Centrar cada canal apaga
    a *localização absoluta* do gesto, e em Libras localização é fonema — PAI e
    MÃE têm a mesma configuração de mão em lugares diferentes do rosto. Para o
    DTW, que não tem como aprender a compensar articulador, a troca valeu. Para
    o encoder, que é treinado exatamente para isso, jogar a informação fora é
    pagar duas vezes — e está medido: **10,3% de recall@5 sem, 7,1% com**, no
    mesmo leave-one-articulator-out. Com ela o encoder perde até para o DTW.

    Canal constante — a mão que não apareceu em nenhum frame — sai zerado, e não
    dividido por quase-zero.
    """
    v = np.asarray(vetores, dtype=np.float32)
    if v.ndim < 2:
        raise ValueError(f"esperado (T, D) ou (N, T, D), recebido {v.shape}")

    media = v.mean(axis=-2, keepdims=True)
    desvio = v.std(axis=-2, keepdims=True)
    return np.where(desvio > eps, (v - media) / np.maximum(desvio, eps), 0.0).astype(
        np.float32
    )


# A escala de cada mão: do pulso à base do dedo médio. É a medida que menos
# depende de a mão estar aberta ou fechada — os dedos dobram, a palma não.
_MCP_MEDIO = 9
ESCALA_MAO_MINIMA = 1e-3  # abaixo disso não há mão, há um ponto imputado a zero

# Pontos por mão depois de tirar o pulso, que vira a origem e seria zero fixo.
NUM_PONTOS_MAO_LOCAL = pose.NUM_PONTOS_MAO - 1
TAMANHO_MAOS_LOCAIS = 2 * NUM_PONTOS_MAO_LOCAL * 3  # 120


def maos_locais(vetores: np.ndarray) -> np.ndarray:
    """(..., T, 147) → (..., T, 120): a **configuração de mão**, em escala própria.

    Este é o canal que a âncora no corpo apaga. Normalizado pela largura dos
    ombros, o punho percorre mais de uma unidade ao longo de um sinal enquanto um
    dedo se move 0,03 — medido no dicionário, a variância dos canais da mão é
    duas ordens de grandeza menor que a da trajetória. A rede vê a trajetória e
    quase não vê a mão, e configuração de mão é fonema em Libras: SABER e
    COMPREENDER acontecem no mesmo lugar da testa e diferem pelos dedos.

    Aqui cada mão volta a ser o que `alfabeto.landmarks.normalizar` faz para uma
    letra: pulso na origem, escala própria. É a mesma informação da fase 1
    entrando como canal extra, e de graça — sai da geometria já extraída, sem
    reprocessar vídeo nenhum.

    Mão que não apareceu em frame nenhum chega aqui zerada e sai zerada: não há
    escala para dividir, e inventar uma faria ruído virar dedo.
    """
    v = np.asarray(vetores, dtype=np.float32)
    if v.shape[-1] != pose.TAMANHO_VETOR:
        raise ValueError(f"esperado (..., T, 147), recebido {v.shape}")

    pontos = v.reshape(*v.shape[:-1], pose.NUM_PONTOS, 3)
    saidas = []

    for fatia in (pose.MAO_ESQUERDA, pose.MAO_DIREITA):
        mao = pontos[..., fatia, :]
        local = mao[..., 1:, :] - mao[..., :1, :]

        escala = np.linalg.norm(
            mao[..., _MCP_MEDIO, :2] - mao[..., 0, :2], axis=-1
        )[..., None, None]
        local = np.where(escala > ESCALA_MAO_MINIMA, local / np.maximum(escala, ESCALA_MAO_MINIMA), 0.0)

        saidas.append(local.reshape(*v.shape[:-1], NUM_PONTOS_MAO_LOCAL * 3))

    return np.concatenate(saidas, axis=-1, dtype=np.float32)


def caracteristicas(
    vetores: np.ndarray,
    com_velocidade: bool = True,
    z: bool = False,
    com_maos: bool = False,
    com_validade: bool = False,
    limite: float = LIMITE_PLAUSIVEL,
) -> np.ndarray:
    """(..., T, 147 ou 196) → (..., T, D): o que entra no encoder.

    Sempre limita a amplitude primeiro, porque tudo o que vem depois — a
    velocidade, o z-score, a rede — é sensível a um ponto inventado longe do
    corpo.

    A entrada pode vir com ou sem a máscara grudada atrás (ver
    `Sequencia.vetores_com_validade`). Vindo com, ela é separada antes de
    qualquer conta: a máscara não é geometria e não pode entrar na velocidade,
    no z-score nem no cálculo de escala das mãos.

    Args:
        vetores: sequência já normalizada no corpo, com ou sem máscara.
        com_velocidade: concatena a derivada temporal como canais extras.
        z: aplica `normalizar_z` na posição. Ver lá o que isso custa.
        com_maos: concatena `maos_locais` — a configuração de mão em escala
            própria, que a normalização no corpo deixa pequena demais para ser
            vista. A **derivada** desses canais foi medida e não entra: 41,0%
            de recall@5 contra 44,8% sem ela. Duplicar a largura da entrada com
            2 gravações por classe custa mais do que o dedo em movimento paga.
        com_validade: concatena `validade_agrupada` — três canais dizendo quanto
            de cada mão e do corpo foi de fato medido naquele frame. Sem eles um
            ponto imputado entra na rede como se a câmera o tivesse visto.
        limite: amplitude máxima aceita, em larguras de ombro.

    Raises:
        ValueError: se `com_validade` e a entrada não trouxer máscara. Seguir em
            frente inventaria três canais constantes e a rede aprenderia que
            está tudo sempre medido — exatamente a mentira que o parâmetro
            existe para desfazer.
    """
    geometria, mascara = separar_validade(vetores)

    if com_validade and mascara is None:
        raise ValueError(
            "com_validade=True mas a entrada tem largura "
            f"{pose.TAMANHO_VETOR}, sem máscara. Use `Sequencia.vetores_com_validade` "
            "ou reextraia o dicionário com `training/prepare_sinais.py`."
        )

    v = np.clip(geometria, -limite, limite)

    # Antes do z-score, que apagaria a escala de cada mão junto com a do corpo.
    locais = maos_locais(v) if com_maos else None

    if z:
        v = normalizar_z(v)

    canais = [v]
    if com_velocidade:
        canais.append(np.clip(velocidade(v), -limite, limite))
    if locais is not None:
        canais.append(np.clip(locais, -limite, limite))
    if com_validade:
        canais.append(validade_agrupada(mascara))

    if len(canais) == 1:
        return canais[0]
    return np.concatenate(canais, axis=-1, dtype=np.float32)


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
