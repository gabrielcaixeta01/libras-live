"""O pipeline de sinais inteiro, sem câmera, sem vídeo e sem dataset.

Os testes de unidade garantem que cada peça está certa. Este garante que elas
se encaixam — que o formato que o segmentador emite é o que `preparar` aceita, e
que o que `preparar` devolve é o que o dicionário consulta. É onde um contrato
quebrado entre módulos aparece.
"""

from __future__ import annotations

import numpy as np

from libras import pose, sequencia
from libras.dicionario import Dicionario
from libras.segmenter import Estado, Segmentador
from tests.apoio import corpo, frame, mao

FPS = 30.0
DT = 1.0 / FPS


def gesto(altura_final: float, n: int = 30) -> list[np.ndarray]:
    """Mão direita subindo até `altura_final` — um 'sinal' sintético."""
    return [
        frame(
            mao_direita=mao(0.6) + np.array([0.0, -altura_final * i / (n - 1), 0.0]),
            corpo=corpo(),
        )
        for i in range(n)
    ]


def repouso(ultimo: np.ndarray, n: int = 20) -> list[np.ndarray]:
    return [ultimo] * n


def segmentador() -> Segmentador:
    return Segmentador(
        limiar_movimento=0.35,
        frames_para_iniciar=3,
        segundos_repouso=0.5,
        segundos_minimo=0.4,
        segundos_maximo=4.0,
    )


def capturar(frames: list[np.ndarray], t0: float = 0.0) -> np.ndarray:
    """Roda os frames pelo segmentador e devolve a gravação que ele emitir."""
    seg = segmentador()
    for i, f in enumerate(frames):
        gravacao = seg.oferecer(f, t0 + i * DT)
        if gravacao is not None:
            return gravacao
    raise AssertionError("o segmentador não fechou nenhum sinal")


def prototipo(amplitude: float) -> np.ndarray:
    frames = gesto(amplitude)
    return sequencia.preparar(capturar(frames + repouso(frames[-1]))).vetores


def test_da_camera_ao_candidato():
    """Câmera → segmentação → sequência → busca, sem tocar em disco.

    As amplitudes são todas rápidas o bastante para acordar o segmentador: um
    gesto de 0,05 se move a 0,26 larguras de ombro por segundo e fica abaixo do
    limiar — de propósito, e é o que impede o app de disparar sozinho.
    """
    frames = gesto(0.16)
    consulta = sequencia.preparar(capturar(frames + repouso(frames[-1])))
    assert consulta.vetores.shape == (sequencia.T_PADRAO, pose.TAMANHO_VETOR)

    dicionario = Dicionario(
        representacoes=np.stack([prototipo(a) for a in (0.15, 0.30, 0.50)]),
        rotulos=["POUCO", "MEIO", "MUITO"],
        fontes=["art1", "art1", "art1"],
    )

    candidatos = dicionario.buscar(consulta.vetores, k=3)
    assert candidatos[0].rotulo == "POUCO"
    assert len(candidatos) == 3


def test_dois_sinais_seguidos_na_mesma_sessao():
    """O segmentador tem que voltar ao repouso limpo entre uma consulta e outra."""
    seg = segmentador()
    frames = gesto(0.15)
    fluxo = frames + repouso(frames[-1]) + frames + repouso(frames[-1])

    gravacoes = [
        g for i, f in enumerate(fluxo) if (g := seg.oferecer(f, i * DT)) is not None
    ]
    assert len(gravacoes) == 2
    assert seg.estado is Estado.REPOUSO


def test_sinal_de_uma_mao_so_atravessa_o_pipeline():
    """A mão que falta não pode virar NaN em lugar nenhum do caminho."""
    frames = [
        frame(
            mao_esquerda=None,
            mao_direita=mao(0.6) + np.array([0.0, -0.15 * i / 29, 0.0]),
            corpo=corpo(),
        )
        for i in range(30)
    ]
    consulta = sequencia.preparar(capturar(frames + repouso(frames[-1])))

    assert np.isfinite(consulta.vetores).all()
    assert consulta.validade[:, pose.MAO_ESQUERDA].max() == 0.0


def test_ensinar_um_sinal_muda_a_resposta_seguinte():
    """A re-ancoragem, ponta a ponta: o caminho da correção na tela."""
    frames = gesto(0.15)
    consulta = sequencia.preparar(capturar(frames + repouso(frames[-1])))

    outro = sequencia.preparar(capturar(gesto(0.4) + repouso(gesto(0.4)[-1])))
    dicionario = Dicionario(
        representacoes=outro.vetores[None], rotulos=["OUTRO"], fontes=["art1"]
    )

    assert dicionario.buscar(consulta.vetores, k=1)[0].rotulo == "OUTRO"

    dicionario.ancorar(consulta.vetores, "CERTO")
    melhor = dicionario.buscar(consulta.vetores, k=1)[0]
    assert melhor.rotulo == "CERTO"
    assert melhor.fonte == "voce"


def test_prototipos_do_usuario_sobrevivem_ao_disco(tmp_path):
    frames = gesto(0.15)
    consulta = sequencia.preparar(capturar(frames + repouso(frames[-1])))

    dicionario = Dicionario(
        representacoes=consulta.vetores[None], rotulos=["CASA"], fontes=["art1"]
    )
    dicionario.ancorar(consulta.vetores, "MINHA CASA")

    caminho = tmp_path / "meus.npz"
    dicionario.apenas("voce").salvar(caminho)

    recarregado = Dicionario.carregar(caminho)
    assert recarregado.vocabulario == ["MINHA CASA"]
    assert recarregado.buscar(consulta.vetores, k=1)[0].fonte == "voce"
