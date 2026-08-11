import numpy as np
import pytest

from libras.sinais import pose
from libras.sinais.segmenter import Estado, Segmentador
from tests.sinais.apoio import corpo, frame, mao

FPS = 30.0
DT = 1.0 / FPS


def segmentador(**kwargs) -> Segmentador:
    base = dict(
        limiar_movimento=0.35,
        frames_para_iniciar=3,
        segundos_repouso=0.5,
        segundos_minimo=0.4,
        segundos_maximo=4.0,
    )
    return Segmentador(**{**base, **kwargs})


def parado(largura_ombros: float = 0.2) -> np.ndarray:
    return frame(corpo=corpo(largura_ombros=largura_ombros))


def movendo(passo: int, velocidade: float = 0.05, largura_ombros: float = 0.2):
    """Mão direita subindo `velocidade` (em larguras de ombro) por frame."""
    deslocamento = np.array([0.0, -velocidade * largura_ombros * passo, 0.0])
    return frame(
        mao_direita=mao(0.6) + deslocamento,
        corpo=corpo(largura_ombros=largura_ombros),
    )


def alimentar(seg: Segmentador, frames, t0: float = 0.0):
    """Empurra frames a 30fps e devolve a primeira sequência emitida."""
    saida = None
    for i, f in enumerate(frames):
        resultado = seg.oferecer(f, t0 + i * DT)
        if resultado is not None and saida is None:
            saida = resultado
    return saida


# --- estado inicial ---


def test_comeca_em_repouso():
    seg = segmentador()
    assert seg.estado is Estado.REPOUSO
    assert seg.velocidade == 0.0


def test_maos_paradas_nao_iniciam_nada():
    seg = segmentador()
    assert alimentar(seg, [parado()] * 60) is None
    assert seg.estado is Estado.REPOUSO


def test_sem_pessoa_no_quadro_nao_inicia_nada():
    seg = segmentador()
    assert alimentar(seg, [None] * 60) is None
    assert seg.estado is Estado.REPOUSO


# --- início ---


def test_movimento_sustentado_inicia_o_sinal():
    seg = segmentador()
    alimentar(seg, [movendo(i) for i in range(10)])
    assert seg.estado is Estado.SINALIZANDO


def test_um_tranco_isolado_nao_inicia():
    """Frames_para_iniciar existe para isto: coçar o nariz não é uma consulta."""
    seg = segmentador(frames_para_iniciar=5)
    alimentar(seg, [parado(), movendo(1), movendo(1), parado(), parado()])
    assert seg.estado is Estado.REPOUSO


# --- fim ---


def test_parar_encerra_e_devolve_a_sequencia():
    seg = segmentador()
    gesto = [movendo(i) for i in range(30)]  # 1s de movimento
    repouso = [movendo(29) for _ in range(20)]  # parado no fim
    saida = alimentar(seg, gesto + repouso)

    assert saida is not None
    assert saida.ndim == 3 and saida.shape[1:] == (pose.NUM_PONTOS, 3)
    assert seg.estado is Estado.REPOUSO


def test_sinal_curto_demais_e_descartado():
    seg = segmentador(segundos_minimo=1.0)
    gesto = [movendo(i) for i in range(8)]  # ~0,27s
    assert alimentar(seg, gesto + [movendo(7)] * 20) is None
    assert seg.estado is Estado.REPOUSO


def test_duracao_maxima_corta_em_vez_de_gravar_para_sempre():
    seg = segmentador(segundos_maximo=1.0)
    saida = alimentar(seg, [movendo(i) for i in range(120)])  # 4s de movimento
    assert saida is not None
    assert len(saida) <= int(1.0 * FPS) + 2


def test_maos_sumirem_encerra_como_repouso():
    seg = segmentador()
    gesto = [movendo(i) for i in range(30)]
    sumico = [frame(mao_esquerda=None, mao_direita=None)] * 20
    assert alimentar(seg, gesto + sumico) is not None


def test_pessoa_sair_do_quadro_encerra():
    seg = segmentador()
    assert alimentar(seg, [movendo(i) for i in range(30)] + [None] * 20) is not None


# --- o conteúdo do que sai ---


def test_o_repouso_do_fim_e_aparado():
    """A mão parada esperando o app não faz parte do sinal."""
    seg = segmentador(segundos_repouso=0.3)
    gesto = [movendo(i) for i in range(30)]
    saida = alimentar(seg, gesto + [movendo(29)] * 30)
    assert len(saida) <= len(gesto) + 2


def test_a_sequencia_devolvida_contem_o_movimento_inteiro():
    """Inclusive os frames que dispararam o início, que vêm de antes da decisão."""
    seg = segmentador(frames_para_iniciar=3)
    saida = alimentar(seg, [movendo(i) for i in range(30)] + [movendo(29)] * 20)
    alturas = saida[:, pose.MAO_DIREITA, 1]
    assert alturas.max() - alturas.min() > 0.1


def test_volta_a_funcionar_depois_de_um_sinal():
    seg = segmentador()
    ciclo = [movendo(i) for i in range(30)] + [movendo(29)] * 20
    assert alimentar(seg, ciclo) is not None
    assert alimentar(seg, ciclo, t0=100.0) is not None


# --- invariância ---


def test_velocidade_e_indiferente_a_distancia_da_camera():
    """Medida em larguras de ombro por segundo — mesma invariante da normalização.

    Sem isso, chegar perto da câmera dispararia sinais sozinho e ficar longe
    deixaria o app surdo.
    """
    velocidades = []
    for largura in (0.1, 0.4):
        seg = segmentador()
        for i in range(6):
            seg.oferecer(movendo(i, largura_ombros=largura), i * DT)
        velocidades.append(seg.velocidade)

    assert velocidades[0] == pytest.approx(velocidades[1], rel=1e-3)


def test_frame_sem_ombros_nao_conta_como_movimento():
    """Sem âncora não dá para medir velocidade; inventar um número dispararia."""
    seg = segmentador()
    sem_corpo = frame(corpo=None)
    alimentar(seg, [parado(), sem_corpo, parado(), sem_corpo] * 15)
    assert seg.estado is Estado.REPOUSO


# --- controle ---


def test_reiniciar_esquece_o_sinal_em_andamento():
    seg = segmentador()
    alimentar(seg, [movendo(i) for i in range(10)])
    seg.reiniciar()
    assert seg.estado is Estado.REPOUSO
    assert seg.duracao == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limiar_movimento": 0.0},
        {"frames_para_iniciar": 0},
        {"segundos_maximo": 0.2},  # menor que segundos_minimo
    ],
)
def test_rejeita_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        segmentador(**kwargs)
