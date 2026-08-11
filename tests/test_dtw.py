import numpy as np
import pytest

from libras import dtw


def rampa(n: int = 32, direcao=(1.0, 0.0)) -> np.ndarray:
    """Uma trajetória reta de n frames em 2 dimensões."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (t * np.array(direcao, dtype=np.float32)).astype(np.float32)


def acelerada(n: int = 32) -> np.ndarray:
    """A mesma trajetória da rampa, feita devagar no começo e rápido no fim."""
    t = np.linspace(0.0, 1.0, n)[:, None] ** 2
    return (t * np.array([1.0, 0.0], dtype=np.float32)).astype(np.float32)


# --- propriedades básicas ---


def test_distancia_de_uma_sequencia_a_si_mesma_e_zero():
    a = rampa()
    assert dtw.distancia(a, a) == pytest.approx(0.0, abs=1e-6)


def test_distancia_nunca_e_negativa():
    assert dtw.distancia(rampa(), rampa(direcao=(0.0, 1.0))) >= 0.0


def test_e_simetrica():
    a, b = rampa(), acelerada()
    assert dtw.distancia(a, b) == pytest.approx(dtw.distancia(b, a), rel=1e-5)


def test_trajetorias_diferentes_ficam_longe():
    igual = dtw.distancia(rampa(), rampa())
    diferente = dtw.distancia(rampa(), rampa(direcao=(0.0, 1.0)))
    assert diferente > igual


# --- a razão de existir ---


def test_absorve_diferenca_de_ritmo():
    """É por isto que a métrica é DTW e não distância ponto a ponto.

    O mesmo sinal feito devagar e feito rápido é a mesma palavra. Uma distância
    euclidiana frame a frame chamaria os dois de coisas diferentes.
    """
    mesmo_caminho_outro_ritmo = dtw.distancia(rampa(), acelerada())
    outro_caminho = dtw.distancia(rampa(), rampa(direcao=(0.0, 1.0)))
    assert mesmo_caminho_outro_ritmo < outro_caminho / 3


def test_sequencias_de_tamanhos_diferentes_se_comparam():
    """Não dá zero exato — 20 e 50 pontos na mesma reta caem em grades
    diferentes, e o melhor emparelhamento ainda junta pontos que não coincidem.
    O que precisa valer é que a sobra seja pequena perto de trajetória errada."""
    mesma_reta = dtw.distancia(rampa(20), rampa(50))
    outra_reta = dtw.distancia(rampa(20), rampa(50, direcao=(0.0, 1.0)))
    assert mesma_reta < outra_reta / 10


# --- banda de Sakoe-Chiba ---


def test_banda_estreita_restringe_o_alinhamento():
    a, b = rampa(), acelerada()
    assert dtw.distancia(a, b, banda=0.05) >= dtw.distancia(a, b, banda=1.0)


def test_banda_cheia_permite_qualquer_alinhamento():
    a, b = rampa(20), rampa(50)
    assert np.isfinite(dtw.distancia(a, b, banda=1.0))


def test_banda_estreita_demais_ainda_devolve_numero_finito():
    """A diagonal sempre cabe na banda, por menor que ela seja."""
    assert np.isfinite(dtw.distancia(rampa(32), rampa(32), banda=0.0))


# --- lote ---


def test_lote_concorda_com_o_calculo_individual():
    consulta = acelerada()
    base = np.stack([rampa(), rampa(direcao=(0.0, 1.0)), acelerada()])

    lote = dtw.distancias(consulta, base)
    individual = [dtw.distancia(consulta, b) for b in base]

    assert np.allclose(lote, individual, rtol=1e-4)


def test_lote_de_um_candidato_so():
    d = dtw.distancias(rampa(), rampa()[None])
    assert d.shape == (1,)
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def test_lote_encontra_o_mais_proximo():
    base = np.stack([rampa(direcao=(0.0, 1.0)), rampa(), rampa(direcao=(1.0, 1.0))])
    assert int(np.argmin(dtw.distancias(acelerada(), base))) == 1


# --- validação ---


@pytest.mark.parametrize(
    "a, b",
    [
        (np.zeros((0, 2)), np.zeros((5, 2))),
        (np.zeros((5, 2)), np.zeros((0, 2))),
        (np.zeros((5, 2)), np.zeros((5, 3))),
        (np.zeros(5), np.zeros((5, 2))),
    ],
)
def test_rejeita_entradas_invalidas(a, b):
    with pytest.raises(ValueError):
        dtw.distancia(a, b)


def test_lote_exige_tres_dimensoes():
    with pytest.raises(ValueError):
        dtw.distancias(rampa(), rampa())
