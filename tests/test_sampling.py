import numpy as np
import pytest

from libras.sampling import ColetorDiverso


def vetor(valor: float) -> np.ndarray:
    """Vetor de 63 floats constantes — a distância entre dois é previsível."""
    return np.full(63, valor, dtype=np.float32)


def coletor(total: int = 5, distancia: float = 0.1, **kwargs) -> ColetorDiverso:
    return ColetorDiverso(total=total, distancia_minima=distancia, **kwargs)


# --- construção ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total": 0},
        {"total": -1},
        {"distancia_minima": 0.0},
        {"distancia_minima": -0.5},
    ],
)
def test_rejeita_parametros_invalidos(kwargs):
    base = {"total": 5, "distancia_minima": 0.1}
    with pytest.raises(ValueError):
        ColetorDiverso(**{**base, **kwargs})


def test_comeca_vazio():
    c = coletor()
    assert len(c.amostras) == 0
    assert not c.completo
    assert c.dispersao == 0.0


# --- critério de diversidade ---


def test_primeira_amostra_e_sempre_aceita():
    assert coletor().oferecer(vetor(0.0), 0.0) is True


def test_amostra_identica_e_rejeitada():
    c = coletor()
    c.oferecer(vetor(0.0), 0.0)
    assert c.oferecer(vetor(0.0), 0.1) is False
    assert len(c.amostras) == 1


def test_amostra_quase_identica_e_rejeitada():
    """O caso real: 30 frames por segundo da mão parada."""
    c = coletor(distancia=0.5)
    c.oferecer(vetor(0.0), 0.0)
    # 63 coordenadas deslocadas de 0.001 => distância ~0.008
    assert c.oferecer(vetor(0.001), 0.1) is False


def test_amostra_distante_e_aceita():
    c = coletor(distancia=0.1)
    c.oferecer(vetor(0.0), 0.0)
    assert c.oferecer(vetor(1.0), 0.1) is True
    assert len(c.amostras) == 2


def test_compara_com_todas_as_aceitas_nao_so_a_ultima():
    """Voltar a uma pose já gravada não pode contar de novo."""
    c = coletor(distancia=0.5)
    c.oferecer(vetor(0.0), 0.0)
    c.oferecer(vetor(1.0), 0.1)
    assert c.oferecer(vetor(0.0), 0.2) is False


# --- afrouxamento ---


def test_limiar_afrouxa_quando_nada_e_aceito():
    c = coletor(distancia=1.0, paciencia=1.0, decaimento=0.5)
    c.oferecer(vetor(0.0), 0.0)
    inicial = c.limiar

    c.oferecer(vetor(0.0), 2.0)  # muito tempo sem aceitar
    assert c.limiar < inicial


def test_limiar_nao_afrouxa_antes_da_paciencia():
    c = coletor(distancia=1.0, paciencia=5.0, decaimento=0.5)
    c.oferecer(vetor(0.0), 0.0)
    c.oferecer(vetor(0.0), 1.0)
    assert c.limiar == pytest.approx(1.0)


def test_limiar_tem_piso():
    c = coletor(distancia=1.0, paciencia=0.1, decaimento=0.5, limiar_minimo=0.3)
    c.oferecer(vetor(0.0), 0.0)
    for passo in range(1, 40):
        c.oferecer(vetor(0.0), passo * 1.0)
    assert c.limiar == pytest.approx(0.3)


def test_afrouxar_acaba_destravando_a_coleta():
    """Garante que a coleta sempre termina, mesmo com a mão parada."""
    c = coletor(total=3, distancia=10.0, paciencia=0.1, decaimento=0.5, limiar_minimo=1e-6)
    instante = 0.0
    while not c.completo and instante < 500:
        instante += 0.5
        c.oferecer(vetor(0.001 * instante), instante)
    assert c.completo


def test_aceitar_zera_o_relogio_do_afrouxamento():
    c = coletor(distancia=0.5, paciencia=1.0, decaimento=0.5)
    c.oferecer(vetor(0.0), 0.0)
    c.oferecer(vetor(5.0), 0.5)  # aceita, relogio zera
    c.oferecer(vetor(5.0), 1.2)  # so 0.7s desde a ultima aceita
    assert c.limiar == pytest.approx(0.5)


# --- estado e feedback ---


def test_parado_indica_travamento():
    c = coletor(distancia=1.0, paciencia=1.0)
    c.oferecer(vetor(0.0), 0.0)
    assert not c.parado(0.5)
    assert c.parado(3.0)


def test_completo_depois_do_total():
    c = coletor(total=3, distancia=0.1)
    for i in range(3):
        c.oferecer(vetor(float(i)), i * 0.1)
    assert c.completo
    assert len(c.amostras) == 3


def test_nao_aceita_depois_de_completo():
    c = coletor(total=2, distancia=0.1)
    c.oferecer(vetor(0.0), 0.0)
    c.oferecer(vetor(1.0), 0.1)
    assert c.oferecer(vetor(5.0), 0.2) is False
    assert len(c.amostras) == 2


def test_amostras_saem_no_formato_do_treino():
    c = coletor(total=2, distancia=0.1)
    c.oferecer(vetor(0.0), 0.0)
    c.oferecer(vetor(1.0), 0.1)

    amostras = c.amostras
    assert amostras.shape == (2, 63)
    assert amostras.dtype == np.float32


def test_dispersao_cresce_com_amostras_variadas():
    c = coletor(total=10, distancia=0.1)
    c.oferecer(vetor(0.0), 0.0)
    apertada = c.dispersao
    c.oferecer(vetor(2.0), 0.1)
    assert c.dispersao > apertada


def test_progresso_vai_de_zero_a_um():
    c = coletor(total=2, distancia=0.1)
    assert c.progresso == 0.0
    c.oferecer(vetor(0.0), 0.0)
    assert c.progresso == pytest.approx(0.5)
    c.oferecer(vetor(1.0), 0.1)
    assert c.progresso == pytest.approx(1.0)


def test_rejeita_vetor_de_tamanho_errado():
    c = coletor()
    with pytest.raises(ValueError):
        c.oferecer(np.zeros(10, dtype=np.float32), 0.0)
