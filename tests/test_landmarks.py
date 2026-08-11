import numpy as np
import pytest

from libras.landmarks import NUM_PONTOS, TAMANHO_VETOR, normalizar


def mao_falsa(semente: int = 0) -> np.ndarray:
    """21 pontos plausíveis: pulso na origem, dedos espalhados."""
    rng = np.random.default_rng(semente)
    pontos = rng.uniform(-0.2, 0.2, size=(NUM_PONTOS, 3)).astype(np.float32)
    pontos[0] = [0.5, 0.5, 0.0]  # pulso longe da origem, de propósito
    return pontos


def test_retorna_vetor_de_63_floats():
    assert normalizar(mao_falsa()).shape == (TAMANHO_VETOR,)


def test_rejeita_shape_errado():
    with pytest.raises(ValueError):
        normalizar(np.zeros((10, 3), dtype=np.float32))


def test_pulso_vai_para_a_origem():
    vetor = normalizar(mao_falsa()).reshape(NUM_PONTOS, 3)
    assert np.allclose(vetor[0], [0, 0, 0], atol=1e-6)


def test_invariante_a_translacao():
    """A mesma mão em outro canto da tela gera o mesmo vetor."""
    pontos = mao_falsa()
    deslocada = pontos + np.array([0.3, -0.15, 0.05], dtype=np.float32)

    assert np.allclose(normalizar(pontos), normalizar(deslocada), atol=1e-5)


def test_invariante_a_escala():
    """A mesma mão mais perto da câmera gera o mesmo vetor."""
    pontos = mao_falsa()
    ampliada = pontos * 2.5

    assert np.allclose(normalizar(pontos), normalizar(ampliada), atol=1e-5)


def test_maior_distancia_ao_pulso_vira_um():
    vetor = normalizar(mao_falsa()).reshape(NUM_PONTOS, 3)
    assert np.isclose(np.max(np.linalg.norm(vetor, axis=1)), 1.0, atol=1e-5)


def test_mao_esquerda_espelha_o_eixo_x():
    pontos = mao_falsa()
    direita = normalizar(pontos, mao_esquerda=False).reshape(NUM_PONTOS, 3)
    esquerda = normalizar(pontos, mao_esquerda=True).reshape(NUM_PONTOS, 3)

    assert np.allclose(direita[:, 0], -esquerda[:, 0], atol=1e-5)
    assert np.allclose(direita[:, 1:], esquerda[:, 1:], atol=1e-5)


def test_mao_degenerada_nao_divide_por_zero():
    """Todos os pontos no mesmo lugar não deve gerar NaN."""
    vetor = normalizar(np.zeros((NUM_PONTOS, 3), dtype=np.float32))
    assert not np.isnan(vetor).any()


def test_nao_modifica_a_entrada():
    pontos = mao_falsa()
    copia = pontos.copy()
    normalizar(pontos)
    assert np.array_equal(pontos, copia)
