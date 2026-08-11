import numpy as np
import pytest

from libras.augment import (
    equilibrar,
    escalar_profundidade,
    perturbar,
    rotacionar,
    variar,
)
from libras.landmarks import NUM_PONTOS, TAMANHO_VETOR, normalizar


def mao_falsa(semente: int = 0) -> np.ndarray:
    """Um vetor normalizado plausível, como o que sai de `normalizar`."""
    rng = np.random.default_rng(semente)
    pontos = rng.uniform(-0.2, 0.2, size=(NUM_PONTOS, 3)).astype(np.float32)
    pontos[0] = [0.5, 0.5, 0.0]
    return normalizar(pontos)


def esta_normalizado(vetor: np.ndarray) -> bool:
    """Invariantes que `normalizar` garante: pulso na origem, raio máximo 1."""
    pontos = vetor.reshape(NUM_PONTOS, 3)
    return bool(
        np.allclose(pontos[0], 0, atol=1e-5)
        and np.isclose(np.max(np.linalg.norm(pontos, axis=1)), 1.0, atol=1e-5)
    )


# --- rotacionar ---


def test_rotacao_nula_e_identidade():
    vetor = mao_falsa()
    assert np.allclose(rotacionar(vetor, 0, 0, 0), vetor, atol=1e-5)


def test_rotacao_preserva_as_invariantes():
    girado = rotacionar(mao_falsa(), 10, -8, 15)
    assert esta_normalizado(girado)


def test_rotacao_e_reversivel():
    vetor = mao_falsa()
    ida = rotacionar(vetor, 0, 0, 30)
    volta = rotacionar(ida, 0, 0, -30)
    assert np.allclose(volta, vetor, atol=1e-4)


def test_rotacao_muda_o_vetor():
    vetor = mao_falsa()
    assert not np.allclose(rotacionar(vetor, 0, 0, 25), vetor, atol=1e-3)


def test_rotacao_em_z_nao_mexe_na_profundidade():
    vetor = mao_falsa()
    girado = rotacionar(vetor, 0, 0, 40)
    z_antes = vetor.reshape(NUM_PONTOS, 3)[:, 2]
    z_depois = girado.reshape(NUM_PONTOS, 3)[:, 2]
    # A renormalização é identidade aqui: rotação preserva as normas.
    assert np.allclose(z_antes, z_depois, atol=1e-5)


# --- perturbar ---


def test_ruido_zero_e_identidade():
    vetor = mao_falsa()
    rng = np.random.default_rng(1)
    assert np.allclose(perturbar(vetor, 0.0, rng), vetor, atol=1e-5)


def test_ruido_perturba_mas_mantem_as_invariantes():
    rng = np.random.default_rng(1)
    ruidoso = perturbar(mao_falsa(), 0.02, rng)
    assert not np.allclose(ruidoso, mao_falsa(), atol=1e-4)
    assert esta_normalizado(ruidoso)


# --- escalar_profundidade ---


def test_profundidade_fator_um_e_identidade():
    vetor = mao_falsa()
    assert np.allclose(escalar_profundidade(vetor, 1.0), vetor, atol=1e-5)


def test_profundidade_so_mexe_em_z():
    vetor = mao_falsa()
    achatado = escalar_profundidade(vetor, 0.5)
    antes = vetor.reshape(NUM_PONTOS, 3)
    depois = achatado.reshape(NUM_PONTOS, 3)

    # x e y podem ser reescalados pela renormalização, mas a proporção entre
    # eles não muda; z encolhe em relação a x.
    assert esta_normalizado(achatado)
    razao_z = np.abs(depois[:, 2]).sum() / max(np.abs(depois[:, 0]).sum(), 1e-8)
    razao_z_antes = np.abs(antes[:, 2]).sum() / max(np.abs(antes[:, 0]).sum(), 1e-8)
    assert razao_z < razao_z_antes


# --- variar ---


def test_variar_e_deterministico_por_semente():
    vetor = mao_falsa()
    a = variar(vetor, np.random.default_rng(7))
    b = variar(vetor, np.random.default_rng(7))
    assert np.allclose(a, b)


def test_variar_devolve_vetor_valido():
    vetor = variar(mao_falsa(), np.random.default_rng(3))
    assert vetor.shape == (TAMANHO_VETOR,)
    assert vetor.dtype == np.float32
    assert esta_normalizado(vetor)


# --- equilibrar ---


def conjunto_desbalanceado() -> tuple[np.ndarray, np.ndarray]:
    """Imita o problema real: uma classe farta e uma classe faminta."""
    X = np.array([mao_falsa(i) for i in range(12)], dtype=np.float32)
    y = np.array(["B"] * 10 + ["N"] * 2)
    return X, y


def test_equilibrar_leva_todas_as_classes_ao_alvo():
    X, y = conjunto_desbalanceado()
    _, y_eq = equilibrar(X, y, alvo=10, semente=0)

    contagem = {letra: int((y_eq == letra).sum()) for letra in set(y_eq)}
    assert contagem == {"B": 10, "N": 10}


def test_equilibrar_preserva_as_amostras_originais():
    X, y = conjunto_desbalanceado()
    X_eq, y_eq = equilibrar(X, y, alvo=10, semente=0)

    for vetor in X:
        assert any(np.allclose(vetor, candidato, atol=1e-6) for candidato in X_eq)
    assert len(X_eq) == len(y_eq)


def test_equilibrar_nao_encolhe_classe_acima_do_alvo():
    X, y = conjunto_desbalanceado()
    _, y_eq = equilibrar(X, y, alvo=5, semente=0)
    assert int((y_eq == "B").sum()) == 10


def test_equilibrar_nao_inventa_classes():
    X, y = conjunto_desbalanceado()
    _, y_eq = equilibrar(X, y, alvo=20, semente=0)
    assert set(y_eq) == set(y)


def test_equilibrar_e_deterministico():
    X, y = conjunto_desbalanceado()
    a, ya = equilibrar(X, y, alvo=20, semente=42)
    b, yb = equilibrar(X, y, alvo=20, semente=42)
    assert np.allclose(a, b)
    assert np.array_equal(ya, yb)


def test_equilibrar_gera_vetores_normalizados():
    X, y = conjunto_desbalanceado()
    X_eq, _ = equilibrar(X, y, alvo=30, semente=1)
    assert all(esta_normalizado(vetor) for vetor in X_eq)


def test_equilibrar_rejeita_alvo_invalido():
    X, y = conjunto_desbalanceado()
    with pytest.raises(ValueError):
        equilibrar(X, y, alvo=0)
