import numpy as np
import pytest

from libras import pose
from tests.apoio import corpo, frame, mao


# --- montagem do frame ---


def test_frame_vazio_e_todo_nan():
    f = pose.montar_frame()
    assert f.shape == (pose.NUM_PONTOS, 3)
    assert np.isnan(f).all()


def test_mao_ausente_vira_nan_sem_afetar_a_outra():
    f = frame(mao_esquerda=None)
    assert np.isnan(f[pose.MAO_ESQUERDA]).all()
    assert not np.isnan(f[pose.MAO_DIREITA]).any()


def test_corpo_e_reduzido_ao_subconjunto():
    f = frame()
    assert np.allclose(f[pose.OMBRO_ESQUERDO], corpo()[pose.POSE_OMBRO_ESQUERDO])
    assert np.allclose(f[pose.NARIZ], corpo()[pose.POSE_NARIZ])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mao_esquerda": np.zeros((20, 3))},
        {"mao_direita": np.zeros((21, 2))},
        {"corpo": np.zeros((7, 3))},
    ],
)
def test_rejeita_shapes_invalidos(kwargs):
    with pytest.raises(ValueError):
        frame(**kwargs)


# --- normalização ancorada no corpo ---


def test_ombros_viram_origem_e_escala_unitaria():
    n = pose.normalizar_frame(frame())
    meio = (n[pose.OMBRO_ESQUERDO] + n[pose.OMBRO_DIREITO]) / 2
    assert np.allclose(meio, 0.0, atol=1e-5)
    largura = np.linalg.norm(n[pose.OMBRO_DIREITO][:2] - n[pose.OMBRO_ESQUERDO][:2])
    assert largura == pytest.approx(1.0, abs=1e-5)


def test_indiferente_a_posicao_na_tela():
    perto_da_borda = frame(corpo=corpo(centro=(0.15, 0.8, 0.0)))
    no_meio = frame(corpo=corpo(centro=(0.5, 0.5, 0.0)))
    assert np.allclose(
        pose.normalizar_frame(perto_da_borda)[pose.POSE],
        pose.normalizar_frame(no_meio)[pose.POSE],
        atol=1e-5,
    )


def test_indiferente_a_distancia_da_camera():
    longe = frame(corpo=corpo(largura_ombros=0.1))
    perto = frame(corpo=corpo(largura_ombros=0.4))
    assert np.allclose(
        pose.normalizar_frame(longe)[pose.POSE],
        pose.normalizar_frame(perto)[pose.POSE],
        atol=1e-5,
    )


def test_preserva_onde_a_mao_esta_no_corpo():
    """A propriedade que separa esta normalização da do alfabeto.

    Mão na testa e mão no peito são a mesma forma de mão em posições
    diferentes. Para a fase 1 os dois vetores seriam idênticos; aqui não podem
    ser, porque localização é fonema em Libras.
    """
    c = corpo()
    na_testa = frame(mao_direita=mao() + np.array([0.0, -0.3, 0.0]), corpo=c)
    no_peito = frame(mao_direita=mao() + np.array([0.0, 0.2, 0.0]), corpo=c)

    a = pose.normalizar_frame(na_testa)[pose.MAO_DIREITA]
    b = pose.normalizar_frame(no_peito)[pose.MAO_DIREITA]
    assert not np.allclose(a, b, atol=1e-3)


def test_sem_ombros_o_frame_inteiro_e_descartado():
    c = corpo()
    c[pose.POSE_OMBRO_ESQUERDO] = np.nan
    assert np.isnan(pose.normalizar_frame(frame(corpo=c))).all()


def test_ombros_colados_nao_explodem_a_escala():
    n = pose.normalizar_frame(frame(corpo=corpo(largura_ombros=0.0)))
    assert np.isnan(n).all()


def test_nan_da_mao_sobrevive_a_normalizacao():
    n = pose.normalizar_frame(frame(mao_esquerda=None))
    assert np.isnan(n[pose.MAO_ESQUERDA]).all()
    assert not np.isnan(n[pose.MAO_DIREITA]).any()


# --- espelhamento ---


def test_espelhar_troca_as_maos_de_lado():
    n = pose.normalizar_frame(frame())
    e = pose.espelhar_frame(n)
    assert np.allclose(e[pose.MAO_ESQUERDA][:, 0], -n[pose.MAO_DIREITA][:, 0])
    assert np.allclose(e[pose.MAO_ESQUERDA][:, 1:], n[pose.MAO_DIREITA][:, 1:])


def test_espelhar_duas_vezes_volta_ao_original():
    n = pose.normalizar_frame(frame())
    assert np.allclose(pose.espelhar_frame(pose.espelhar_frame(n)), n, atol=1e-6)


def test_espelhar_troca_os_pares_da_pose():
    n = pose.normalizar_frame(frame())
    e = pose.espelhar_frame(n)
    esperado = n[pose.OMBRO_DIREITO] * np.array([-1.0, 1.0, 1.0])
    assert np.allclose(e[pose.OMBRO_ESQUERDO], esperado, atol=1e-6)


def test_espelhar_mantem_o_nariz_no_lugar():
    """O nariz não tem par — só o x dele inverte, e ele está no eixo."""
    n = pose.normalizar_frame(frame())
    e = pose.espelhar_frame(n)
    assert np.allclose(e[pose.NARIZ][1:], n[pose.NARIZ][1:], atol=1e-6)


# --- sequências ---


def test_normalizar_sequencia_frame_a_frame():
    seq = np.stack([frame(), frame(corpo=corpo(centro=(0.2, 0.3, 0.0)))])
    n = pose.normalizar_sequencia(seq)
    assert n.shape == (2, pose.NUM_PONTOS, 3)
    assert np.allclose(n[0][pose.POSE], n[1][pose.POSE], atol=1e-5)


def test_vetores_achata_para_147():
    seq = pose.normalizar_sequencia(np.stack([frame(), frame()]))
    assert pose.vetores(seq).shape == (2, pose.TAMANHO_VETOR)
