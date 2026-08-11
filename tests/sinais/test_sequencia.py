import numpy as np
import pytest

from libras.sinais import pose, sequencia
from tests.sinais.apoio import gravacao as bruta


# --- validade ---


def test_ponto_medido_e_valido_ponto_ausente_nao():
    seq = bruta(3)
    seq[1, pose.MAO_DIREITA] = np.nan
    v = sequencia.validade(seq)

    assert v.shape == (3, pose.NUM_PONTOS)
    assert v[1, pose.MAO_DIREITA].sum() == 0
    assert v[0].all() and v[2].all()


def test_ponto_com_uma_coordenada_nan_nao_e_valido():
    seq = bruta(2)
    seq[0, 5, 2] = np.nan
    assert not sequencia.validade(seq)[0, 5]


# --- imputação ---


def test_preenche_um_buraco_entre_frames_bons():
    seq = bruta(5)
    seq[:, 0, 0] = [0.0, np.nan, np.nan, np.nan, 4.0]
    cheia = sequencia.imputar(seq)
    assert np.allclose(cheia[:, 0, 0], [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-4)


def test_nao_mexe_no_que_foi_medido():
    seq = bruta(5)
    seq[:, 0, 0] = [0.0, 1.0, np.nan, 3.0, 4.0]
    cheia = sequencia.imputar(seq)
    assert np.allclose(cheia[[0, 1, 3, 4], 0, 0], [0.0, 1.0, 3.0, 4.0])


def test_borda_segura_o_valor_em_vez_de_extrapolar():
    """Spline cúbica extrapolada dispara. Nas bordas ela é travada de propósito."""
    seq = bruta(8)
    seq[:, 0, 0] = [np.nan, np.nan, 1.0, 2.0, 4.0, 8.0, np.nan, np.nan]
    cheia = sequencia.imputar(seq)
    assert np.allclose(cheia[:2, 0, 0], 1.0)
    assert np.allclose(cheia[-2:, 0, 0], 8.0)


def test_canal_sem_nenhum_valor_vira_zero():
    """Não há de onde interpolar. Zero, e a máscara de validade conta a verdade."""
    seq = bruta(6)
    seq[:, pose.MAO_ESQUERDA] = np.nan
    cheia = sequencia.imputar(seq)
    assert np.allclose(cheia[:, pose.MAO_ESQUERDA], 0.0)
    assert np.isfinite(cheia).all()


def test_dois_pontos_validos_caem_para_linear():
    seq = bruta(5)
    seq[:, 0, 0] = [1.0, np.nan, np.nan, np.nan, 5.0]
    assert np.allclose(sequencia.imputar(seq)[:, 0, 0], [1, 2, 3, 4, 5], atol=1e-4)


# --- reamostragem ---


def test_reamostrar_encolhe_e_estica_preservando_as_pontas():
    arr = np.linspace(0, 1, 20).reshape(20, 1, 1).astype(np.float32)
    for alvo in (5, 20, 64):
        r = sequencia.reamostrar(arr, alvo)
        assert r.shape == (alvo, 1, 1)
        assert r[0, 0, 0] == pytest.approx(0.0, abs=1e-5)
        assert r[-1, 0, 0] == pytest.approx(1.0, abs=1e-5)


def test_reamostrar_de_um_frame_so_repete():
    r = sequencia.reamostrar(np.full((1, 2, 3), 7.0, dtype=np.float32), 4)
    assert r.shape == (4, 2, 3)
    assert np.allclose(r, 7.0)


def test_reamostrar_rejeita_sequencia_vazia():
    with pytest.raises(ValueError):
        sequencia.reamostrar(np.empty((0, 49, 3), dtype=np.float32), 32)


# --- preparar: o pipeline inteiro ---


def test_preparar_entrega_o_formato_que_a_busca_consome():
    s = sequencia.preparar(bruta(47))
    assert s.pontos.shape == (sequencia.T_PADRAO, pose.NUM_PONTOS, 3)
    assert s.validade.shape == (sequencia.T_PADRAO, pose.NUM_PONTOS)
    assert s.vetores.shape == (sequencia.T_PADRAO, pose.TAMANHO_VETOR)
    assert np.isfinite(s.pontos).all()


def test_preparar_normaliza_no_corpo():
    s = sequencia.preparar(bruta(20))
    largura = np.linalg.norm(
        s.pontos[:, pose.OMBRO_DIREITO, :2] - s.pontos[:, pose.OMBRO_ESQUERDO, :2],
        axis=1,
    )
    assert np.allclose(largura, 1.0, atol=1e-4)


def test_validade_denuncia_a_mao_que_faltou():
    seq = bruta(20)
    seq[:, pose.MAO_ESQUERDA] = np.nan
    s = sequencia.preparar(seq)
    assert s.validade[:, pose.MAO_ESQUERDA].max() == 0.0
    assert s.validade[:, pose.MAO_DIREITA].min() == 1.0


def test_mao_nunca_vista_repousa_no_centro_do_peito():
    """Sem nada para interpolar, o ponto vai para a origem — o meio dos ombros.

    Não é a posição verdadeira; é uma posição *combinada*. Duas gravações sem a
    mão esquerda ficam idênticas nesses canais em vez de discordarem por ruído,
    e o canto da imagem (que é onde o zero cru cairia) não vira uma mão a três
    larguras de ombro do corpo, inflando toda distância.
    """
    seq = bruta(20)
    seq[:, pose.MAO_ESQUERDA] = np.nan
    s = sequencia.preparar(seq)
    assert np.allclose(s.pontos[:, pose.MAO_ESQUERDA], 0.0)


def test_frame_sem_ombros_e_reconstruido_pelos_vizinhos():
    seq = bruta(20)
    seq[9, pose.POSE] = np.nan
    s = sequencia.preparar(seq)
    assert np.isfinite(s.pontos).all()
    assert s.validade.min() < 1.0


def test_gravacao_sem_nenhum_ombro_e_recusada():
    seq = bruta(10)
    seq[:, pose.POSE] = np.nan
    with pytest.raises(ValueError, match="ombro"):
        sequencia.preparar(seq)


def test_espelhar_troca_tambem_a_mascara_de_validade():
    """Se a validade não acompanhar o espelho, o modelo passa a acreditar que
    mediu a mão que faltou."""
    seq = bruta(20)
    seq[:, pose.MAO_ESQUERDA] = np.nan

    s = sequencia.preparar(seq, espelhar=True)
    assert s.validade[:, pose.MAO_DIREITA].max() == 0.0
    assert s.validade[:, pose.MAO_ESQUERDA].min() == 1.0
