import numpy as np
import pytest

from libras import pose, sequencia
from tests.apoio import gravacao as bruta


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


# --- características: o que entra no encoder ---


def test_velocidade_zera_o_primeiro_frame_e_mede_o_resto():
    """Zerado e não descartado: posição e velocidade têm que continuar alinhadas
    frame a frame para poderem ser canais da mesma sequência."""
    seq = np.array([[0.0, 0.0], [1.0, 2.0], [1.0, 5.0]], dtype=np.float32)

    v = sequencia.velocidade(seq)

    assert v.shape == seq.shape
    assert np.allclose(v[0], 0.0)
    assert np.allclose(v[1], [1.0, 2.0])
    assert np.allclose(v[2], [0.0, 3.0])


def test_velocidade_aceita_lote():
    assert sequencia.velocidade(np.zeros((4, 8, 3), dtype=np.float32)).shape == (4, 8, 3)


def test_z_normaliza_por_canal_ao_longo_do_tempo():
    seq = np.stack([np.arange(10.0), 100.0 + 3.0 * np.arange(10.0)], axis=1)

    z = sequencia.normalizar_z(seq)

    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-5)
    # Dois canais que só diferem por escala e deslocamento saem idênticos — é
    # exatamente a diferença entre pessoas que o z-score existe para absorver.
    assert np.allclose(z[:, 0], z[:, 1], atol=1e-5)


def test_z_de_canal_constante_e_zero_e_nao_infinito():
    """A mão que não apareceu em frame nenhum é constante. Dividir pelo desvio
    dela mandaria a sequência inteira para NaN."""
    z = sequencia.normalizar_z(np.zeros((8, 4), dtype=np.float32))

    assert np.isfinite(z).all()
    assert np.allclose(z, 0.0)


def test_caracteristicas_dobra_os_canais_com_a_velocidade():
    seq = np.zeros((sequencia.T_PADRAO, pose.TAMANHO_VETOR), dtype=np.float32)

    assert sequencia.caracteristicas(seq).shape == (32, 2 * pose.TAMANHO_VETOR)
    assert sequencia.caracteristicas(seq, com_velocidade=False).shape == (32, 147)


def test_caracteristicas_corta_o_ponto_implausivel():
    """O dicionário tem amostras chegando a ±400 larguras de ombro — imputação
    ruim, não gesto. Sem o corte, um ponto desses domina a rede inteira."""
    seq = np.full((8, 147), 400.0, dtype=np.float32)

    saida = sequencia.caracteristicas(seq)

    assert saida.max() <= sequencia.LIMITE_PLAUSIVEL
    assert saida.min() >= -sequencia.LIMITE_PLAUSIVEL


def test_caracteristicas_preserva_a_localizacao_por_padrao():
    """Localização é fonema em Libras: PAI e MÃE são a mesma mão em lugares
    diferentes. O padrão não pode apagar isso — só o `z=True` apaga, e ele avisa."""
    alto = np.full((8, 147), 0.5, dtype=np.float32)
    baixo = np.full((8, 147), -0.5, dtype=np.float32)

    assert not np.allclose(
        sequencia.caracteristicas(alto), sequencia.caracteristicas(baixo)
    )
    assert np.allclose(
        sequencia.caracteristicas(alto, z=True),
        sequencia.caracteristicas(baixo, z=True),
    )


def test_caracteristicas_aceita_lote():
    lote = np.zeros((5, 32, pose.TAMANHO_VETOR), dtype=np.float32)

    assert sequencia.caracteristicas(lote).shape == (5, 32, 294)


# --- configuração de mão ---


def _com_maos(esquerda: np.ndarray, direita: np.ndarray) -> np.ndarray:
    """(T, 147) com as duas mãos postas e o corpo zerado."""
    pontos = np.zeros((len(esquerda), pose.NUM_PONTOS, 3), dtype=np.float32)
    pontos[:, pose.MAO_ESQUERDA] = esquerda
    pontos[:, pose.MAO_DIREITA] = direita
    return pontos.reshape(len(esquerda), pose.TAMANHO_VETOR)


def _mao_aberta(escala: float, deslocamento: float) -> np.ndarray:
    """(1, 21, 3): uma mão de tamanho `escala`, posta em `deslocamento`."""
    m = np.zeros((1, pose.NUM_PONTOS_MAO, 3), dtype=np.float32)
    m[0, :, 0] = np.linspace(0.0, escala, pose.NUM_PONTOS_MAO)
    m[0, :, 1] = np.linspace(0.0, escala / 2, pose.NUM_PONTOS_MAO)
    return m + deslocamento


def test_maos_locais_ignoram_onde_a_mao_esta_e_de_que_tamanho_ela_e():
    """É o ponto do canal: a mesma mão perto e longe da câmera, em lugares
    diferentes do corpo, tem que sair idêntica. Quem guarda o *onde* é a
    posição, que continua na entrada ao lado deste."""
    perto = _com_maos(_mao_aberta(0.3, 0.0), _mao_aberta(0.3, 0.0))
    longe = _com_maos(_mao_aberta(0.15, 1.7), _mao_aberta(0.15, 1.7))

    assert np.allclose(
        sequencia.maos_locais(perto), sequencia.maos_locais(longe), atol=1e-5
    )


def test_maos_locais_separam_configuracoes_diferentes():
    aberta = _com_maos(_mao_aberta(0.3, 0.0), _mao_aberta(0.3, 0.0))
    torta = aberta.copy()
    torta.reshape(1, pose.NUM_PONTOS, 3)[0, 5, 1] += 0.2

    assert not np.allclose(
        sequencia.maos_locais(aberta), sequencia.maos_locais(torta)
    )


def test_mao_ausente_sai_zerada_e_nao_dividida_por_quase_zero():
    """`preparar` zera o ponto nunca visto. Dividir isso pela própria escala
    transformaria ruído numérico em dedo."""
    seq = _com_maos(
        np.zeros((1, pose.NUM_PONTOS_MAO, 3), dtype=np.float32),
        _mao_aberta(0.3, 0.0),
    )
    locais = sequencia.maos_locais(seq)

    esquerda = locais[:, : sequencia.NUM_PONTOS_MAO_LOCAL * 3]
    assert np.allclose(esquerda, 0.0)
    assert np.isfinite(locais).all()


def test_caracteristicas_acrescenta_os_canais_de_mao():
    seq = np.zeros((32, pose.TAMANHO_VETOR), dtype=np.float32)

    assert sequencia.caracteristicas(seq, com_maos=True).shape == (
        32,
        2 * pose.TAMANHO_VETOR + sequencia.TAMANHO_MAOS_LOCAIS,
    )
    assert sequencia.caracteristicas(
        np.zeros((5, 32, pose.TAMANHO_VETOR), dtype=np.float32), com_maos=True
    ).shape == (5, 32, 414)


def test_maos_locais_saem_antes_do_z_e_sobrevivem_a_ele():
    """O z-score apagaria a escala da mão junto com a do corpo. Os canais de mão
    são calculados antes dele, de propósito."""
    seq = _com_maos(_mao_aberta(0.3, 0.0), _mao_aberta(0.3, 0.5))

    com_z = sequencia.caracteristicas(seq, com_maos=True, z=True)
    sem_z = sequencia.caracteristicas(seq, com_maos=True, z=False)

    assert np.allclose(
        com_z[..., -sequencia.TAMANHO_MAOS_LOCAIS :],
        sem_z[..., -sequencia.TAMANHO_MAOS_LOCAIS :],
    )


# --- imputação que não ultrapassa ---


def test_imputacao_nao_sai_do_intervalo_medido():
    """A cúbica natural disparava: um buraco longo entre dois valores próximos
    saía com um pico de centenas de larguras de ombro, e o dicionário extraído
    com ela tinha pontos a 218. PCHIP não tem como ultrapassar."""
    seq = bruta(24)
    medido = [0.0, 0.1, 0.0, -0.1] + [np.nan] * 16 + [0.2, 0.1, 0.0, -0.1]
    seq[:, 0, 0] = medido

    cheia = sequencia.imputar(seq)[:, 0, 0]

    validos = [v for v in medido if np.isfinite(v)]
    assert cheia.min() >= min(validos) - 1e-5
    assert cheia.max() <= max(validos) + 1e-5
