import numpy as np
import pytest

from libras.sinais.dicionario import Dicionario


def traj(inclinacao: float, n: int = 16) -> np.ndarray:
    """Uma trajetória (T, 2) cuja forma depende só de `inclinacao`."""
    t = np.linspace(0.0, 1.0, n)[:, None].astype(np.float32)
    return np.concatenate([t, t * inclinacao], axis=1)


def dicionario(**kwargs) -> Dicionario:
    """Três sinais, dois articuladores cada — a forma do V-LIBRASIL, em miniatura."""
    base = dict(
        representacoes=np.stack(
            [
                traj(0.0), traj(0.05),      # CASA
                traj(1.0), traj(1.05),      # PORTA
                traj(2.0), traj(2.05),      # LIVRO
            ]
        ),
        rotulos=["CASA", "CASA", "PORTA", "PORTA", "LIVRO", "LIVRO"],
        fontes=["art1", "art2", "art1", "art2", "art1", "art2"],
    )
    return Dicionario(**{**base, **kwargs})


# --- construção ---


def test_tamanho_e_vocabulario():
    d = dicionario()
    assert len(d) == 6
    assert d.vocabulario == ["CASA", "LIVRO", "PORTA"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rotulos": ["CASA"]},
        {"fontes": ["art1"]},
        {"representacoes": np.zeros((6, 16))},  # ndim errado para DTW
    ],
)
def test_rejeita_entradas_desalinhadas(kwargs):
    with pytest.raises(ValueError):
        dicionario(**kwargs)


def test_dicionario_vazio_nao_devolve_candidato():
    d = Dicionario(representacoes=np.zeros((0, 16, 2)), rotulos=[], fontes=[])
    assert d.buscar(traj(0.0)) == []


# --- busca ---


def test_acha_o_sinal_certo():
    assert dicionario().buscar(traj(0.02))[0].rotulo == "CASA"


def test_devolve_no_maximo_k_candidatos_em_ordem():
    resultado = dicionario().buscar(traj(0.9), k=2)
    assert len(resultado) == 2
    assert resultado[0].distancia <= resultado[1].distancia


def test_k_maior_que_o_vocabulario_devolve_o_vocabulario():
    assert len(dicionario().buscar(traj(0.0), k=99)) == 3


def test_cada_rotulo_aparece_uma_vez_so():
    """Dois articuladores do mesmo sinal não podem ocupar duas linhas da tela."""
    rotulos = [c.rotulo for c in dicionario().buscar(traj(0.0), k=5)]
    assert len(rotulos) == len(set(rotulos))


def test_o_rotulo_fica_com_a_distancia_do_seu_melhor_prototipo():
    d = dicionario()
    consulta = traj(0.05)
    casa = next(c for c in d.buscar(consulta, k=3) if c.rotulo == "CASA")
    assert casa.distancia == pytest.approx(0.0, abs=1e-5)


def test_reporta_de_qual_fonte_veio_a_melhor_correspondencia():
    casa = dicionario().buscar(traj(0.05), k=1)[0]
    assert casa.fonte == "art2"


# --- similaridade ---


def test_similaridade_cai_quando_a_distancia_sobe():
    resultado = dicionario().buscar(traj(0.0), k=3)
    similaridades = [c.similaridade for c in resultado]
    assert similaridades == sorted(similaridades, reverse=True)


def test_similaridade_fica_entre_zero_e_um():
    for c in dicionario().buscar(traj(5.0), k=3):
        assert 0.0 <= c.similaridade <= 1.0


# --- re-ancoragem ---


def test_ancorar_um_sinal_novo_aumenta_o_vocabulario():
    d = dicionario()
    d.ancorar(traj(-3.0), "OBRIGADO")
    assert "OBRIGADO" in d.vocabulario
    assert d.buscar(traj(-3.0), k=1)[0].rotulo == "OBRIGADO"


def test_ancorar_a_sua_mao_reordena_o_resultado():
    """A mitigação do viés dos três articuladores, medida.

    Antes, a consulta cai mais perto de PORTA. Depois de gravar a sua versão de
    CASA, ela passa a cair em CASA — sem retreinar nada.
    """
    d = dicionario()
    consulta = traj(0.8)
    assert d.buscar(consulta, k=1)[0].rotulo == "PORTA"

    d.ancorar(consulta, "CASA")
    assert d.buscar(consulta, k=1)[0].rotulo == "CASA"


def test_prototipo_seu_nasce_marcado_como_seu():
    d = dicionario()
    d.ancorar(traj(9.0), "TCHAU")
    assert d.buscar(traj(9.0), k=1)[0].fonte == "voce"


def test_remover_uma_fonte_nao_toca_nas_outras():
    d = dicionario()
    d.ancorar(traj(9.0), "TCHAU")
    d.remover_fonte("voce")
    assert "TCHAU" not in d.vocabulario
    assert len(d) == 6


def test_ancorar_recusa_representacao_de_outro_formato():
    with pytest.raises(ValueError):
        dicionario().ancorar(np.zeros((16, 5), dtype=np.float32), "X")


# --- separar, juntar, filtrar ---


def test_apenas_isola_os_prototipos_de_uma_fonte():
    """É o que se grava em disco ao sair: só o seu, nunca o dicionário base."""
    d = dicionario()
    d.ancorar(traj(9.0), "TCHAU")

    meus = d.apenas("voce")
    assert len(meus) == 1
    assert meus.vocabulario == ["TCHAU"]
    assert len(d) == 7  # o original não foi tocado


def test_apenas_de_uma_fonte_inexistente_da_dicionario_vazio():
    assert len(dicionario().apenas("ninguem")) == 0


def test_juntar_soma_os_prototipos():
    base = dicionario()
    meus = Dicionario(
        representacoes=traj(9.0)[None], rotulos=["TCHAU"], fontes=["voce"]
    )
    juntos = base.juntar(meus)

    assert len(juntos) == 7
    assert "TCHAU" in juntos.vocabulario
    assert len(base) == 6  # juntar não modifica nenhum dos dois


def test_juntar_com_vazio_nos_dois_sentidos():
    base = dicionario()
    vazio = Dicionario(np.zeros((0, 16, 2), dtype=np.float32), [], [])
    assert len(base.juntar(vazio)) == 6
    assert len(vazio.juntar(base)) == 6


def test_juntar_metricas_diferentes_e_recusado():
    """Sequência e embedding não se misturam no mesmo índice."""
    base = dicionario()
    outro = Dicionario(np.eye(2, dtype=np.float32), ["A", "B"], ["x", "x"],
                       metrica="cosseno")
    with pytest.raises(ValueError, match="métrica"):
        base.juntar(outro)


def test_separar_fonte_tira_o_articulador_do_indice():
    base, consultas = dicionario().separar_fonte("art2")
    assert "art2" not in base.fontes
    assert len(base) == 3 and len(consultas) == 3


# --- persistência ---


def test_salvar_e_carregar_preserva_a_busca(tmp_path):
    d = dicionario()
    d.ancorar(traj(-3.0), "OBRIGADO")

    caminho = tmp_path / "dic.npz"
    d.salvar(caminho)
    recarregado = Dicionario.carregar(caminho)

    assert recarregado.vocabulario == d.vocabulario
    assert len(recarregado) == len(d)
    assert recarregado.buscar(traj(-3.0), k=1)[0].rotulo == "OBRIGADO"


def test_carregar_preserva_a_metrica(tmp_path):
    d = dicionario(representacoes=np.eye(6, 4, dtype=np.float32), metrica="cosseno")
    caminho = tmp_path / "dic.npz"
    d.salvar(caminho)
    assert Dicionario.carregar(caminho).metrica == "cosseno"


# --- métrica alternativa (o lugar onde o encoder vai entrar) ---


def test_cosseno_funciona_sobre_vetores_achatados():
    d = Dicionario(
        representacoes=np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        ),
        rotulos=["DIREITA", "CIMA", "ESQUERDA"],
        fontes=["e", "e", "e"],
        metrica="cosseno",
    )
    assert d.buscar(np.array([0.9, 0.1], dtype=np.float32), k=1)[0].rotulo == "DIREITA"


def test_metrica_desconhecida_e_recusada_na_construcao():
    with pytest.raises(ValueError, match="métrica"):
        dicionario(metrica="telepatia")
