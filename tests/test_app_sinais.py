"""A persistência do que a pessoa ensina ao dicionário.

Só isto é testável sem câmera no `app_sinais`, e é justamente a parte que não
pode falhar em silêncio: a re-ancoragem manual é a mitigação do viés dos três
articuladores, e ela vive em memória até `salvar_prototipos`. Um protótipo
perdido aqui é uma correção que a pessoa vai ter que fazer de novo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libras import app_sinais, sequencia
from libras.dicionario import FONTE_USUARIO, Dicionario


def prototipo(valor: float = 0.0) -> np.ndarray:
    return np.full((sequencia.T_PADRAO, 147), valor, dtype=np.float32)


def dicionario_base() -> Dicionario:
    return Dicionario(
        representacoes=np.stack([prototipo(0.1), prototipo(0.2)]),
        rotulos=["CASA", "ÁGUA"],
        fontes=["articulador_1", "articulador_2"],
        metrica="dtw",
    )


@pytest.fixture
def caminhos(tmp_path: Path, monkeypatch):
    base = tmp_path / "dicionario.npz"
    meus = tmp_path / "meus_prototipos.npz"
    monkeypatch.setattr(app_sinais.config, "DICIONARIO_SINAIS", base)
    monkeypatch.setattr(app_sinais.config, "PROTOTIPOS_USUARIO", meus)
    return base, meus


def test_salvar_grava_so_o_que_e_seu(caminhos):
    """O dicionário base nunca é reescrito — senão uma sessão ruim o estragaria."""
    _, meus = caminhos
    dicionario = dicionario_base()
    dicionario.ancorar(prototipo(0.9), "LIVRO")

    quantos = app_sinais.salvar_prototipos(dicionario)

    assert quantos == 1
    salvo = Dicionario.carregar(meus)
    assert salvo.rotulos == ["LIVRO"]
    assert salvo.fontes == [FONTE_USUARIO]


def test_sem_nada_ensinado_nao_cria_arquivo(caminhos):
    _, meus = caminhos

    assert app_sinais.salvar_prototipos(dicionario_base()) == 0
    assert not meus.exists()


def test_carregar_junta_a_base_com_os_seus(caminhos):
    base, _ = caminhos
    dicionario_base().salvar(base)

    meu = dicionario_base()
    meu.ancorar(prototipo(0.9), "LIVRO")
    app_sinais.salvar_prototipos(meu)

    carregado = app_sinais.carregar_dicionario()

    assert len(carregado) == 3
    assert "LIVRO" in carregado.rotulos
    assert FONTE_USUARIO in carregado.fontes


def dicionario_com_sinal_de_fora() -> Dicionario:
    """CASA está no núcleo curado; ABACAXI não."""
    return Dicionario(
        representacoes=np.stack([prototipo(0.1), prototipo(0.2)]),
        rotulos=["CASA", "ABACAXI"],
        fontes=["articulador_1", "articulador_2"],
        metrica="dtw",
    )


def test_por_padrao_o_app_indexa_so_o_nucleo(caminhos):
    """1.364 sinais dão 7,5% de recall@5; 163 dão 37,3%. O recorte é o produto."""
    base, _ = caminhos
    dicionario_com_sinal_de_fora().salvar(base)

    assert app_sinais.carregar_dicionario().vocabulario == ["CASA"]


def test_tudo_devolve_o_vocabulario_inteiro(caminhos):
    base, _ = caminhos
    dicionario_com_sinal_de_fora().salvar(base)

    carregado = app_sinais.carregar_dicionario(apenas_nucleo=False)

    assert carregado.vocabulario == ["ABACAXI", "CASA"]


def test_o_que_voce_ensinou_entra_mesmo_fora_do_nucleo(caminhos):
    """O recorte é do V-LIBRASIL, não seu. Ensinar um sinal de fora é o caso em
    que ele não pode atrapalhar — é para isso que a tecla N existe."""
    base, _ = caminhos
    dicionario_com_sinal_de_fora().salvar(base)

    meu = dicionario_base()
    meu.ancorar(prototipo(0.9), "ABACAXI")
    app_sinais.salvar_prototipos(meu)

    carregado = app_sinais.carregar_dicionario()

    assert carregado.vocabulario == ["ABACAXI", "CASA"]
    assert carregado.rotulos.count("ABACAXI") == 1  # o seu, não o do dataset


def test_carregar_sem_dicionario_diz_o_que_fazer(caminhos):
    """A mensagem é a única pista de quem chegou aqui sem rodar a extração."""
    with pytest.raises(FileNotFoundError, match="prepare_sinais"):
        app_sinais.carregar_dicionario()


def test_ida_e_volta_preserva_os_vetores(caminhos):
    _, meus = caminhos
    dicionario = dicionario_base()
    dicionario.ancorar(prototipo(0.42), "LIVRO")

    app_sinais.salvar_prototipos(dicionario)

    salvo = Dicionario.carregar(meus)
    assert np.allclose(salvo.buscar(prototipo(0.42), k=1)[0].similaridade, 1.0, atol=0.2)


# --- qual busca o app escolhe ---
#
# O encoder entra sem que o loop saiba: quando o dicionário de embeddings existe,
# a representação passa a ser um vetor de 256d e a métrica passa a ser cosseno.
# O que não pode acontecer é os dois se misturarem — um embedding gravado no npz
# das sequências daria um dicionário que carrega e erra tudo.


def gravacao_pronta() -> sequencia.Sequencia:
    pontos = np.zeros((sequencia.T_PADRAO, 49, 3), dtype=np.float32)
    return sequencia.Sequencia(pontos=pontos, validade=np.ones((sequencia.T_PADRAO, 49)))


@pytest.fixture
def caminhos_encoder(caminhos, tmp_path, monkeypatch):
    """Os quatro arquivos do caminho do encoder, todos fora do repo."""
    alvos = {
        "DICIONARIO_EMBEDDINGS": tmp_path / "embeddings.npz",
        "PROTOTIPOS_USUARIO_EMBEDDINGS": tmp_path / "meus_embeddings.npz",
        "ENCODER_SINAIS": tmp_path / "encoder.pt",
    }
    for nome, caminho in alvos.items():
        monkeypatch.setattr(app_sinais.config, nome, caminho)
    return alvos


def test_sem_encoder_o_app_usa_a_baseline_dtw(caminhos, caminhos_encoder):
    base, _ = caminhos
    dicionario_base().salvar(base)

    busca = app_sinais.carregar_busca()

    assert busca.dicionario.metrica == "dtw"
    assert busca.prototipos == app_sinais.config.PROTOTIPOS_USUARIO
    # Com a máscara junto, mesmo o DTW não a usando: o protótipo que você ensina
    # tem que ter a largura dos que vieram do disco, senão a re-ancoragem quebra.
    assert busca.representar(gravacao_pronta()).shape == (
        sequencia.T_PADRAO,
        sequencia.TAMANHO_COM_VALIDADE,
    )


def test_com_encoder_a_consulta_vira_embedding(caminhos, caminhos_encoder):
    encoder = pytest.importorskip("libras.encoder")

    modelo = encoder.Codificador(
        encoder.Hiperparametros(dimensao=16, oculto=8, camadas=1, dropout=0.0)
    )
    encoder.salvar(modelo, caminhos_encoder["ENCODER_SINAIS"])
    Dicionario(
        representacoes=np.zeros((2, 16), dtype=np.float32),
        rotulos=["CASA", "ÁGUA"],
        fontes=["articulador_1", "articulador_2"],
        metrica="cosseno",
    ).salvar(caminhos_encoder["DICIONARIO_EMBEDDINGS"])

    busca = app_sinais.carregar_busca()

    assert busca.dicionario.metrica == "cosseno"
    assert busca.prototipos == caminhos_encoder["PROTOTIPOS_USUARIO_EMBEDDINGS"]
    assert busca.representar(gravacao_pronta()).shape == (16,)


def test_embeddings_sem_modelo_diz_o_que_rodar(caminhos, caminhos_encoder):
    pytest.importorskip("libras.encoder")
    Dicionario(
        representacoes=np.zeros((1, 16), dtype=np.float32),
        rotulos=["CASA"],
        fontes=["articulador_1"],
        metrica="cosseno",
    ).salvar(caminhos_encoder["DICIONARIO_EMBEDDINGS"])

    with pytest.raises(FileNotFoundError, match="train_sinais"):
        app_sinais.carregar_busca()


def test_prototipos_de_embedding_nao_caem_no_npz_das_sequencias(
    caminhos, caminhos_encoder
):
    dicionario = Dicionario(
        representacoes=np.zeros((1, 16), dtype=np.float32),
        rotulos=["CASA"],
        fontes=["articulador_1"],
        metrica="cosseno",
    )
    dicionario.ancorar(np.full(16, 0.5, dtype=np.float32), "LIVRO")

    assert app_sinais.salvar_prototipos(dicionario) == 1
    assert caminhos_encoder["PROTOTIPOS_USUARIO_EMBEDDINGS"].exists()
    assert not app_sinais.config.PROTOTIPOS_USUARIO.exists()
