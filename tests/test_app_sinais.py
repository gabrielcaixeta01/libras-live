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
