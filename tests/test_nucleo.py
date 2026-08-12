"""O vocabulário núcleo e o recorte que ele faz no dicionário.

O que estes testes protegem é sutil: um erro de grafia na lista curada tiraria
um sinal do índice **em silêncio**. O dicionário ficaria menor, o recall subiria
(menos candidatos para confundir), e nada quebraria. Por isso `ausentes` existe e
por isso ela é testada aqui contra o dicionário de verdade quando ele está
presente.
"""

from __future__ import annotations

import numpy as np
import pytest

from libras import catalogo, config, nucleo
from libras.dicionario import Dicionario


def traj(inclinacao: float, n: int = 8) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None].astype(np.float32)
    return np.concatenate([t, t * inclinacao], axis=1)


# --- a lista ---


def test_nucleo_nao_tem_repetidos():
    """Duas grafias da mesma chave seriam duas linhas para uma classe só."""
    chaves = [catalogo.chave(p) for p in nucleo.VOCABULARIO]
    assert len(chaves) == len(set(chaves)), "há palavras repetidas por chave"


def test_nucleo_e_pequeno_de_proposito():
    """O recorte só compra acerto enquanto for recorte.

    Com 1.363 sinais o recall@5 é 7,5%; com ~160 ele é 37,3%. Uma lista que
    cresce até virar o vocabulário inteiro devolve o problema que ela resolve.
    """
    assert 80 <= len(nucleo.CHAVES) <= 300


def test_nucleo_exclui_o_alfabeto_manual():
    """`LETRA A`…`LETRA Z` são o outro modo do projeto, com classificador próprio."""
    assert not [p for p in nucleo.VOCABULARIO if p.startswith("LETRA ")]


def test_contem_ignora_acento_e_caixa():
    assert nucleo.contem("obrigado")
    assert nucleo.contem("AMANHA")
    assert not nucleo.contem("ABACAXI")


def test_ausentes_aponta_o_que_falta():
    assert nucleo.ausentes(["OI", "OBRIGADO"])  # o dicionário só tem dois
    assert nucleo.ausentes(nucleo.VOCABULARIO) == []


def test_grupos_cobrem_o_vocabulario():
    """A lista plana e os grupos não podem divergir — a documentação usa os dois."""
    dos_grupos = [p for grupo in nucleo.GRUPOS.values() for p in grupo]
    assert dos_grupos == nucleo.VOCABULARIO


# --- o recorte no dicionário ---


def dicionario_de_teste() -> Dicionario:
    rotulos = ["OI", "OI", "ABACAXI", "ABACAXI", "OBRIGADO"]
    return Dicionario(
        representacoes=np.stack([traj(float(i)) for i in range(len(rotulos))]),
        rotulos=rotulos,
        fontes=["art1", "art2", "art1", "art2", "art1"],
    )


def test_restringir_mantem_so_o_vocabulario_pedido():
    recortado = dicionario_de_teste().restringir(nucleo.CHAVES)
    assert recortado.vocabulario == ["OBRIGADO", "OI"]
    assert len(recortado) == 3


def test_restringir_preserva_fontes_e_representacoes():
    """O recorte é de linhas, não uma reconstrução: quem sobra sobra inteiro."""
    completo = dicionario_de_teste()
    recortado = completo.restringir(["OI"])

    assert recortado.procedencias == ["art1", "art2"]
    np.testing.assert_array_equal(
        recortado.representacoes, completo.representacoes[:2]
    )


def test_restringir_aceita_grafia_sem_acento():
    d = Dicionario(
        representacoes=np.stack([traj(0.0), traj(1.0)]),
        rotulos=["AMANHÃ", "ABACAXI"],
        fontes=["art1", "art1"],
    )
    assert d.restringir(["AMANHA"]).vocabulario == ["AMANHÃ"]


def test_restringir_com_vocabulario_desconhecido_esvazia():
    """Devolver o dicionário inteiro aqui seria pior: o app indexaria 1.364."""
    assert len(dicionario_de_teste().restringir(["INEXISTENTE"])) == 0


def test_restringir_preserva_a_metrica():
    d = Dicionario(
        representacoes=np.zeros((2, 4), dtype=np.float32),
        rotulos=["OI", "ABACAXI"],
        fontes=["art1", "art1"],
        metrica="cosseno",
    )
    assert d.restringir(nucleo.CHAVES).metrica == "cosseno"


# --- o limiar de rejeição ---


def test_limiar_por_metrica():
    assert config.limiar_de_rejeicao("cosseno") == config.REJEICAO_COSSENO
    assert config.limiar_de_rejeicao("dtw") == config.REJEICAO_DTW


def test_metrica_desconhecida_nao_rejeita():
    """None desliga a rejeição; inventar um corte seria pior que não ter."""
    assert config.limiar_de_rejeicao("inexistente") is None


# --- contra o dicionário de verdade ---


@pytest.mark.skipif(
    not config.DICIONARIO_SINAIS.exists(),
    reason="precisa do dicionário extraído do V-LIBRASIL",
)
def test_todo_o_nucleo_existe_no_vlibrasil():
    """A grafia da lista curada tem que casar com a do dataset, ou o sinal some."""
    d = Dicionario.carregar(config.DICIONARIO_SINAIS)
    assert nucleo.ausentes(d.vocabulario) == []
