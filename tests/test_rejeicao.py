"""A regra de recusa e a sua calibração.

O que se testa aqui é o que quebraria em silêncio: um corte calibrado na ponta
errada da distribuição não dá erro nenhum — ele só recusa exatamente as
consultas que teriam acertado, e o app fica pior sem que nada pareça quebrado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from libras import rejeicao


def consulta(primeira: float, segunda: float = math.inf, acertou: bool = True):
    return rejeicao.Consulta(primeira=primeira, segunda=segunda, acertou=acertou)


@dataclass(frozen=True)
class Falso:
    """Um candidato com o único campo que a regra olha."""

    distancia: float


# --- a margem ---


def test_margem_e_a_distancia_ate_o_segundo():
    assert consulta(0.2, 0.5).margem == pytest.approx(0.3)


def test_candidato_unico_tem_margem_infinita():
    """Não há com quem empatar: é o caso mais confiante que existe."""
    assert consulta(0.2).margem == math.inf


# --- calibração por distância ---


def test_o_limiar_separa_o_que_tem_resposta_do_que_nao_tem():
    """Distâncias baixas acertam, altas não: o corte tem que cair no meio."""
    medidas = [
        consulta(0.1, acertou=True),
        consulta(0.2, acertou=True),
        consulta(0.8, acertou=False),
        consulta(0.9, acertou=False),
    ]
    corte = rejeicao.calibrar(medidas)

    assert 0.2 <= corte.limiar < 0.8
    assert corte.aceitos_certos == 2
    assert corte.aceitos_errados == 0
    assert corte.precisao == 1.0
    assert corte.cobertura == 1.0
    assert corte.recusa_correta == 1.0


def test_o_corte_preserva_a_cobertura_pedida():
    """A regra é um quantil dos acertos, e é isso que o app promete manter."""
    medidas = [consulta(i / 100) for i in range(100)]
    corte = rejeicao.calibrar(medidas, cobertura_minima=0.90)

    assert corte.cobertura >= 0.90
    assert corte.limiar == pytest.approx(0.89)


def test_cobertura_mais_alta_nao_corta_menos():
    """Pedir mais acertos preservados só pode afrouxar o corte."""
    medidas = [
        consulta(0.1, acertou=True),
        consulta(0.5, acertou=True),
        consulta(0.6, acertou=False),
        consulta(0.9, acertou=True),
    ]

    frouxo = rejeicao.calibrar(medidas, cobertura_minima=1.0)
    apertado = rejeicao.calibrar(medidas, cobertura_minima=0.5)

    assert frouxo.limiar >= apertado.limiar
    assert frouxo.cobertura == 1.0


def test_o_limiar_conta_o_que_ele_custa():
    """Nenhum corte separa tudo, e o relatório precisa dizer o que se perde."""
    medidas = [
        consulta(0.1, acertou=True),
        consulta(0.5, acertou=False),
        consulta(0.6, acertou=True),
        consulta(0.9, acertou=False),
    ]
    corte = rejeicao.calibrar(medidas)

    assert corte.aceitos_certos + corte.recusados_certos == 2
    assert corte.aceitos_errados + corte.recusados_errados == 2
    assert 0.0 <= corte.precisao <= 1.0


def test_sem_acerto_nenhum_nao_ha_o_que_calibrar():
    """O corte é um quantil dos acertos; sem acertos ele sairia do acaso."""
    assert rejeicao.calibrar([consulta(0.1, acertou=False)]) is None
    assert rejeicao.calibrar([]) is None


def test_so_acertos_aceita_tudo():
    corte = rejeicao.calibrar([consulta(0.1), consulta(0.2)])

    assert corte.aceitos_certos == 2
    assert corte.recusados_errados == 0


def test_cobertura_invalida_e_recusada():
    with pytest.raises(ValueError):
        rejeicao.calibrar([consulta(0.1)], cobertura_minima=0.0)


def test_criterio_desconhecido_e_recusado():
    with pytest.raises(ValueError):
        rejeicao.calibrar([consulta(0.1)], criterio="chute")


# --- calibração por margem ---


def test_a_margem_corta_na_ponta_de_baixo():
    """Margem grande é boa. O quantil vem da outra ponta que o da distância, e
    trocar as pontas recusaria exatamente quem tinha razão."""
    medidas = [
        consulta(0.1, 0.9, acertou=True),   # margem 0.8, folgada
        consulta(0.1, 0.7, acertou=True),   # margem 0.6
        consulta(0.4, 0.41, acertou=False),  # margem 0.01, empate
        consulta(0.4, 0.42, acertou=False),
    ]
    corte = rejeicao.calibrar(medidas, criterio="margem")

    assert corte.aceitos_certos == 2
    assert corte.recusados_errados == 2
    assert corte.recusa_correta == 1.0


def test_margem_tambem_preserva_a_cobertura_pedida():
    medidas = [consulta(0.0, i / 100) for i in range(1, 101)]
    corte = rejeicao.calibrar(medidas, criterio="margem", cobertura_minima=0.90)

    assert corte.cobertura >= 0.90


# --- a regra como o app a aplica ---


def test_lista_vazia_nunca_e_recusa():
    """Não há o que recusar, e quem chama já sabe que não tem o que mostrar."""
    assert rejeicao.recusar([], 0.5, 0.1) is False


def test_recusa_por_distancia():
    longe = [Falso(0.9), Falso(0.95)]
    perto = [Falso(0.1), Falso(0.9)]

    assert rejeicao.recusar(longe, 0.5, None) is True
    assert rejeicao.recusar(perto, 0.5, None) is False


def test_recusa_por_empate_mesmo_estando_perto():
    """É o caso que a distância não pega: parece com tudo, e por isso com nada."""
    empatado = [Falso(0.1), Falso(0.11)]

    assert rejeicao.recusar(empatado, 0.5, None) is False
    assert rejeicao.recusar(empatado, 0.5, 0.05) is True


def test_qualquer_um_dos_dois_basta_para_recusar():
    longe_e_folgado = [Falso(0.9), Falso(0.99)]
    assert rejeicao.recusar(longe_e_folgado, 0.5, 0.05) is True


def test_limiar_nulo_desliga_o_seu_criterio():
    qualquer = [Falso(0.9), Falso(0.91)]
    assert rejeicao.recusar(qualquer, None, None) is False


def test_candidato_unico_nao_e_recusado_por_margem():
    """Sem segundo colocado não há empate a medir, e inventar um seria recusar
    justamente a busca mais decidida que existe."""
    assert rejeicao.recusar([Falso(0.1)], None, 0.5) is False
