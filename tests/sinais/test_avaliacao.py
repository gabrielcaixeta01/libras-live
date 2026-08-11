import numpy as np
import pytest

from libras.sinais import avaliacao
from libras.sinais.dicionario import Candidato, Dicionario


def candidatos(*rotulos: str) -> list[Candidato]:
    return [
        Candidato(rotulo=r, distancia=float(i), similaridade=0.5, fonte="x")
        for i, r in enumerate(rotulos)
    ]


# --- posição do acerto ---


def test_acerto_no_topo_e_posicao_um():
    assert avaliacao.posicao(candidatos("CASA", "PORTA"), "CASA") == 1


def test_acerto_no_meio_da_lista():
    assert avaliacao.posicao(candidatos("A", "B", "CASA", "D"), "CASA") == 3


def test_sinal_ausente_da_lista_nao_tem_posicao():
    assert avaliacao.posicao(candidatos("A", "B"), "CASA") is None


def test_acento_no_nome_do_arquivo_nao_conta_como_erro():
    assert avaliacao.posicao(candidatos("AMANHÃ"), "amanha") == 1


def test_lista_vazia():
    assert avaliacao.posicao([], "CASA") is None


# --- agregação ---


def test_placar_de_um_caso_conhecido():
    m = avaliacao.agregar([1, 2, None, 5])
    assert m.consultas == 4
    assert m.recall_1 == pytest.approx(0.25)
    assert m.recall_5 == pytest.approx(0.75)
    assert m.mrr == pytest.approx((1.0 + 0.5 + 0.0 + 0.2) / 4)


def test_consulta_que_falhou_conta_como_erro_e_nao_some():
    """Descartar o que falhou é a forma mais fácil de inventar número bonito."""
    assert avaliacao.agregar([1, None]).recall_5 == pytest.approx(0.5)


def test_acerto_na_sexta_posicao_nao_entra_no_recall_5():
    assert avaliacao.agregar([6]).recall_5 == 0.0
    assert avaliacao.agregar([6]).mrr == pytest.approx(1 / 6)


def test_sem_consultas_o_placar_e_zero():
    m = avaliacao.agregar([])
    assert m.consultas == 0 and m.mrr == 0.0


def test_placar_vira_texto_legivel():
    assert "recall@5" in str(avaliacao.agregar([1, 2]))


# --- leave-one-articulator-out ---


def traj(inclinacao: float, n: int = 12) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None].astype(np.float32)
    return np.concatenate([t, t * inclinacao], axis=1)


def dicionario_de_tres_articuladores() -> Dicionario:
    representacoes, rotulos, fontes = [], [], []
    for sinal, inclinacao in (("CASA", 0.0), ("PORTA", 3.0), ("LIVRO", 6.0)):
        for articulador, ruido in (("a1", 0.0), ("a2", 0.1), ("a3", -0.1)):
            representacoes.append(traj(inclinacao + ruido))
            rotulos.append(sinal)
            fontes.append(articulador)
    return Dicionario(np.stack(representacoes), rotulos, fontes)


def test_roda_um_rodizio_por_articulador_mais_o_total():
    resultado = avaliacao.leave_one_articulator_out(dicionario_de_tres_articuladores())
    assert set(resultado) == {"a1", "a2", "a3", "total"}


def test_cada_articulador_e_consultado_uma_vez_por_sinal():
    resultado = avaliacao.leave_one_articulator_out(dicionario_de_tres_articuladores())
    assert resultado["a1"].consultas == 3
    assert resultado["total"].consultas == 9


def test_sinais_bem_separados_sao_encontrados():
    resultado = avaliacao.leave_one_articulator_out(dicionario_de_tres_articuladores())
    assert resultado["total"].recall_1 == pytest.approx(1.0)


def test_o_articulador_avaliado_nao_esta_no_indice():
    """A garantia central do protocolo. Se ele estivesse, a distância seria zero
    e o placar mediria memorização em vez de generalização."""
    d = dicionario_de_tres_articuladores()
    base, consultas = d.separar_fonte("a2")

    assert "a2" not in base.fontes
    assert len(consultas) == 3
    assert len(base) == 6


def test_um_articulador_so_nao_permite_rodizio():
    d = Dicionario(np.stack([traj(0.0), traj(1.0)]), ["A", "B"], ["a1", "a1"])
    with pytest.raises(ValueError, match="articulador"):
        avaliacao.leave_one_articulator_out(d)


def test_k_menor_que_cinco_e_recusado():
    with pytest.raises(ValueError, match="k"):
        avaliacao.leave_one_articulator_out(dicionario_de_tres_articuladores(), k=3)
