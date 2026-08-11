import pytest

from libras.practice import Pratica

LETRAS = list("ABCDE")


def pratica(rodadas: int = 5, letras=None, semente: int = 0) -> Pratica:
    return Pratica(letras or LETRAS, rodadas=rodadas, semente=semente)


# --- construção ---


def test_rejeita_lista_de_letras_vazia():
    with pytest.raises(ValueError):
        Pratica([], rodadas=3)


def test_rejeita_numero_de_rodadas_invalido():
    with pytest.raises(ValueError):
        Pratica(LETRAS, rodadas=0)


def test_alvo_sai_das_letras_conhecidas():
    sessao = pratica()
    assert sessao.alvo in LETRAS


def test_comeca_na_primeira_rodada():
    sessao = pratica(rodadas=5)
    assert sessao.rodada == 1
    assert sessao.total == 5
    assert not sessao.concluida


def test_e_deterministico_por_semente():
    a = [Pratica(LETRAS, rodadas=6, semente=9).alvo for _ in range(3)]
    assert len(set(a)) == 1


def test_nao_repete_a_mesma_letra_seguida():
    sessao = Pratica(LETRAS, rodadas=40, semente=3)
    vistas = []
    while not sessao.concluida:
        vistas.append(sessao.alvo)
        sessao.pular(1.0)

    assert all(a != b for a, b in zip(vistas, vistas[1:]))


def test_letra_unica_pode_repetir():
    """Com um alfabeto de uma letra só, repetir é a única opção."""
    sessao = Pratica(["A"], rodadas=3, semente=1)
    while not sessao.concluida:
        assert sessao.alvo == "A"
        sessao.pular(1.0)


# --- acerto e erro ---


def test_acerto_avanca_a_rodada():
    sessao = pratica(rodadas=3)
    assert sessao.registrar(sessao.alvo, 2.0) is True
    assert sessao.rodada == 2
    assert sessao.acertos == 1


def test_erro_nao_avanca_e_conta():
    sessao = pratica(rodadas=3)
    errada = next(l for l in LETRAS if l != sessao.alvo)
    alvo_antes = sessao.alvo

    assert sessao.registrar(errada, 2.0) is False
    assert sessao.rodada == 1
    assert sessao.alvo == alvo_antes
    assert sessao.erros == 1
    assert sessao.acertos == 0


def test_varios_erros_na_mesma_rodada():
    sessao = pratica(rodadas=3)
    errada = next(l for l in LETRAS if l != sessao.alvo)
    sessao.registrar(errada, 1.0)
    sessao.registrar(errada, 1.0)
    assert sessao.erros == 2
    assert sessao.rodada == 1


def test_pular_avanca_sem_contar_acerto():
    sessao = pratica(rodadas=3)
    sessao.pular(1.0)
    assert sessao.rodada == 2
    assert sessao.acertos == 0
    assert sessao.puladas == 1


# --- fim da sessão ---


def test_termina_depois_de_todas_as_rodadas():
    sessao = pratica(rodadas=3)
    for _ in range(3):
        sessao.registrar(sessao.alvo, 1.0)

    assert sessao.concluida
    assert sessao.acertos == 3
    assert sessao.alvo is None


def test_registrar_depois_do_fim_nao_muda_nada():
    sessao = pratica(rodadas=1)
    sessao.registrar(sessao.alvo, 1.0)

    assert sessao.registrar("A", 1.0) is False
    assert sessao.acertos == 1


def test_tempo_medio_considera_so_os_acertos():
    sessao = pratica(rodadas=2)
    sessao.registrar(sessao.alvo, 2.0)
    sessao.registrar(sessao.alvo, 4.0)
    assert sessao.tempo_medio == pytest.approx(3.0)


def test_tempo_medio_e_zero_sem_acertos():
    assert pratica(rodadas=2).tempo_medio == 0.0


def test_precisao_e_a_fracao_de_acertos_nas_tentativas():
    sessao = pratica(rodadas=2)
    errada = next(l for l in LETRAS if l != sessao.alvo)
    sessao.registrar(errada, 1.0)       # 1 erro
    sessao.registrar(sessao.alvo, 1.0)  # 1 acerto
    assert sessao.precisao == pytest.approx(0.5)


def test_precisao_e_zero_sem_tentativas():
    assert pratica().precisao == 0.0


def test_resumo_menciona_o_placar():
    sessao = pratica(rodadas=1)
    sessao.registrar(sessao.alvo, 1.5)
    resumo = sessao.resumo()
    assert "1/1" in resumo


def test_reiniciar_zera_o_placar_e_troca_a_sequencia():
    sessao = pratica(rodadas=3)
    sessao.registrar(sessao.alvo, 1.0)
    sessao.reiniciar()

    assert sessao.rodada == 1
    assert sessao.acertos == 0
    assert sessao.erros == 0
    assert not sessao.concluida
