from libras.speller import Soletrador


def test_comeca_vazio():
    assert Soletrador().texto == ""


def test_acumula_letras():
    s = Soletrador()
    for letra in "OLA":
        s.adicionar(letra)
    assert s.texto == "OLA"


def test_ausencia_curta_nao_insere_espaco():
    s = Soletrador(segundos_para_espaco=1.5)
    s.adicionar("A")
    assert s.registrar_ausencia(1.0) is False
    assert s.texto == "A"


def test_ausencia_longa_insere_espaco():
    s = Soletrador(segundos_para_espaco=1.5)
    s.adicionar("A")
    s.registrar_ausencia(1.0)
    assert s.registrar_ausencia(0.6) is True
    assert s.texto == "A "


def test_ausencia_prolongada_insere_um_unico_espaco():
    s = Soletrador(segundos_para_espaco=1.5)
    s.adicionar("A")
    for _ in range(100):
        s.registrar_ausencia(0.1)
    assert s.texto == "A "


def test_nao_insere_espaco_no_inicio():
    s = Soletrador(segundos_para_espaco=1.5)
    assert s.registrar_ausencia(5.0) is False
    assert s.texto == ""


def test_presenca_reinicia_o_contador():
    s = Soletrador(segundos_para_espaco=1.5)
    s.adicionar("A")
    s.registrar_ausencia(1.4)
    s.registrar_presenca()
    assert s.registrar_ausencia(1.4) is False
    assert s.texto == "A"


def test_nova_ausencia_apos_letra_insere_outro_espaco():
    s = Soletrador(segundos_para_espaco=1.0)
    s.adicionar("A")
    s.registrar_ausencia(1.0)
    s.adicionar("B")
    s.registrar_ausencia(1.0)
    assert s.texto == "A B "


def test_palavra_atual_ignora_o_que_veio_antes_do_espaco():
    s = Soletrador(segundos_para_espaco=1.0)
    for letra in "OI":
        s.adicionar(letra)
    s.registrar_ausencia(1.0)
    s.adicionar("A")
    assert s.texto == "OI A"
    assert s.palavra_atual == "A"


def test_apagar_remove_o_ultimo():
    s = Soletrador()
    for letra in "OLA":
        s.adicionar(letra)
    s.apagar()
    assert s.texto == "OL"


def test_apagar_vazio_nao_quebra():
    s = Soletrador()
    s.apagar()
    assert s.texto == ""


def test_limpar_zera_tudo():
    s = Soletrador()
    for letra in "OLA":
        s.adicionar(letra)
    s.limpar()
    assert s.texto == ""
