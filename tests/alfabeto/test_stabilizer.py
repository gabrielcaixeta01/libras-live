from libras.alfabeto.stabilizer import Estabilizador


def alimentar(est, letra, confianca, vezes):
    """Empurra N frames iguais e devolve tudo que foi confirmado."""
    return [r for _ in range(vezes) if (r := est.atualizar(letra, confianca))]


def test_nao_confirma_com_buffer_incompleto():
    est = Estabilizador(tamanho_buffer=10)
    assert alimentar(est, "A", 0.99, 9) == []


def test_confirma_quando_buffer_enche():
    est = Estabilizador(tamanho_buffer=10)
    assert alimentar(est, "A", 0.99, 10) == ["A"]


def test_confirma_apesar_de_ruido_dentro_do_limite():
    """1 frame errado em 10 não deve impedir a confirmação (dominância 0.8)."""
    est = Estabilizador(tamanho_buffer=10, dominancia_minima=0.8)
    est.atualizar("B", 0.9)
    assert alimentar(est, "A", 0.99, 9) == ["A"]


def test_nao_confirma_sem_dominancia():
    est = Estabilizador(tamanho_buffer=10, dominancia_minima=0.8)
    for i in range(10):
        assert est.atualizar("A" if i % 2 else "B", 0.99) is None


def test_nao_confirma_com_confianca_baixa():
    est = Estabilizador(tamanho_buffer=10, confianca_minima=0.7)
    assert alimentar(est, "A", 0.5, 10) == []


def test_letra_segurada_confirma_uma_unica_vez():
    """O bug clássico: manter a mão parada emitiria AAAAAA."""
    est = Estabilizador(tamanho_buffer=10)
    assert alimentar(est, "A", 0.99, 60) == ["A"]


def test_mesma_letra_de_novo_apos_trocar_de_estado():
    est = Estabilizador(tamanho_buffer=10)
    assert alimentar(est, "A", 0.99, 10) == ["A"]
    alimentar(est, "B", 0.99, 10)          # muda de estado
    assert alimentar(est, "A", 0.99, 10) == ["A"]


def test_mao_ausente_libera_a_letra_bloqueada():
    est = Estabilizador(tamanho_buffer=10)
    assert alimentar(est, "A", 0.99, 10) == ["A"]
    alimentar(est, None, 0.0, 10)          # tira a mão
    assert alimentar(est, "A", 0.99, 10) == ["A"]


def test_ausencia_de_mao_nunca_vira_letra():
    est = Estabilizador(tamanho_buffer=10)
    assert alimentar(est, None, 0.0, 30) == []


def test_sequencia_de_letras_diferentes():
    est = Estabilizador(tamanho_buffer=10)
    saida = []
    for letra in "OLA":
        saida += alimentar(est, letra, 0.99, 10)
    assert saida == ["O", "L", "A"]


def test_reiniciar_limpa_o_estado():
    est = Estabilizador(tamanho_buffer=10)
    alimentar(est, "A", 0.99, 10)
    est.reiniciar()
    assert est.preenchimento == 0.0
    assert alimentar(est, "A", 0.99, 10) == ["A"]
