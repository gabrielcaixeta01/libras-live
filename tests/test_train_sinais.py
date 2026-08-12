"""O protocolo do treino do encoder — a parte que decide se o número é honesto.

O modelo em si é testado em `test_encoder.py`. Aqui o que está sob teste é o
rodízio: quem entra no treino, quem fica de fora, e se o que ficou de fora
ficou de fora *de verdade*. Um vazamento aqui não daria erro nenhum — daria um
recall bonito e mentiroso, que é o mesmo erro do L↔G da fase 1 num lugar novo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import train_sinais  # noqa: E402

from libras import avaliacao, encoder, pose  # noqa: E402
from libras.dicionario import Dicionario  # noqa: E402

T = 8  # frames; o suficiente para a GRU e rápido o bastante para o teste
ARTICULADORES = ("articulador_1", "articulador_2", "articulador_3")


def amostras(sinais: int = 6) -> train_sinais.Amostras:
    """`sinais` palavras, três articuladores cada — a forma do V-LIBRASIL em miniatura.

    Cada sinal tem uma trajetória própria e cada articulador a faz com um desvio
    pequeno, que é a estrutura que o encoder existe para aprender.
    """
    rng = np.random.default_rng(0)
    vetores, rotulos, fontes = [], [], []

    for s in range(sinais):
        forma = rng.normal(0.0, 0.5, size=(T, pose.TAMANHO_VETOR)).astype(np.float32)
        for articulador in ARTICULADORES:
            vetores.append(forma + rng.normal(0.0, 0.05, size=forma.shape).astype(np.float32))
            rotulos.append(f"SINAL{s}")
            fontes.append(articulador)

    return train_sinais.Amostras(np.stack(vetores), rotulos, fontes)


def hp_pequeno() -> encoder.Hiperparametros:
    return encoder.Hiperparametros(dimensao=16, oculto=8, camadas=1, dropout=0.0)


# --- carregar ---


def test_carrega_as_sequencias_alinhadas(tmp_path):
    dados = amostras(3)
    caminho = tmp_path / "dicionario.npz"
    Dicionario(dados.vetores, dados.rotulos, dados.fontes, metrica="dtw").salvar(caminho)

    lido = train_sinais.carregar_amostras(caminho)

    assert lido.vetores.shape == dados.vetores.shape
    assert lido.rotulos == dados.rotulos
    assert lido.fontes == dados.fontes


def test_recusa_o_dicionario_de_embeddings(tmp_path):
    """Treinar o encoder sobre a saída dele mesmo é um erro fácil de cometer com
    dois npz parecidos na mesma pasta, e sem esta guarda ele treinaria."""
    caminho = tmp_path / "embeddings.npz"
    Dicionario(
        np.zeros((2, 16), dtype=np.float32),
        ["CASA", "ÁGUA"],
        list(ARTICULADORES[:2]),
        metrica="cosseno",
    ).salvar(caminho)

    with pytest.raises(ValueError, match="cosseno"):
        train_sinais.carregar_amostras(caminho)


def test_chave_junta_o_que_o_acento_separa():
    dados = train_sinais.Amostras(
        np.zeros((2, T, pose.TAMANHO_VETOR), dtype=np.float32),
        ["AMANHÃ", "amanha"],
        list(ARTICULADORES[:2]),
    )

    assert dados.chaves == ["AMANHA", "AMANHA"]


# --- vocabulário aberto ---


def test_abertos_saem_espalhados_pelo_vocabulario():
    """Um bloco alfabético concentraria o teste numa vizinhança do vocabulário."""
    chaves = [f"SINAL{i:03d}" for i in range(100) for _ in range(2)]

    abertos = train_sinais.escolher_abertos(chaves, 10)

    assert len(abertos) == 10
    assert max(abertos) > "SINAL050"  # não são só os primeiros
    assert train_sinais.escolher_abertos(chaves, 10) == abertos  # reprodutível


def test_sinal_com_uma_gravacao_nao_serve_de_vocabulario_aberto():
    """Ele não pode estar no índice e na consulta ao mesmo tempo, então não
    mediria nada — e ocuparia uma vaga de quem mediria."""
    chaves = ["SOZINHO"] + [f"PAR{i}" for i in range(5) for _ in range(2)]

    abertos = train_sinais.escolher_abertos(chaves, 3)

    assert "SOZINHO" not in abertos


def test_zero_desliga_o_vocabulario_aberto():
    assert train_sinais.escolher_abertos(["A", "A", "B", "B"], 0) == set()


def test_vocabulario_aberto_maior_que_a_base_e_recusado():
    with pytest.raises(ValueError, match="nada para treinar"):
        train_sinais.escolher_abertos(["A", "A", "B", "B"], 5)


def test_o_treino_nao_ve_nenhuma_gravacao_do_vocabulario_aberto():
    """A promessa é "um sinal novo entra com uma gravação". Se o encoder tiver
    visto essa gravação no treino, a promessa não foi medida."""
    dados = amostras(6)
    abertos = {"SINAL0", "SINAL3"}

    vetores, alvos, classes = train_sinais._preparar_treino(
        dados, list(range(len(dados))), abertos
    )

    assert classes == 4
    assert len(vetores) == 4 * len(ARTICULADORES)
    assert set(alvos.tolist()) == {0, 1, 2, 3}  # índices contíguos


def test_o_treino_de_um_rodizio_nao_ve_o_articulador_de_fora():
    dados = amostras(4)
    dentro = [i for i, f in enumerate(dados.fontes) if f != "articulador_2"]

    vetores, _, _ = train_sinais._preparar_treino(dados, dentro, abertos=set())

    assert len(vetores) == 4 * 2


# --- recorte de vocabulário ---


def test_limitar_sinais_mantem_todas_as_gravacoes_dos_escolhidos():
    recortado = train_sinais.limitar_sinais(amostras(6), 2)

    assert set(recortado.chaves) == {"SINAL0", "SINAL1"}
    assert len(recortado) == 2 * len(ARTICULADORES)


def test_restringir_ao_nucleo_mantem_so_o_vocabulario_curado(monkeypatch):
    monkeypatch.setattr(train_sinais.nucleo, "CHAVES", frozenset({"SINAL1", "SINAL3"}))
    recortado = train_sinais.restringir_ao_nucleo(amostras(6))

    assert set(recortado.chaves) == {"SINAL1", "SINAL3"}
    assert len(recortado) == 2 * len(ARTICULADORES)


# --- avaliação ---


def test_avaliar_recorta_o_indice_ao_vocabulario_pedido():
    """O recorte vale para o índice e para as consultas: é o app como entregue."""
    dados = amostras(6)
    modelo = encoder.Codificador(hp_pequeno()).eval()

    rodizio = train_sinais.avaliar(
        modelo,
        dados,
        "articulador_1",
        abertos=set(),
        dispositivo=torch.device("cpu"),
        chaves_indice=frozenset({"SINAL0", "SINAL1"}),
    )

    assert len(rodizio.posicoes) == 2


def test_avaliar_junta_as_consultas_de_fora_do_indice_na_calibracao():
    """Os sinais que o índice não tem são a metade negativa do limiar.

    Sem eles o corte sairia de consultas que todas tinham resposta possível, e o
    limiar não pegaria justamente o caso que ele existe para pegar.
    """
    dados = amostras(6)
    modelo = encoder.Codificador(hp_pequeno()).eval()

    rodizio = train_sinais.avaliar(
        modelo,
        dados,
        "articulador_1",
        abertos=set(),
        dispositivo=torch.device("cpu"),
        chaves_indice=frozenset({"SINAL0", "SINAL1"}),
    )

    # 2 consultas de dentro + 4 sinais de fora, todos do articulador de teste.
    assert len(rodizio.confiancas) == 6
    assert sum(1 for _, certo in rodizio.confiancas if not certo) >= 4


def test_sem_recorte_nao_ha_consulta_de_fora():
    dados = amostras(4)
    modelo = encoder.Codificador(hp_pequeno()).eval()

    rodizio = train_sinais.avaliar(
        modelo, dados, "articulador_1", abertos=set(), dispositivo=torch.device("cpu")
    )

    assert len(rodizio.confiancas) == 4


def test_avaliar_consulta_com_o_articulador_que_ficou_fora():
    dados = amostras(5)
    modelo = encoder.Codificador(hp_pequeno()).eval()

    rodizio = train_sinais.avaliar(
        modelo, dados, "articulador_3", abertos=set(), dispositivo=torch.device("cpu")
    )

    assert rodizio.fonte == "articulador_3"
    assert len(rodizio.posicoes) == 5          # uma consulta por sinal
    assert rodizio.posicoes_abertas == []
    assert rodizio.metricas.consultas == 5


def test_as_metricas_abertas_contam_so_os_sinais_de_fora_do_treino():
    dados = amostras(6)
    modelo = encoder.Codificador(hp_pequeno()).eval()

    rodizio = train_sinais.avaliar(
        modelo,
        dados,
        "articulador_1",
        abertos={"SINAL0", "SINAL5"},
        dispositivo=torch.device("cpu"),
    )

    assert len(rodizio.posicoes) == 6
    assert len(rodizio.posicoes_abertas) == 2


def test_o_rodizio_gira_por_todos_os_articuladores():
    """Três treinos, um por pessoa de fora. É o que custa o número honesto."""
    dados = amostras(4)

    rodizios = train_sinais.rodiziar(
        dados,
        abertos=set(),
        hp=hp_pequeno(),
        dispositivo=torch.device("cpu"),
        epocas=2,
        lote=8,
    )

    assert [r.fonte for r in rodizios] == list(ARTICULADORES)
    assert all(r.metricas.consultas == 4 for r in rodizios)


def test_treinar_aproxima_o_mesmo_sinal_de_pessoas_diferentes():
    """O teste do objetivo, não da perda: depois do treino, duas gravações do
    mesmo sinal têm que ficar mais parecidas do que duas de sinais diferentes.
    É a razão 0,80 da baseline sendo atacada."""
    dados = amostras(8)
    vetores, alvos, classes = train_sinais._preparar_treino(
        dados, list(range(len(dados))), abertos=set()
    )

    modelo = train_sinais.treinar(
        vetores,
        alvos,
        classes,
        hp_pequeno(),
        torch.device("cpu"),
        epocas=30,
        lote=8,
        aumentar=False,
    )

    embeddings = encoder.codificar(modelo, vetores)
    chaves = np.array(alvos)
    iguais = [
        float(embeddings[i] @ embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
        if chaves[i] == chaves[j]
    ]
    diferentes = [
        float(embeddings[i] @ embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
        if chaves[i] != chaves[j]
    ]

    assert np.mean(iguais) > np.mean(diferentes)


# --- limiar de rejeição ---


def test_o_limiar_separa_o_que_tem_resposta_do_que_nao_tem():
    """Distâncias baixas acertam, altas não: o corte tem que cair no meio."""
    confiancas = [(0.1, True), (0.2, True), (0.8, False), (0.9, False)]
    corte = train_sinais.calibrar_rejeicao(confiancas)

    assert 0.2 <= corte.limiar < 0.8
    assert corte.aceitos_certos == 2
    assert corte.aceitos_errados == 0
    assert corte.precisao == 1.0
    assert corte.cobertura == 1.0
    assert corte.recusa_correta == 1.0


def test_o_corte_preserva_a_cobertura_pedida():
    """A regra é um quantil dos acertos, e é isso que o app promete manter."""
    confiancas = [(i / 100, True) for i in range(100)]
    corte = train_sinais.calibrar_rejeicao(confiancas, cobertura_minima=0.90)

    assert corte.cobertura >= 0.90
    assert corte.limiar == pytest.approx(0.89)


def test_cobertura_mais_alta_nao_corta_menos():
    """Pedir mais acertos preservados só pode afrouxar o corte."""
    confiancas = [(0.1, True), (0.5, True), (0.6, False), (0.9, True)]

    frouxo = train_sinais.calibrar_rejeicao(confiancas, cobertura_minima=1.0)
    apertado = train_sinais.calibrar_rejeicao(confiancas, cobertura_minima=0.5)

    assert frouxo.limiar >= apertado.limiar
    assert frouxo.cobertura == 1.0


def test_o_limiar_conta_o_que_ele_custa():
    """Nenhum corte separa tudo, e o relatório precisa dizer o que se perde."""
    confiancas = [(0.1, True), (0.5, False), (0.6, True), (0.9, False)]
    corte = train_sinais.calibrar_rejeicao(confiancas)

    assert corte.aceitos_certos + corte.recusados_certos == 2
    assert corte.aceitos_errados + corte.recusados_errados == 2
    assert 0.0 <= corte.precisao <= 1.0


def test_sem_acerto_nenhum_nao_ha_o_que_calibrar():
    """O corte é um quantil dos acertos; sem acertos ele sairia do acaso."""
    assert train_sinais.calibrar_rejeicao([(0.1, False)]) is None
    assert train_sinais.calibrar_rejeicao([]) is None


def test_so_acertos_aceita_tudo():
    """Sem consulta errada nenhuma, o limiar não tem o que recusar."""
    corte = train_sinais.calibrar_rejeicao([(0.1, True), (0.2, True)])

    assert corte.aceitos_certos == 2
    assert corte.recusados_errados == 0


def test_cobertura_invalida_e_recusada():
    with pytest.raises(ValueError):
        train_sinais.calibrar_rejeicao([(0.1, True)], cobertura_minima=0.0)


# --- veredito e relatório ---


def metricas(recall_5: float) -> avaliacao.Metricas:
    return avaliacao.Metricas(consultas=100, recall_1=0.0, recall_5=recall_5, mrr=0.0)


def test_veredito_recusa_o_torch_quando_nao_bate_a_baseline():
    assert "NÃO ENTRA" in train_sinais._veredito(metricas(0.05), limite_sinais=None)


def test_veredito_devolve_a_decisao_quando_o_ganho_e_pequeno():
    assert "DECISÃO DO DONO" in train_sinais._veredito(metricas(0.09), limite_sinais=None)


def test_veredito_aprova_o_torch_com_margem_clara():
    assert "TORCH ENTRA" in train_sinais._veredito(metricas(0.40), limite_sinais=None)


def test_vocabulario_recortado_nao_tem_veredito():
    """Menos candidatos é menos confusão: o número sobe sem o modelo melhorar."""
    assert "SEM VEREDITO" in train_sinais._veredito(metricas(0.90), limite_sinais=40)


def test_o_relatorio_traz_o_total_e_a_comparacao_com_a_baseline():
    dados = amostras(4)
    rodizios = [
        train_sinais.Rodizio(fonte, [1, 2, None, 5], [1]) for fonte in ARTICULADORES
    ]

    relatorio = train_sinais.montar_relatorio(
        dados,
        rodizios,
        hp_pequeno(),
        abertos={"SINAL0"},
        epocas=10,
        dispositivo=torch.device("cpu"),
        segundos=61.0,
        limite_sinais=None,
    )

    assert "TOTAL" in relatorio
    assert "VOCABULÁRIO ABERTO" in relatorio
    assert "baseline DTW" in relatorio
    assert "12 consultas" in relatorio  # 4 posições x 3 rodízios
