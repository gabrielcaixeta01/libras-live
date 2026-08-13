"""O encoder: o contrato que o dicionário e o app dependem dele.

Nada aqui treina de verdade — o que se testa é o que quebraria em silêncio. Um
embedding sem norma 1 faria o cosseno ordenar por tamanho em vez de por forma;
um `Hiperparametros` perdido no `.pt` faria a entrada ser montada de um jeito no
treino e de outro no app, e o dicionário sairia inútil sem nenhum erro; um
aumento que espelha trocaria a palavra e treinaria a rede no rótulo errado.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from libras import encoder, pose, sequencia  # noqa: E402


def sequencias(n: int = 4, t: int = sequencia.T_PADRAO) -> np.ndarray:
    """Lote (n, t, 147) com trajetórias diferentes entre si, e reprodutível."""
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 0.3, size=(n, t, pose.TAMANHO_VETOR)).astype(np.float32)


def modelo_pequeno(**kwargs) -> encoder.Codificador:
    """Menor que o de produção, pelo mesmo motivo de sempre: o teste roda em 1s."""
    hp = encoder.Hiperparametros(
        **{**dict(dimensao=16, oculto=8, camadas=1, dropout=0.0), **kwargs}
    )
    return encoder.Codificador(hp).eval()


# --- o embedding ---


def test_embedding_sai_na_esfera_unitaria():
    """A busca é por cosseno. Sem norma 1, um protótipo de norma grande ganharia
    consultas por tamanho, não por forma do sinal."""
    embeddings = encoder.codificar(modelo_pequeno(), sequencias())

    assert embeddings.shape == (4, 16)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_uma_sequencia_devolve_um_vetor_e_bate_com_o_lote():
    modelo = modelo_pequeno()
    lote = sequencias()

    solo = encoder.codificar(modelo, lote[0])

    assert solo.shape == (16,)
    assert np.allclose(solo, encoder.codificar(modelo, lote)[0], atol=1e-5)


def test_codificar_e_deterministico():
    """Com dropout ligado a mesma gravação daria embeddings diferentes a cada
    chamada, e a re-ancoragem viraria loteria."""
    modelo = encoder.Codificador(
        encoder.Hiperparametros(dimensao=16, oculto=8, camadas=1, dropout=0.5)
    )
    lote = sequencias()

    assert np.allclose(
        encoder.codificar(modelo, lote), encoder.codificar(modelo, lote), atol=1e-6
    )


def test_codificar_devolve_o_modelo_ao_modo_em_que_estava():
    """Chamar a busca no meio do treino não pode desligar o dropout do treino."""
    modelo = modelo_pequeno()
    modelo.train()

    encoder.codificar(modelo, sequencias())

    assert modelo.training


def test_lote_pequeno_e_lote_grande_dao_o_mesmo():
    modelo = modelo_pequeno()
    lote = sequencias(n=5)

    assert np.allclose(
        encoder.codificar(modelo, lote, lote=2),
        encoder.codificar(modelo, lote, lote=64),
        atol=1e-5,
    )


def test_entrada_dobra_de_largura_com_a_velocidade():
    assert encoder.Hiperparametros().dim_entrada == 2 * pose.TAMANHO_VETOR
    assert encoder.Hiperparametros(com_velocidade=False).dim_entrada == 147

    hp = encoder.Hiperparametros(com_velocidade=False)
    assert encoder.entrada_do_modelo(sequencias(), hp).shape[-1] == 147


def test_os_canais_de_mao_entram_na_largura_declarada():
    """`dim_entrada` é o que dimensiona a primeira camada. Se ele mentir sobre o
    que `entrada_do_modelo` produz, a rede nem constrói."""
    hp = encoder.Hiperparametros(com_maos=True)

    assert hp.dim_entrada == 2 * pose.TAMANHO_VETOR + sequencia.TAMANHO_MAOS_LOCAIS
    assert encoder.entrada_do_modelo(sequencias(), hp).shape[-1] == hp.dim_entrada


def test_um_pt_antigo_carrega_sem_os_canais_de_mao(tmp_path):
    """`Hiperparametros(**dados)` de um `.pt` gravado antes do campo tem que dar
    o modelo de antes, e não um com a entrada maior que os pesos."""
    assert encoder.Hiperparametros().com_maos is False

    antigo = {"dimensao": 16, "oculto": 8, "camadas": 1, "dropout": 0.0}
    assert encoder.Hiperparametros(**antigo).dim_entrada == 2 * pose.TAMANHO_VETOR


def test_modelo_com_maos_faz_ida_e_volta(tmp_path):
    modelo = modelo_pequeno(com_maos=True)
    lote = sequencias()
    antes = encoder.codificar(modelo, lote)

    caminho = tmp_path / "encoder.pt"
    encoder.salvar(modelo, caminho)
    recarregado = encoder.carregar(caminho, torch.device("cpu"))

    assert recarregado.hp.com_maos is True
    assert np.allclose(encoder.codificar(recarregado, lote), antes, atol=1e-5)


# --- persistência ---


def test_ida_e_volta_preserva_o_embedding_e_a_arquitetura(tmp_path):
    """O `.pt` carrega os hiperparâmetros junto porque um `com_velocidade`
    divergente entre treino e app não daria erro — daria um dicionário
    silenciosamente inútil."""
    modelo = modelo_pequeno(z=True, com_velocidade=False)
    lote = sequencias()
    antes = encoder.codificar(modelo, lote)

    caminho = tmp_path / "encoder.pt"
    encoder.salvar(modelo, caminho)
    recarregado = encoder.carregar(caminho, torch.device("cpu"))

    assert recarregado.hp == modelo.hp
    assert np.allclose(encoder.codificar(recarregado, lote), antes, atol=1e-5)


def test_modelo_carregado_vem_em_modo_de_avaliacao(tmp_path):
    caminho = tmp_path / "encoder.pt"
    encoder.salvar(modelo_pequeno(), caminho)

    assert not encoder.carregar(caminho, torch.device("cpu")).training


# --- ArcFace ---


def test_arcface_devolve_um_logit_por_classe():
    cabeca = encoder.ArcFace(dimensao=16, classes=5)
    embeddings = torch.nn.functional.normalize(torch.randn(3, 16), dim=1)

    logits = cabeca(embeddings, torch.tensor([0, 1, 2]), margem=0.3)

    assert logits.shape == (3, 5)


def test_a_margem_desconta_do_logit_da_classe_certa():
    """É o que a margem *é*: exigir que o embedding fique com folga dentro do seu
    setor. Sem esse desconto, ArcFace seria um softmax comum."""
    torch.manual_seed(0)
    cabeca = encoder.ArcFace(dimensao=16, classes=5)
    embeddings = torch.nn.functional.normalize(torch.randn(3, 16), dim=1)
    alvos = torch.tensor([0, 1, 2])

    sem = cabeca(embeddings, alvos, margem=0.0)
    com = cabeca(embeddings, alvos, margem=0.4)

    certos = torch.arange(3)
    assert (com[certos, alvos] < sem[certos, alvos]).all()
    # As outras classes não são tocadas.
    assert torch.allclose(com[:, 4], sem[:, 4], atol=1e-5)


def test_arcface_recusa_uma_classe_so():
    with pytest.raises(ValueError, match="2 classes"):
        encoder.ArcFace(dimensao=16, classes=1)


# --- aumento de dados ---


def test_aumentar_preserva_o_formato():
    rng = np.random.default_rng(0)
    original = sequencias(n=1)[0]

    variado = encoder.aumentar(original, rng)

    assert variado.shape == original.shape
    assert not np.allclose(variado, original)


def test_mesma_semente_mesma_variacao():
    original = sequencias(n=1)[0]

    a = encoder.aumentar(original, np.random.default_rng(3))
    b = encoder.aumentar(original, np.random.default_rng(3))

    assert np.allclose(a, b)


def test_a_deformacao_do_tempo_mantem_o_comeco_e_o_fim():
    """Um sinal é delimitado pelas suas pontas — deslocá-las seria recortar o
    gesto, não variá-lo. Mesma razão pela qual `reamostrar` as preserva."""
    original = sequencias(n=1)[0]

    variado = encoder.aumentar(
        original, np.random.default_rng(1), rotacao_graus=0.0, escala=0.0, ruido=0.0
    )

    assert np.allclose(variado[0], original[0], atol=1e-5)
    assert np.allclose(variado[-1], original[-1], atol=1e-5)
    assert not np.allclose(variado[len(variado) // 2], original[len(original) // 2])


def test_o_aumento_nao_espelha_o_sinalizante():
    """No alfabeto trocar de mão dá a mesma letra. Aqui o lado é fonema: espelhar
    geraria um sinal diferente com o rótulo antigo."""
    rng = np.random.default_rng(0)
    pontos = np.zeros((8, pose.NUM_PONTOS, 3), dtype=np.float32)
    pontos[:, pose.MAO_ESQUERDA, 0] = -0.8   # mão esquerda à esquerda do peito
    pontos[:, pose.MAO_DIREITA, 0] = 0.8

    for _ in range(20):
        variado = encoder.aumentar(pontos.reshape(8, -1), rng).reshape(8, 49, 3)
        assert variado[:, pose.MAO_ESQUERDA, 0].mean() < 0
        assert variado[:, pose.MAO_DIREITA, 0].mean() > 0


def test_a_oclusao_apaga_uma_mao_e_deixa_a_outra():
    """A mão que o MediaPipe perde some por um trecho, não a gravação inteira e
    nunca as duas — se as duas somem não sobra sinal nenhum para aprender."""
    rng = np.random.default_rng(3)
    pontos = np.zeros((16, pose.NUM_PONTOS, 3), dtype=np.float32)
    pontos[:, pose.MAO_ESQUERDA, 0] = np.linspace(0, 1, 16)[:, None]
    pontos[:, pose.MAO_DIREITA, 1] = np.linspace(0, 1, 16)[:, None]
    plano = pontos.reshape(16, -1)

    mudou_esquerda = mudou_direita = 0
    for _ in range(60):
        variado = encoder.aumentar(
            plano, rng, rotacao_graus=0, escala=0, ruido=0, tempo=0, oclusao=1.0
        ).reshape(16, 49, 3)
        e = not np.allclose(variado[:, pose.MAO_ESQUERDA], pontos[:, pose.MAO_ESQUERDA])
        d = not np.allclose(variado[:, pose.MAO_DIREITA], pontos[:, pose.MAO_DIREITA])
        assert not (e and d), "uma mão por vez"
        mudou_esquerda += e
        mudou_direita += d

    assert mudou_esquerda and mudou_direita


def test_sem_oclusao_o_aumento_nao_toca_nas_maos():
    rng = np.random.default_rng(3)
    plano = sequencias(1)[0]

    for _ in range(20):
        variado = encoder.aumentar(
            plano, rng, rotacao_graus=0, escala=0, ruido=0, tempo=0, oclusao=0.0
        )
        assert np.allclose(variado, plano, atol=1e-5)


def test_a_oclusao_preenche_o_buraco_entre_as_bordas_em_vez_de_zerar():
    """Zerar ensinaria um padrão que a rede nunca vai ver: o que chega até ela
    sempre passou por `sequencia.imputar`."""
    rng = np.random.default_rng(1)
    # Mão parada e longe de zero: apagá-la tem que devolver ela mesma — a
    # interpolação entre duas bordas iguais é a borda —, e zerá-la apareceria.
    plano = np.full((12, pose.TAMANHO_VETOR), 1.0, dtype=np.float32)

    for _ in range(30):
        variado = encoder.aumentar(
            plano, rng, rotacao_graus=0, escala=0, ruido=0, tempo=0, oclusao=1.0
        )
        assert np.allclose(variado, 1.0, atol=1e-5)


# --- média de codificações ---


def test_a_media_de_codificacoes_volta_para_a_esfera():
    modelo = modelo_pequeno()
    lote = sequencias()

    media = encoder.codificar_medio(modelo, lote, repeticoes=4)

    assert media.shape == (len(lote), modelo.hp.dimensao)
    assert np.allclose(np.linalg.norm(media, axis=1), 1.0, atol=1e-5)


def test_uma_repeticao_e_a_codificacao_crua():
    modelo = modelo_pequeno()
    lote = sequencias()

    assert np.allclose(
        encoder.codificar_medio(modelo, lote, repeticoes=1),
        encoder.codificar(modelo, lote),
    )


def test_a_media_e_reprodutivel_e_aceita_uma_sequencia_so():
    modelo = modelo_pequeno()
    uma = sequencias(1)[0]

    primeira = encoder.codificar_medio(modelo, uma, repeticoes=3, semente=5)
    segunda = encoder.codificar_medio(modelo, uma, repeticoes=3, semente=5)

    assert primeira.shape == (modelo.hp.dimensao,)
    assert np.allclose(primeira, segunda)


# --- dispositivo ---


def test_dispositivo_padrao_e_cpu_ou_mps():
    assert encoder.dispositivo_padrao().type in {"cpu", "mps"}


# --- a máscara de validade atravessando o aumento ---


def sequencia_com_mascara(t: int = 16) -> np.ndarray:
    """(T, 196) com tudo medido: a máscara só pode piorar a partir daqui."""
    geometria = np.full((t, pose.TAMANHO_VETOR), 0.3, dtype=np.float32)
    mascara = np.ones((t, sequencia.TAMANHO_VALIDADE), dtype=np.float32)
    return np.concatenate([geometria, mascara], axis=-1, dtype=np.float32)


def test_o_aumento_preserva_a_largura_com_mascara():
    rng = np.random.default_rng(0)
    entrada = sequencia_com_mascara()

    variado = encoder.aumentar(entrada, rng)

    assert variado.shape == entrada.shape


def test_a_oclusao_marca_como_nao_medida_a_mao_que_ela_apagou():
    """Se a máscara continuasse dizendo 'medi', o aumento estaria ensinando à
    rede exatamente a mentira que a máscara existe para desmentir."""
    rng = np.random.default_rng(7)
    entrada = sequencia_com_mascara()

    apagou_alguma = False
    for _ in range(40):
        variado = encoder.aumentar(
            entrada, rng, rotacao_graus=0, escala=0, ruido=0, tempo=0, oclusao=1.0
        )
        mascara = variado[:, pose.TAMANHO_VETOR :]
        if mascara.min() < 1.0:
            apagou_alguma = True
            # Some uma mão inteira por vez, nunca o corpo.
            assert np.allclose(mascara[:, pose.POSE], 1.0)

    assert apagou_alguma, "com oclusao=1.0 alguma mão tinha que ter sido apagada"


def test_sem_oclusao_a_mascara_atravessa_intacta():
    rng = np.random.default_rng(2)
    entrada = sequencia_com_mascara()

    variado = encoder.aumentar(
        entrada, rng, rotacao_graus=0, escala=0, ruido=0, tempo=0, oclusao=0.0
    )

    assert np.allclose(variado[:, pose.TAMANHO_VETOR :], 1.0)


def test_a_deformacao_temporal_move_geometria_e_mascara_pela_mesma_grade():
    """Uma máscara reamostrada por outra grade descreveria outro instante do
    sinal — pior que máscara nenhuma, porque a rede confiaria nela."""
    rng = np.random.default_rng(4)
    t = 16
    rampa = np.linspace(0.0, 1.0, t, dtype=np.float32)[:, None]
    entrada = np.concatenate(
        [
            np.repeat(rampa, pose.TAMANHO_VETOR, axis=1),
            np.repeat(rampa, sequencia.TAMANHO_VALIDADE, axis=1),
        ],
        axis=-1,
        dtype=np.float32,
    )

    variado = encoder.aumentar(
        entrada, rng, rotacao_graus=0, escala=0, ruido=0, tempo=0.4, oclusao=0.0
    )

    geometria = variado[:, 0]
    mascara = variado[:, pose.TAMANHO_VETOR]
    assert np.allclose(geometria, mascara, atol=1e-5)


def test_com_validade_muda_a_dimensao_de_entrada_do_modelo():
    hp = encoder.Hiperparametros(com_validade=True)
    sem = encoder.Hiperparametros(com_validade=False)

    assert hp.dim_entrada == sem.dim_entrada + sequencia.NUM_GRUPOS_VALIDADE


def test_com_validade_sobrevive_ao_salvar_e_carregar(tmp_path):
    """Um `com_validade` que se perdesse no `.pt` montaria a entrada de um jeito
    no treino e de outro no app, sem erro nenhum e com embedding sem sentido."""
    modelo = encoder.Codificador(
        encoder.Hiperparametros(
            dimensao=16, oculto=8, camadas=1, dropout=0.0, com_validade=True
        )
    )
    caminho = tmp_path / "encoder.pt"

    encoder.salvar(modelo, caminho)
    devolta = encoder.carregar(caminho)

    assert devolta.hp.com_validade is True
    assert devolta.hp.dim_entrada == modelo.hp.dim_entrada


def test_um_pt_antigo_carrega_sem_validade():
    """Backward-compat do campo novo: o padrão tem que descrever o que os
    modelos já gravados fazem, e eles não viram máscara nenhuma."""
    hp = encoder.Hiperparametros()

    assert hp.com_validade is False
