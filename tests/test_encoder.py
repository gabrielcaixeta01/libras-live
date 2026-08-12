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


# --- dispositivo ---


def test_dispositivo_padrao_e_cpu_ou_mps():
    assert encoder.dispositivo_padrao().type in {"cpu", "mps"}
