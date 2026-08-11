import pytest

from libras.sinais import catalogo


# --- rótulo ---


@pytest.mark.parametrize(
    "arquivo, esperado",
    [
        ("casa.mp4", "CASA"),
        ("CASA.MP4", "CASA"),
        ("bom_dia.mp4", "BOM DIA"),
        ("bom-dia.mov", "BOM DIA"),
        ("pão.mp4", "PÃO"),
        ("  casa  .mp4", "CASA"),
    ],
)
def test_rotulo_vem_do_nome_do_arquivo(arquivo, esperado):
    assert catalogo.rotulo_do_arquivo(arquivo) == esperado


@pytest.mark.parametrize(
    "arquivo",
    ["casa_2.mp4", "casa-02.mp4", "casa (3).mp4", "casa_v2.mp4", "casa take 1.mp4"],
)
def test_sufixo_de_repeticao_nao_vira_sinal_novo(arquivo):
    """Sem isto, CASA e CASA_2 viram dois sinais e a avaliação despenca calada."""
    assert catalogo.rotulo_do_arquivo(arquivo) == "CASA"


def test_numero_que_e_o_sinal_inteiro_sobrevive():
    assert catalogo.rotulo_do_arquivo("cinco.mp4") == "CINCO"


def test_caminho_completo_usa_so_o_nome():
    assert catalogo.rotulo_do_arquivo("/dados/art1/videos/casa_2.mp4") == "CASA"


# --- chave de comparação ---


def test_chave_ignora_acento_e_caixa():
    assert catalogo.chave("AMANHÃ") == catalogo.chave("amanha")


def test_chave_nao_substitui_o_rotulo_exibido():
    """O acento some da comparação, não da tela."""
    assert catalogo.rotulo_do_arquivo("amanhã.mp4") == "AMANHÃ"


# --- articulador ---


def test_articulador_e_a_pasta_logo_abaixo_da_raiz(tmp_path):
    video = tmp_path / "articulador_1" / "casa.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    assert catalogo.articulador_do_caminho(video, tmp_path) == "articulador_1"


def test_pastas_mais_fundas_nao_confundem(tmp_path):
    video = tmp_path / "art2" / "lote3" / "casa.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    assert catalogo.articulador_do_caminho(video, tmp_path) == "art2"


def test_video_solto_na_raiz_fica_desconhecido(tmp_path):
    video = tmp_path / "casa.mp4"
    video.touch()
    assert catalogo.articulador_do_caminho(video, tmp_path) == "desconhecido"


def test_video_fora_da_raiz_fica_desconhecido(tmp_path):
    assert catalogo.articulador_do_caminho("/outro/lugar/casa.mp4", tmp_path) == (
        "desconhecido"
    )


# --- listagem ---


def test_lista_so_video_e_em_ordem_estavel(tmp_path):
    (tmp_path / "art1").mkdir()
    for nome in ("b.mp4", "a.mov", "leiame.txt", "c.MP4"):
        (tmp_path / "art1" / nome).touch()

    encontrados = catalogo.listar_videos(tmp_path)
    assert [p.name for p in encontrados] == ["a.mov", "b.mp4", "c.MP4"]
    assert encontrados == catalogo.listar_videos(tmp_path)


def test_diretorio_inexistente_reclama(tmp_path):
    with pytest.raises(FileNotFoundError):
        catalogo.listar_videos(tmp_path / "nao_existe")
