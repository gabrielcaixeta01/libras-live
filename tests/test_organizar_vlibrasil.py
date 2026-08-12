"""A tradução do layout do espelho para o layout que o catálogo lê.

Esta é a peça que decide se a avaliação é honesta. O espelho do Kaggle entrega
os 4.086 vídeos numa pasta só, com o articulador no nome do arquivo; o
`catalogo` lê o articulador da pasta. Se esta tradução errar em silêncio, todo
vídeo vira `desconhecido`, o leave-one-articulator-out deixa de separar pessoas
e o recall@5 passa a medir memorização — sem que nada pareça quebrado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import organizar_vlibrasil  # noqa: E402

from libras import catalogo  # noqa: E402

PREFIXO = "videos UFPE (V-LIBRASIL)/data"


@pytest.mark.parametrize(
    "membro, esperado",
    [
        (f"{PREFIXO}/Abacaxi_Articulador1.mp4", "articulador_1/Abacaxi.mp4"),
        (f"{PREFIXO}/Abacaxi_Articulador3.mp4", "articulador_3/Abacaxi.mp4"),
        # Acento é conteúdo, não ruído: é ele que aparece na tela.
        (f"{PREFIXO}/Abençoar_Articulador2.mp4", "articulador_2/Abençoar.mp4"),
        # Nome com espaços e com hífen sobrevivem inteiros.
        (f"{PREFIXO}/À noite toda_Articulador1.mp4", "articulador_1/À noite toda.mp4"),
        (f"{PREFIXO}/Quarta-feira_Articulador2.mp4", "articulador_2/Quarta-feira.mp4"),
    ],
)
def test_traduz_nome_plano_para_pasta_por_articulador(membro: str, esperado: str):
    assert organizar_vlibrasil.destino_do_membro(membro) == Path(esperado)


@pytest.mark.parametrize(
    "membro",
    [
        f"{PREFIXO}/../annotations.csv",
        "videos UFPE (V-LIBRASIL)/annotations.csv",
        f"{PREFIXO}/SemArticulador.mp4",
        f"{PREFIXO}/",
    ],
)
def test_ignora_o_que_nao_e_video_no_padrao(membro: str):
    assert organizar_vlibrasil.destino_do_membro(membro) is None


def test_o_destino_devolve_o_articulador_que_o_catalogo_le():
    """O contrato de ponta a ponta: pasta escrita aqui, lida lá."""
    raiz = Path("/tmp/v-librasil")
    relativo = organizar_vlibrasil.destino_do_membro(
        f"{PREFIXO}/Abacaxi_Articulador2.mp4"
    )

    assert catalogo.articulador_do_caminho(raiz / relativo, raiz) == "articulador_2"


def test_o_destino_devolve_o_rotulo_sem_o_sufixo_do_articulador():
    """`Abacaxi_Articulador1` não pode virar o sinal "ABACAXI ARTICULADOR1"."""
    relativo = organizar_vlibrasil.destino_do_membro(
        f"{PREFIXO}/Abacaxi_Articulador1.mp4"
    )

    assert catalogo.rotulo_do_arquivo(relativo) == "ABACAXI"
