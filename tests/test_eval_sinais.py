"""O relatório da baseline, e a armadilha que ele precisa denunciar.

Existem dois npz parecidos em `data/sinais/`: o de sequências e o de embeddings.
Apontar este script para o segundo roda sem erro e imprime um número alto —
porque o encoder salvo viu os três articuladores, e o rodízio deixa de separar
pessoas. É o erro do L↔G da fase 1 com roupa nova, e a única defesa é o aviso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import eval_sinais  # noqa: E402

from libras import avaliacao  # noqa: E402
from libras.dicionario import Dicionario  # noqa: E402


def resultados() -> dict[str, avaliacao.Metricas]:
    placar = avaliacao.agregar([1, 3, None])
    return {"articulador_1": placar, "total": placar}


def dicionario(metrica: str) -> Dicionario:
    formato = (2, 8, 147) if metrica == "dtw" else (2, 16)
    return Dicionario(
        representacoes=np.zeros(formato, dtype=np.float32),
        rotulos=["CASA", "ÁGUA"],
        fontes=["articulador_1", "articulador_2"],
        metrica=metrica,
    )


def test_o_relatorio_da_baseline_nao_traz_aviso():
    relatorio = eval_sinais.montar_relatorio(dicionario("dtw"), resultados(), 12.0)

    assert "ATENÇÃO" not in relatorio
    assert "métrica de busca ......... dtw" in relatorio


def test_apontar_para_os_embeddings_avisa_que_o_numero_nao_e_honesto():
    relatorio = eval_sinais.montar_relatorio(dicionario("cosseno"), resultados(), 12.0)

    assert "ATENÇÃO" in relatorio
    assert "relatorio_encoder.txt" in relatorio


def test_o_veredito_muda_com_o_recall():
    ruim = avaliacao.Metricas(consultas=10, recall_1=0.0, recall_5=0.075, mrr=0.04)
    bom = avaliacao.Metricas(consultas=10, recall_1=0.5, recall_5=0.85, mrr=0.6)

    assert "não sustenta o produto" in eval_sinais._veredito_sobre_o_encoder(ruim)
    assert "já entrega o produto" in eval_sinais._veredito_sobre_o_encoder(bom)
