"""Mede o dicionário de sinais com leave-one-articulator-out.

    python training/eval_sinais.py
    python training/eval_sinais.py --nucleo

Indexa com dois articuladores, consulta com o terceiro, três vezes. É o único
protocolo honesto com uma base de três pessoas: qualquer divisão aleatória
deixaria o mesmo articulador dos dois lados, e o placar mediria memorização de
pessoa — o mesmo erro que a fase 1 cometeu com o L↔G.

O relatório vai para `models/relatorio_sinais.txt` e é ele que decide se o
encoder neural vale o torch: a baseline DTW registra aqui o número que o
encoder precisa bater.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libras import avaliacao, catalogo, config, nucleo  # noqa: E402
from libras.dicionario import Dicionario  # noqa: E402


def montar_relatorio(
    dicionario: Dicionario,
    resultados: dict[str, avaliacao.Metricas],
    segundos: float,
) -> str:
    por_sinal = Counter(catalogo.chave(r) for r in dicionario.rotulos)
    linhas = [
        "=" * 72,
        "DICIONÁRIO REVERSO DE SINAIS — leave-one-articulator-out",
        "=" * 72,
        "",
        f"métrica de busca ......... {dicionario.metrica}",
        f"protótipos ............... {len(dicionario)}",
        f"sinais distintos ......... {len(por_sinal)}",
        f"gravações por sinal ...... {len(dicionario) / max(len(por_sinal), 1):.2f}",
        f"articuladores ............ {', '.join(dicionario.fontes)}",
        f"tempo de avaliação ....... {segundos / 60:.1f} min",
        "",
        "-" * 72,
        "RODÍZIOS",
        "-" * 72,
        "",
    ]
    for fonte, metricas in resultados.items():
        rotulo = "TOTAL" if fonte == "total" else f"sem {fonte}"
        linhas.append(f"  {rotulo:<16s} {metricas}")

    total = resultados["total"]

    if dicionario.metrica != "dtw":
        linhas += [
            "",
            "-" * 72,
            "ATENÇÃO",
            "-" * 72,
            "",
            "Este dicionário não é de sequências, então as representações saíram de",
            "um encoder — e o encoder salvo por train_sinais.py foi treinado com os",
            "**três** articuladores. Rodar leave-one-articulator-out sobre ele deixa",
            "o articulador de teste dentro do treino e mede memorização de pessoa,",
            "que é exatamente o erro do L↔G da fase 1.",
            "",
            "O número honesto do encoder é o de models/relatorio_encoder.txt, onde",
            "cada rodízio tem o seu próprio treino.",
        ]
    linhas += [
        "",
        "-" * 72,
        "LEITURA",
        "-" * 72,
        "",
        f"A resposta certa aparece na tela em {total.recall_5:.1%} das consultas.",
        f"Ela vem em primeiro lugar em {total.recall_1:.1%}.",
        "",
        "recall@5 é a métrica do produto: a tela mostra cinco candidatos, e um",
        "dicionário em que você reconhece a palavra certa entre cinco funciona.",
        "recall@1 descreve o conforto, não o serviço.",
        "",
        "Referência da área para busca em vocabulário grande: 50,8% de MRR em",
        "10.235 sinais (arXiv:2502.20171).",
        "",
        _veredito_sobre_o_encoder(total),
        "",
    ]
    return "\n".join(linhas)


def _veredito_sobre_o_encoder(total: avaliacao.Metricas) -> str:
    if total.recall_5 >= 0.80:
        return (
            "A baseline DTW já entrega o produto. O encoder neural só entra se\n"
            "bater este número por margem clara — torch é a maior mudança de\n"
            "dependência do projeto e precisa comprar alguma coisa."
        )
    return (
        "A baseline DTW não sustenta o produto sozinha. É aqui que o encoder\n"
        "neural com metric learning se justifica: ele tem este número para bater."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dicionario", type=Path, default=config.DICIONARIO_SINAIS)
    ap.add_argument("--saida", type=Path, default=config.RELATORIO_SINAIS)
    ap.add_argument("--k", type=int, default=config.CANDIDATOS_NA_TELA)
    ap.add_argument(
        "--nucleo",
        action="store_true",
        help="mede só o vocabulário núcleo, que é o que o app indexa por padrão",
    )
    args = ap.parse_args()

    if not args.dicionario.exists():
        print(
            f"dicionário não encontrado em {args.dicionario}\n"
            "Rode antes: python training/prepare_sinais.py --videos <raiz>"
        )
        raise SystemExit(1)

    dicionario = Dicionario.carregar(args.dicionario)
    if args.nucleo:
        faltando = nucleo.ausentes(dicionario.vocabulario)
        if faltando:
            print(
                f"{len(faltando)} sinais do núcleo não existem neste dicionário: "
                f"{', '.join(faltando)}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        dicionario = dicionario.restringir(nucleo.CHAVES)

    print(f"{len(dicionario)} protótipos, {dicionario.sinais} sinais")
    print(f"rodízio por articulador: {', '.join(dicionario.fontes)}\n")

    inicio = time.time()
    resultados = avaliacao.leave_one_articulator_out(dicionario, k=args.k)
    segundos = time.time() - inicio

    relatorio = montar_relatorio(dicionario, resultados, segundos)
    print(relatorio)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(relatorio, encoding="utf-8")
    print(f"relatório salvo em {args.saida}")


if __name__ == "__main__":
    main()
