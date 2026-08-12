"""Extrai o zip do V-LIBRASIL no layout que `libras.catalogo` espera.

O espelho do Kaggle entrega tudo numa pasta só, com o articulador no nome do
arquivo:

    videos UFPE (V-LIBRASIL)/data/Abacaxi_Articulador1.mp4

`catalogo.articulador_do_caminho` lê o articulador da **pasta**, porque é dela
que sai o leave-one-articulator-out. Sem reorganizar, os 4.086 vídeos cairiam
todos em `desconhecido` e a avaliação mediria memorização de pessoa — o mesmo
erro que a fase 1 cometeu com o L↔G. Daí este passo existir:

    data/raw/v-librasil/articulador_1/Abacaxi.mp4
    data/raw/v-librasil/articulador_2/Abacaxi.mp4

Usa `zipfile` do Python e não o `unzip` do sistema porque o unzip do macOS
corrompe os nomes UTF-8 do pacote (`Abençoar` vira `Aben+?oar`) e aborta a
extração com um falso "disk full".
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

NOME_ARTICULADOR = re.compile(r"^(?P<nome>.+)_Articulador(?P<numero>\d+)$", re.I)


def destino_do_membro(membro: str) -> Path | None:
    """`.../Abacaxi_Articulador1.mp4` → `articulador_1/Abacaxi.mp4`.

    Devolve None para o que não é vídeo no padrão esperado — os CSVs de
    anotação, por exemplo, que são copiados à parte.
    """
    caminho = Path(membro)
    if caminho.suffix.lower() != ".mp4":
        return None

    casou = NOME_ARTICULADOR.match(caminho.stem)
    if not casou:
        return None

    return Path(f"articulador_{int(casou['numero'])}") / f"{casou['nome']}.mp4"


def organizar(zip_path: Path, destino: Path) -> Counter:
    contagem: Counter = Counter()

    with zipfile.ZipFile(zip_path) as pacote:
        membros = pacote.infolist()
        total = sum(1 for m in membros if destino_do_membro(m.filename))
        feitos = 0

        for membro in membros:
            relativo = destino_do_membro(membro.filename)

            if relativo is None:
                if membro.filename.lower().endswith(".csv"):
                    alvo = destino / Path(membro.filename).name
                    alvo.parent.mkdir(parents=True, exist_ok=True)
                    with pacote.open(membro) as origem, open(alvo, "wb") as saida:
                        shutil.copyfileobj(origem, saida)
                    contagem["anotações"] += 1
                else:
                    contagem["ignorado"] += 1
                continue

            alvo = destino / relativo
            alvo.parent.mkdir(parents=True, exist_ok=True)

            # Retomável: um arquivo já do tamanho certo não é reescrito.
            if alvo.exists() and alvo.stat().st_size == membro.file_size:
                contagem["já existia"] += 1
            else:
                with pacote.open(membro) as origem, open(alvo, "wb") as saida:
                    shutil.copyfileobj(origem, saida, length=1024 * 1024)
                contagem[relativo.parts[0]] += 1

            feitos += 1
            print(f"\r{feitos}/{total} vídeos", end="", flush=True)

    print()
    return contagem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=Path("data/raw/v-librasil.zip"))
    ap.add_argument("--destino", type=Path, default=Path("data/raw/v-librasil"))
    args = ap.parse_args()

    if not args.zip.exists():
        print(f"zip não encontrado: {args.zip}", file=sys.stderr)
        print("Rode antes: bash scripts/download_vlibrasil.sh", file=sys.stderr)
        return 1

    contagem = organizar(args.zip, args.destino)

    for chave, quantos in sorted(contagem.items()):
        print(f"  {chave:20s} {quantos:5d}")

    videos = sum(v for k, v in contagem.items() if k.startswith("articulador"))
    print(f"\n{videos} vídeos organizados em {args.destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
