"""Baixa a base pública de Libras e converte as imagens em landmarks.

Base: Brazilian Sign Language Alphabet Dataset (biankatpas) — 4.411 imagens do
alfabeto de Libras, 15 letras. É Libras mesmo, não ASL.

A saída é o mesmo formato de `collect.py`: vetores de 63 floats. É isso que
permite misturar imagem baixada e webcam sem reconciliar formato nenhum.

    python training/prepare_dataset.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libras import config
from libras.landmarks import DetectorImagem

URL_ZIP = (
    "https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset"
    "/archive/refs/heads/master.zip"
)
DIR_BRUTO = config.DIR_DADOS / "raw"
SAIDA = config.DIR_DADOS / "dataset_publico.npz"
EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp"}


def baixar() -> Path:
    """Baixa e extrai o zip. Reaproveita o que já estiver em disco."""
    destino = DIR_BRUTO / "Brazilian-Sign-Language-Alphabet-Dataset-master" / "dataset"
    if destino.exists():
        print(f"Dataset já extraído em {destino}")
        return destino

    DIR_BRUTO.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {URL_ZIP} (~57MB)...")

    with urllib.request.urlopen(URL_ZIP) as resposta:
        conteudo = resposta.read()

    print("Extraindo...")
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zip_arquivo:
        zip_arquivo.extractall(DIR_BRUTO)

    if not destino.exists():
        raise RuntimeError(f"Esperava encontrar {destino} após extrair o zip.")

    return destino


def extrair_landmarks(dir_dataset: Path) -> tuple[np.ndarray, np.ndarray]:
    """Roda o MediaPipe em cada imagem. Imagens sem mão detectável são puladas."""
    vetores: list[np.ndarray] = []
    rotulos: list[str] = []
    puladas: Counter[str] = Counter()

    letras = sorted(p.name for p in dir_dataset.iterdir() if p.is_dir())
    print(f"Letras encontradas: {' '.join(letras)}")

    with DetectorImagem() as detector:
        for letra in letras:
            imagens = sorted(
                p for p in (dir_dataset / letra).iterdir()
                if p.suffix.lower() in EXTENSOES
            )
            aproveitadas = 0

            for caminho in imagens:
                imagem = cv2.imread(str(caminho))
                if imagem is None:
                    puladas[letra] += 1
                    continue

                rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                deteccao = detector.detectar(np.ascontiguousarray(rgb))

                if deteccao is None:
                    puladas[letra] += 1
                    continue

                vetores.append(deteccao.vetor)
                rotulos.append(letra)
                aproveitadas += 1

            total = len(imagens)
            taxa = aproveitadas / total if total else 0.0
            print(f"  {letra}: {aproveitadas}/{total} imagens ({taxa:.0%})")

    if not vetores:
        raise RuntimeError("Nenhuma mão detectada em nenhuma imagem.")

    print(f"\nTotal aproveitado: {len(vetores)} | puladas: {sum(puladas.values())}")
    return np.array(vetores, dtype=np.float32), np.array(rotulos)


def main() -> int:
    dir_dataset = baixar()
    X, y = extrair_landmarks(dir_dataset)

    config.DIR_DADOS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAIDA, X=X, y=y)
    print(f"\nSalvo em {SAIDA} — X{X.shape}, {len(set(y))} classes")

    faltando = sorted(set(config.ALFABETO) - set(y))
    if faltando:
        print(f"\nFaltam {len(faltando)} letras: {' '.join(faltando)}")
        print("Grave essas com: python training/collect.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
