"""Transforma os vídeos do V-LIBRASIL no dicionário de protótipos.

    python training/prepare_sinais.py --videos data/raw/v-librasil

Baixe a base em https://libras.cin.ufpe.br (ou o espelho no Kaggle) e extraia
mantendo **uma pasta por articulador**, porque é dela que sai a única avaliação
honesta possível com três pessoas — leave-one-articulator-out:

    data/raw/v-librasil/
      articulador_1/casa.mp4
      articulador_2/casa.mp4
      articulador_3/casa.mp4

São ~4.089 vídeos e ~370 mil frames: conte 2 a 5 horas de CPU. O trabalho é
retomável — interromper e rodar de novo continua de onde parou.

Nenhum frame é gravado. Cada vídeo vira 32×147 números e a imagem é descartada,
exatamente como na fase 1.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from libras import catalogo, config, sequencia  # noqa: E402
from libras.detector import DetectorSinais  # noqa: E402
from libras.dicionario import Dicionario  # noqa: E402

LARGURA_PROCESSAMENTO = 640  # 1440p não melhora landmark e custa 5x mais tempo
SALVAR_A_CADA = 100


def ler_gravacao(
    caminho: Path,
    detector: DetectorSinais,
    passo: int = 1,
    base_ms: int = 0,
) -> tuple[np.ndarray | None, int]:
    """Vídeo → ((T, 49, 3) cru, próximo timestamp livre).

    `base_ms` existe por uma exigência do MediaPipe que é fácil de descobrir
    tarde demais: no modo VIDEO os timestamps precisam crescer ao longo de toda
    a vida do landmarker, e não de cada vídeo. Como o mesmo detector processa os
    4.089 arquivos (recriá-lo por vídeo custaria milhares de cargas de modelo),
    o relógio precisa ser contínuo entre eles — senão o segundo vídeo morre com
    "Input timestamp must be monotonically increasing" e a extração inteira
    devolve zero.
    """
    captura = cv2.VideoCapture(str(caminho))
    if not captura.isOpened():
        return None, base_ms

    frames: list[np.ndarray] = []
    indice = 0

    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                break

            if indice % passo == 0:
                marca = base_ms + len(frames) * 33
                frames.append(_landmarks_do_frame(frame, detector, marca))
            indice += 1
    finally:
        captura.release()

    proximo = base_ms + (len(frames) + 1) * 33
    return (np.stack(frames) if frames else None), proximo


def _landmarks_do_frame(frame_bgr, detector: DetectorSinais, marca_ms: int) -> np.ndarray:
    altura, largura = frame_bgr.shape[:2]
    if largura > LARGURA_PROCESSAMENTO:
        escala = LARGURA_PROCESSAMENTO / largura
        frame_bgr = cv2.resize(frame_bgr, (LARGURA_PROCESSAMENTO, int(altura * escala)))

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Timestamp sintético: o modo VIDEO só exige monotonicidade, e a taxa real
    # do arquivo não muda o resultado.
    return detector.detectar(rgb, timestamp_ms=marca_ms).frame


def extrair(
    raiz: Path,
    passo: int,
    canhotos: set[str],
    limite: int | None,
    parcial: Path,
) -> tuple[list[np.ndarray], list[str], list[str], Counter]:
    videos = catalogo.listar_videos(raiz)
    if limite:
        videos = videos[:limite]

    representacoes, rotulos, fontes, feitos = _retomar(parcial)
    problemas: Counter = Counter()

    if feitos:
        print(f"retomando: {len(feitos)} vídeos já processados")

    pendentes = [v for v in videos if str(v) not in feitos]
    print(f"{len(videos)} vídeos no total, {len(pendentes)} a processar\n")

    inicio = time.time()
    relogio_ms = 0  # contínuo entre vídeos; ver `ler_gravacao`

    with DetectorSinais() as detector:
        for i, video in enumerate(pendentes, start=1):
            articulador = catalogo.articulador_do_caminho(video, raiz)

            try:
                bruta, relogio_ms = ler_gravacao(video, detector, passo, relogio_ms)
                if bruta is None:
                    problemas["vídeo ilegível ou vazio"] += 1
                else:
                    pronta = sequencia.preparar(
                        bruta, espelhar=articulador in canhotos
                    )
                    representacoes.append(pronta.vetores)
                    rotulos.append(catalogo.rotulo_do_arquivo(video))
                    fontes.append(articulador)
            except ValueError as erro:
                problemas[str(erro).split("—")[0].strip()] += 1

            feitos.add(str(video))
            _progresso(i, len(pendentes), inicio, len(representacoes))

            if i % SALVAR_A_CADA == 0:
                _salvar_parcial(parcial, representacoes, rotulos, fontes, feitos)

    print()
    return representacoes, rotulos, fontes, problemas


def _progresso(i: int, total: int, inicio: float, aceitos: int) -> None:
    decorrido = time.time() - inicio
    restante = (decorrido / i) * (total - i)
    print(
        f"\r{i}/{total} — {aceitos} aceitos — "
        f"faltam ~{restante / 60:.0f} min",
        end="",
        flush=True,
    )


def _retomar(parcial: Path):
    if not parcial.exists():
        return [], [], [], set()

    dados = np.load(parcial, allow_pickle=False)
    return (
        list(dados["representacoes"]),
        [str(r) for r in dados["rotulos"]],
        [str(f) for f in dados["fontes"]],
        {str(v) for v in dados["feitos"]},
    )


def _salvar_parcial(parcial: Path, representacoes, rotulos, fontes, feitos) -> None:
    parcial.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        parcial,
        representacoes=np.array(representacoes, dtype=np.float32),
        rotulos=np.array(rotulos),
        fontes=np.array(fontes),
        feitos=np.array(sorted(feitos)),
    )


def relatar(rotulos: list[str], fontes: list[str], problemas: Counter) -> None:
    por_articulador = Counter(fontes)
    por_sinal = Counter(catalogo.chave(r) for r in rotulos)

    print(f"\n{len(rotulos)} gravações aproveitadas")
    print(f"{len(por_sinal)} sinais distintos")
    print(f"{np.mean(list(por_sinal.values())):.2f} gravações por sinal em média\n")

    for articulador, quantos in sorted(por_articulador.items()):
        print(f"  {articulador:20s} {quantos:5d}")

    if problemas:
        print("\ndescartes:")
        for motivo, quantos in problemas.most_common():
            print(f"  {quantos:5d}  {motivo}")

    orfaos = [s for s, n in por_sinal.items() if n < 2]
    if orfaos:
        print(
            f"\n{len(orfaos)} sinais com uma gravação só — eles entram no "
            "dicionário mas não podem ser avaliados com "
            "leave-one-articulator-out."
        )

    if len(por_articulador) < 2:
        print(
            "\nAVISO: um articulador só. A avaliação vai medir memorização, não "
            "generalização — mantenha uma pasta por articulador."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", type=Path, required=True, help="raiz dos vídeos")
    ap.add_argument("--saida", type=Path, default=config.DICIONARIO_SINAIS)
    ap.add_argument(
        "--passo",
        type=int,
        default=1,
        help="processa 1 a cada N frames; 2 corta o tempo pela metade",
    )
    ap.add_argument(
        "--canhoto",
        action="append",
        default=[],
        metavar="ARTICULADOR",
        help="articulador canhoto, a ser espelhado (pode repetir)",
    )
    ap.add_argument("--limite", type=int, help="processa só os N primeiros (teste)")
    args = ap.parse_args()

    parcial = args.saida.with_suffix(".parcial.npz")

    representacoes, rotulos, fontes, problemas = extrair(
        args.videos, args.passo, set(args.canhoto), args.limite, parcial
    )

    if not representacoes:
        # O motivo importa mais aqui do que em qualquer outro lugar: é o único
        # sinal que a pessoa tem sobre o que corrigir antes de tentar de novo.
        print("\nnenhuma gravação aproveitada — nada a salvar")
        if problemas:
            print("\nmotivos:")
            for motivo, quantos in problemas.most_common():
                print(f"  {quantos:5d}  {motivo}")
        raise SystemExit(1)

    relatar(rotulos, fontes, problemas)

    Dicionario(
        representacoes=np.array(representacoes, dtype=np.float32),
        rotulos=rotulos,
        fontes=fontes,
        metrica="dtw",
    ).salvar(args.saida)

    parcial.unlink(missing_ok=True)
    print(f"\ndicionário salvo em {args.saida}")


if __name__ == "__main__":
    main()
