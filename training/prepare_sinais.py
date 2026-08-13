"""Transforma os vídeos do V-LIBRASIL no dicionário de protótipos.

    python training/prepare_sinais.py --videos data/raw/v-librasil

Baixe a base em https://libras.cin.ufpe.br (ou o espelho no Kaggle) e extraia
mantendo **uma pasta por articulador**, porque é dela que sai a única avaliação
honesta possível com três pessoas — leave-one-articulator-out:

    data/raw/v-librasil/
      articulador_1/casa.mp4
      articulador_2/casa.mp4
      articulador_3/casa.mp4

São ~4.089 vídeos e ~370 mil frames. A extração roda em vários processos por
padrão (`--jobs`), o que traz o trabalho de 2-5 horas para a casa da meia hora
numa máquina de 10 núcleos. O trabalho é retomável — interromper e rodar de novo
continua de onde parou.

Nenhum frame é gravado. Cada vídeo vira 32×147 números e a imagem é descartada,
exatamente como na fase 1.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from libras import catalogo, config, sequencia  # noqa: E402
from libras.detector import DetectorSinais  # noqa: E402
from libras.dicionario import Dicionario  # noqa: E402

LARGURA_PROCESSAMENTO = 640  # 1440p não melhora landmark e custa 5x mais tempo
SALVAR_A_CADA = 100


def jobs_padrao() -> int:
    """Quantos processos usar sem que a máquina fique inusável.

    Cada worker carrega dois modelos do MediaPipe (~400 MB), então o teto não é
    o número de núcleos e sim a RAM. Oito é o limite prático em 16 GB.
    """
    return max(1, min(8, (os.cpu_count() or 2) - 2))


class Resultado(NamedTuple):
    """O que sai de um vídeo. Atravessa processos, então só tipos simples."""

    caminho: str
    vetores: np.ndarray | None
    rotulo: str | None
    articulador: str | None
    problema: str | None


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


def processar(
    video: Path,
    raiz: Path,
    passo: int,
    canhotos: set[str],
    detector: DetectorSinais,
    relogio_ms: int,
) -> tuple[Resultado, int]:
    """Um vídeo → `Resultado`. Nenhum erro de um arquivo derruba a extração.

    Devolve o relógio junto porque ele pertence ao detector, não ao vídeo: quem
    chamar de novo com o mesmo detector precisa continuar de onde parou.
    """
    articulador = catalogo.articulador_do_caminho(video, raiz)

    try:
        bruta, relogio_ms = ler_gravacao(video, detector, passo, relogio_ms)
        if bruta is None:
            return Resultado(str(video), None, None, None, "vídeo ilegível ou vazio"), relogio_ms

        pronta = sequencia.preparar(bruta, espelhar=articulador in canhotos)
        return (
            Resultado(
                str(video),
                # Com a máscara: ela não é reconstituível depois da imputação, e
                # recuperá-la custaria justamente esta extração de novo.
                pronta.vetores_com_validade,
                catalogo.rotulo_do_arquivo(video),
                articulador,
                None,
            ),
            relogio_ms,
        )
    except ValueError as erro:
        motivo = str(erro).split("—")[0].strip()
        return Resultado(str(video), None, None, None, motivo), relogio_ms


# --- worker -----------------------------------------------------------------
#
# O detector é global do processo porque criá-lo custa uma carga de modelo: um
# por worker, não um por vídeo. Cada worker tem o seu, e por isso o relógio de
# timestamps também é por worker — a monotonicidade que o MediaPipe exige vale
# por landmarker, e processos separados não compartilham nenhum.

_DETECTOR: DetectorSinais | None = None
_RELOGIO_MS = 0


def _abrir_detector() -> None:
    """Um detector por worker, vivo enquanto o pool viver.

    Não há `fechar()` nem `atexit` aqui de propósito: o worker morre junto com o
    pool e o sistema recupera os recursos nativos. Fechá-los durante o shutdown
    do interpretador só produzia tracebacks de `Exception ignored`, porque a
    essa altura o MediaPipe já desmontou parte do que o `close()` procura.
    """
    global _DETECTOR
    _DETECTOR = DetectorSinais()


def _processar_tarefa(tarefa: tuple[str, str, int, frozenset]) -> Resultado:
    global _RELOGIO_MS
    caminho, raiz, passo, canhotos = tarefa
    assert _DETECTOR is not None, "worker sem detector: initializer não rodou"

    resultado, _RELOGIO_MS = processar(
        Path(caminho), Path(raiz), passo, set(canhotos), _DETECTOR, _RELOGIO_MS
    )
    return resultado


def _sequencial(
    videos: list[Path], raiz: Path, passo: int, canhotos: set[str]
) -> Iterator[Resultado]:
    relogio_ms = 0  # contínuo entre vídeos; ver `ler_gravacao`

    with DetectorSinais() as detector:
        for video in videos:
            resultado, relogio_ms = processar(
                video, raiz, passo, canhotos, detector, relogio_ms
            )
            yield resultado


def _paralelo(
    videos: list[Path], raiz: Path, passo: int, canhotos: set[str], jobs: int
) -> Iterator[Resultado]:
    """Mesmos resultados do sequencial, em N processos.

    `spawn` é o padrão do macOS e o único seguro aqui: o MediaPipe segura
    recursos nativos que não sobrevivem a um fork.
    """
    tarefas = [(str(v), str(raiz), passo, frozenset(canhotos)) for v in videos]
    contexto = multiprocessing.get_context("spawn")

    with contexto.Pool(jobs, initializer=_abrir_detector) as pool:
        yield from pool.imap_unordered(_processar_tarefa, tarefas, chunksize=4)


def extrair(
    raiz: Path,
    passo: int,
    canhotos: set[str],
    limite: int | None,
    parcial: Path,
    jobs: int = 1,
) -> tuple[list[np.ndarray], list[str], list[str], Counter]:
    videos = catalogo.listar_videos(raiz)
    if limite:
        videos = videos[:limite]

    representacoes, rotulos, fontes, feitos = _retomar(parcial)
    problemas: Counter = Counter()

    if feitos:
        print(f"retomando: {len(feitos)} vídeos já processados")

    pendentes = [v for v in videos if str(v) not in feitos]
    print(f"{len(videos)} vídeos no total, {len(pendentes)} a processar")
    print(f"{jobs} processo(s) em paralelo\n")

    inicio = time.time()
    fluxo = (
        _sequencial(pendentes, raiz, passo, canhotos)
        if jobs == 1
        else _paralelo(pendentes, raiz, passo, canhotos, jobs)
    )

    try:
        for i, resultado in enumerate(fluxo, start=1):
            if resultado.problema:
                problemas[resultado.problema] += 1
            else:
                representacoes.append(resultado.vetores)
                rotulos.append(resultado.rotulo)
                fontes.append(resultado.articulador)

            feitos.add(resultado.caminho)
            _progresso(i, len(pendentes), inicio, len(representacoes))

            if i % SALVAR_A_CADA == 0:
                _salvar_parcial(parcial, representacoes, rotulos, fontes, feitos)
    except KeyboardInterrupt:
        # Uma hora de trabalho não pode morrer por um Ctrl-C mal colocado. O
        # parcial só é gravado a cada SALVAR_A_CADA, então sem isto a
        # interrupção jogaria fora até 99 vídeos já processados.
        _salvar_parcial(parcial, representacoes, rotulos, fontes, feitos)
        print(f"\n\ninterrompido — {len(feitos)} vídeos salvos em {parcial}")
        print("rode de novo para continuar de onde parou")
        raise

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
    """O que já foi extraído, ou nada — inclusive quando "nada" é a resposta certa.

    Um parcial de uma versão anterior do extrator tem a largura antiga, e
    continuar em cima dele misturaria vetores de 147 e de 196 canais no mesmo
    array. O numpy aceitaria calado (viraria um array de objetos) e o estrago só
    apareceria horas depois, no treino. Largura diferente da atual = recomeçar.
    """
    if not parcial.exists():
        return [], [], [], set()

    dados = np.load(parcial, allow_pickle=False)
    representacoes = dados["representacoes"]

    if representacoes.ndim != 3 or (
        representacoes.shape[-1] != sequencia.TAMANHO_COM_VALIDADE
    ):
        print(
            f"parcial em {parcial} tem largura "
            f"{representacoes.shape[-1] if representacoes.ndim == 3 else '?'}, "
            f"e o extrator agora grava {sequencia.TAMANHO_COM_VALIDADE}. "
            "Recomeçando a extração."
        )
        return [], [], [], set()

    return (
        list(representacoes),
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
    ap.add_argument(
        "--jobs",
        type=int,
        default=jobs_padrao(),
        help=f"processos em paralelo; 1 desliga (padrão: {jobs_padrao()})",
    )
    args = ap.parse_args()

    if args.jobs < 1:
        ap.error("--jobs deve ser >= 1")

    parcial = args.saida.with_suffix(".parcial.npz")

    representacoes, rotulos, fontes, problemas = extrair(
        args.videos, args.passo, set(args.canhoto), args.limite, parcial, args.jobs
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
