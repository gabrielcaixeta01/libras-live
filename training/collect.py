"""Grava suas próprias amostras das letras que faltam na base pública.

São 11 letras: F G H J K P Q T X Y Z.

Cada letra vira um arquivo em `data/coletados/<LETRA>.npy`, então dá para parar
no meio e continuar depois — o script pula o que já existe.

    python training/collect.py            # todas as faltantes, pulando as prontas
    python training/collect.py F G H      # só essas
    python training/collect.py --refazer F

Enquanto grava, mexa um pouco a mão: gire o pulso, aproxime e afaste, mude o
ângulo. Amostras todas idênticas ensinam o modelo a reconhecer uma pose exata em
vez da letra.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libras import config, ui
from libras.camera import Camera, CameraIndisponivel
from libras.landmarks import DetectorMaos

DIR_SAIDA = config.DIR_DADOS / "coletados"
ESC = 27
ESPACO = 32


def gravar_letra(
    camera: Camera, detector: DetectorMaos, letra: str, inicio: float
) -> np.ndarray | None:
    """Grava as amostras de uma letra. Retorna None se o usuário abortou."""
    amostras: list[np.ndarray] = []
    preparando = True
    inicio_contagem = time.perf_counter()

    while True:
        leitura = camera.ler()
        if leitura is None:
            return None

        frame_bgr, frame_rgb = leitura
        agora = time.perf_counter()
        deteccao = detector.detectar(frame_rgb, int((agora - inicio) * 1000))

        if deteccao is not None:
            ui.desenhar_mao(frame_bgr, deteccao.pontos)

        if preparando:
            restante = config.SEGUNDOS_PREPARACAO - (agora - inicio_contagem)
            if restante <= 0:
                preparando = False
            else:
                _texto_central(frame_bgr, f"{letra}  {int(restante) + 1}")
                _rodape(frame_bgr, "posicione a mao - ESC cancela")
        else:
            if deteccao is not None:
                amostras.append(deteccao.vetor)

            progresso = len(amostras) / config.AMOSTRAS_POR_LETRA
            _texto_central(frame_bgr, f"{letra}  {len(amostras)}")
            _barra(frame_bgr, progresso)
            _rodape(frame_bgr, "gravando - mexa a mao devagar")

            if len(amostras) >= config.AMOSTRAS_POR_LETRA:
                cv2.imshow("coleta", frame_bgr)
                cv2.waitKey(300)
                return np.array(amostras, dtype=np.float32)

        cv2.imshow("coleta", frame_bgr)
        if (cv2.waitKey(1) & 0xFF) == ESC:
            return None


def _texto_central(frame, texto: str) -> None:
    altura, largura = frame.shape[:2]
    tamanho = cv2.getTextSize(texto, ui.FONTE, 3.0, 6)[0]
    origem = ((largura - tamanho[0]) // 2, 90)
    cv2.putText(frame, texto, origem, ui.FONTE, 3.0, ui.VERDE, 6, cv2.LINE_AA)


def _barra(frame, progresso: float) -> None:
    altura, largura = frame.shape[:2]
    x0, x1 = 60, largura - 60
    y = 130
    cv2.rectangle(frame, (x0, y), (x1, y + 14), ui.CINZA, 1)
    cv2.rectangle(
        frame, (x0, y), (x0 + int((x1 - x0) * progresso), y + 14), ui.AZUL, -1
    )


def _rodape(frame, texto: str) -> None:
    altura = frame.shape[0]
    cv2.putText(
        frame, texto, (30, altura - 30), ui.FONTE, 0.8, ui.BRANCO, 2, cv2.LINE_AA
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("letras", nargs="*", help="letras a gravar")
    parser.add_argument(
        "--refazer", action="store_true", help="regrava letras já existentes"
    )
    args = parser.parse_args()

    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    alvo = [l.upper() for l in args.letras] or config.LETRAS_A_COLETAR

    pendentes = [
        letra for letra in alvo
        if args.refazer or not (DIR_SAIDA / f"{letra}.npy").exists()
    ]

    if not pendentes:
        print("Todas as letras pedidas já foram gravadas.")
        print(f"Use --refazer para regravar. Arquivos em {DIR_SAIDA}")
        return 0

    print(f"A gravar: {' '.join(pendentes)}")
    print(f"{config.AMOSTRAS_POR_LETRA} amostras por letra. ESC cancela.\n")

    try:
        camera = Camera()
    except CameraIndisponivel as erro:
        print(erro, file=sys.stderr)
        return 1

    inicio = time.perf_counter()

    with camera, DetectorMaos() as detector:
        for letra in pendentes:
            print(f"Letra {letra}...", end=" ", flush=True)
            amostras = gravar_letra(camera, detector, letra, inicio)

            if amostras is None:
                print("cancelado.")
                break

            np.save(DIR_SAIDA / f"{letra}.npy", amostras)
            print(f"{len(amostras)} amostras salvas.")

    cv2.destroyAllWindows()

    prontas = sorted(p.stem for p in DIR_SAIDA.glob("*.npy"))
    faltam = sorted(set(config.LETRAS_A_COLETAR) - set(prontas))
    print(f"\nGravadas: {' '.join(prontas) or '(nenhuma)'}")
    if faltam:
        print(f"Ainda faltam: {' '.join(faltam)}")
    else:
        print("Alfabeto completo. Agora rode: python training/train.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
