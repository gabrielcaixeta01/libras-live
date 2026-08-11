"""Loop principal: liga câmera, detector, classificador, estabilizador e UI.

    python -m libras.app
"""

from __future__ import annotations

import sys
import time

import cv2

from . import config, ui
from .camera import Camera, CameraIndisponivel
from .classifier import Classificador, ModeloAusente
from .landmarks import DetectorMaos
from .speller import Soletrador
from .stabilizer import Estabilizador

ESC = 27
BACKSPACE = (8, 127)  # varia entre plataformas
TECLA_LIMPAR = ord("c")


def executar() -> int:
    try:
        classificador = Classificador()
    except ModeloAusente as erro:
        print(erro, file=sys.stderr)
        return 1

    try:
        camera = Camera()
    except CameraIndisponivel as erro:
        print(erro, file=sys.stderr)
        return 1

    estabilizador = Estabilizador()
    soletrador = Soletrador()

    print(f"Modelo: {classificador.nome} | acuracia {classificador.acuracia:.1%}")
    print(f"Letras: {''.join(classificador.letras)}")
    print("ESC sair | BACKSPACE apagar | C limpar")

    inicio = time.perf_counter()
    instante_anterior = inicio
    fps = 0.0

    with camera, DetectorMaos() as detector:
        while True:
            leitura = camera.ler()
            if leitura is None:
                print("Falha ao ler o frame da camera.", file=sys.stderr)
                return 1

            frame_bgr, frame_rgb = leitura

            agora = time.perf_counter()
            delta = agora - instante_anterior
            instante_anterior = agora
            if delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta)

            # detect_for_video exige timestamps crescentes em milissegundos.
            timestamp_ms = int((agora - inicio) * 1000)
            deteccao = detector.detectar(frame_rgb, timestamp_ms)

            if deteccao is None:
                letra, confianca = None, 0.0
                soletrador.registrar_ausencia(delta)
            else:
                letra, confianca = classificador.prever(deteccao.vetor)
                soletrador.registrar_presenca()
                ui.desenhar_mao(frame_bgr, deteccao.pontos)

            confirmada = estabilizador.atualizar(letra, confianca)
            if confirmada:
                soletrador.adicionar(confirmada)

            ui.desenhar_predicao(
                frame_bgr,
                letra,
                confianca,
                confirmada is not None,
                estabilizador.preenchimento,
            )
            ui.desenhar_texto(frame_bgr, soletrador.texto)
            ui.desenhar_fps(frame_bgr, fps)

            cv2.imshow("libras-live", frame_bgr)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ESC:
                break
            if tecla in BACKSPACE:
                soletrador.apagar()
            elif tecla == TECLA_LIMPAR:
                soletrador.limpar()
                estabilizador.reiniciar()

    cv2.destroyAllWindows()
    print(f"\nTexto final: {soletrador.texto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
