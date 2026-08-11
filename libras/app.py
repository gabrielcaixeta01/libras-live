"""Loop principal: liga câmera, detector, classificador, estabilizador e UI.

    python -m libras.app              # soletrar: sua mão vira texto
    python -m libras.app --praticar   # praticar: ele pede a letra, você faz
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time

import cv2

from . import config, ui
from .camera import Camera, CameraIndisponivel
from .classifier import Classificador, ModeloAusente
from .landmarks import DetectorMaos
from .practice import Pratica
from .speller import Soletrador
from .stabilizer import Estabilizador

ESC = 27
BACKSPACE = (8, 127)  # varia entre plataformas
TECLA_LIMPAR = ord("c")
TECLA_PULAR = (ord("p"), ord(" "))
TECLA_REINICIAR = ord("r")

SEGUNDOS_MOSTRANDO_ERRO = 1.2

AJUDA_SOLETRAR = "ESC sair   BACKSPACE apagar   C limpar"
AJUDA_PRATICAR = "ESC sair   P pular   R reiniciar"


def _copiar(texto: str) -> bool:
    """Manda o texto para a área de transferência. Falhar aqui não é erro fatal."""
    if not texto.strip():
        return False

    comandos = {"Darwin": ["pbcopy"], "Linux": ["xclip", "-selection", "clipboard"]}
    comando = comandos.get(platform.system())
    if comando is None:
        return False

    try:
        subprocess.run(comando, input=texto.encode("utf-8"), check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _anunciar(classificador: Classificador, praticando: bool) -> None:
    print(f"Modelo: {classificador.nome}", end="")
    if classificador.macro_f1:
        print(f" | macro-F1 {classificador.macro_f1:.1%}", end="")
    print(f" | acuracia {classificador.acuracia:.1%}")

    print(f"Letras conhecidas ({len(classificador.letras)}): "
          f"{''.join(classificador.letras)}")

    ausentes = classificador.letras_ausentes
    if ausentes:
        print(f"Sem dados ({len(ausentes)}): {''.join(ausentes)}"
              " — grave com: python training/collect.py")

    print(AJUDA_PRATICAR if praticando else AJUDA_SOLETRAR)


def executar(praticando: bool = False, rodadas: int = config.RODADAS_PRATICA) -> int:
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
    # Só faz sentido pedir letras que o modelo é capaz de reconhecer.
    sessao = Pratica(classificador.letras, rodadas=rodadas) if praticando else None

    _anunciar(classificador, praticando)

    inicio = time.perf_counter()
    instante_anterior = inicio
    inicio_rodada = inicio
    fps = 0.0
    erro_recente: str | None = None
    instante_erro = 0.0

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

            predicao = None
            mao_esquerda = None

            if deteccao is None:
                if sessao is None:
                    soletrador.registrar_ausencia(delta)
            else:
                predicao = classificador.prever(deteccao.vetor)
                mao_esquerda = deteccao.mao_esquerda
                if sessao is None:
                    soletrador.registrar_presenca()
                ui.desenhar_mao(frame_bgr, deteccao.pontos)

            # Uma predição rejeitada não alimenta o estabilizador: entra como
            # "sem letra", igual a um frame sem mão. Assim o "?" nunca vira texto.
            letra = predicao.letra if predicao and predicao.aceita else None
            confianca = predicao.confianca if predicao else 0.0
            confirmada = estabilizador.atualizar(letra, confianca)

            if confirmada and sessao is not None:
                if sessao.registrar(confirmada, agora - inicio_rodada):
                    inicio_rodada = agora
                    erro_recente = None
                else:
                    erro_recente, instante_erro = confirmada, agora
            elif confirmada:
                soletrador.adicionar(confirmada)

            if erro_recente and agora - instante_erro > SEGUNDOS_MOSTRANDO_ERRO:
                erro_recente = None

            ui.desenhar_predicao(
                frame_bgr,
                predicao,
                confirmada is not None,
                estabilizador.preenchimento,
                mao_esquerda,
            )
            ui.desenhar_alfabeto(
                frame_bgr, classificador.letras, classificador.letras_ausentes
            )

            if sessao is not None:
                ui.desenhar_pratica(frame_bgr, sessao, erro_recente, AJUDA_PRATICAR)
            else:
                ui.desenhar_texto(frame_bgr, soletrador.texto, AJUDA_SOLETRAR)

            ui.desenhar_fps(frame_bgr, fps)

            cv2.imshow("libras-live", frame_bgr)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ESC:
                break

            if sessao is not None:
                if tecla in TECLA_PULAR:
                    sessao.pular(agora - inicio_rodada)
                    inicio_rodada = agora
                    erro_recente = None
                elif tecla == TECLA_REINICIAR:
                    sessao.reiniciar()
                    inicio_rodada = agora
                    erro_recente = None
                    estabilizador.reiniciar()
            elif tecla in BACKSPACE:
                soletrador.apagar()
            elif tecla == TECLA_LIMPAR:
                soletrador.limpar()
                estabilizador.reiniciar()

    cv2.destroyAllWindows()

    if sessao is not None:
        print(f"\nPratica: {sessao.resumo()}")
    else:
        print(f"\nTexto final: {soletrador.texto}")
        if _copiar(soletrador.texto):
            print("(copiado para a area de transferencia)")

    return 0


def main(argv: list[str] | None = None) -> int:
    analisador = argparse.ArgumentParser(
        prog="python -m libras.app",
        description="Reconhecimento do alfabeto de Libras em tempo real.",
    )
    analisador.add_argument(
        "--praticar",
        action="store_true",
        help="modo treino: o app pede uma letra e confere se você acertou",
    )
    analisador.add_argument(
        "--rodadas",
        type=int,
        default=config.RODADAS_PRATICA,
        help=f"quantas letras por sessão de prática (padrão: {config.RODADAS_PRATICA})",
    )
    argumentos = analisador.parse_args(argv)

    if argumentos.rodadas < 1:
        analisador.error("--rodadas deve ser >= 1")

    return executar(praticando=argumentos.praticar, rodadas=argumentos.rodadas)


if __name__ == "__main__":
    raise SystemExit(main())
