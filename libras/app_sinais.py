"""Loop do dicionário reverso: você faz um sinal, ele diz que palavra é.

    python -m libras.app --sinais

Fica separado de `app.py` de propósito. O loop do alfabeto tem sete peças
conversando por frame e não precisa de um `if` a mais; este tem outras cinco, e
misturar os dois deixaria os dois piores.

**A correção também é a re-ancoragem.** Quando o resultado sai errado, você
aperta o número do candidato certo — e aquele sinal, feito pela sua mão, vira
protótipo. Da próxima vez ele acerta. É a mitigação do viés dos três
articuladores acontecendo no uso, não numa sessão de coleta separada.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from . import config, desenho, sequencia, ui
from .camera import Camera, CameraIndisponivel
from .detector import DetectorSinais
from .dicionario import FONTE_USUARIO, Dicionario
from .segmenter import Estado, Segmentador

ESC = 27
TECLA_LIMPAR = ord("c")
TECLA_NOVO = ord("n")
TECLAS_CANDIDATO = tuple(ord(str(n)) for n in range(1, 10))

AJUDA = "ESC sair   1-5 corrigir e ensinar   N nomear   C limpar"


def carregar_dicionario() -> Dicionario:
    """O dicionário base mais os seus protótipos, se existirem."""
    if not config.DICIONARIO_SINAIS.exists():
        raise FileNotFoundError(
            f"Dicionário de sinais não encontrado em {config.DICIONARIO_SINAIS}.\n"
            "Rode: python training/prepare_sinais.py --videos <raiz dos vídeos>\n"
            "A base V-LIBRASIL está em https://libras.cin.ufpe.br"
        )

    dicionario = Dicionario.carregar(config.DICIONARIO_SINAIS)

    if config.PROTOTIPOS_USUARIO.exists():
        meus = Dicionario.carregar(config.PROTOTIPOS_USUARIO)
        dicionario = dicionario.juntar(meus)
        print(f"{len(meus)} protótipos seus carregados")

    return dicionario


def salvar_prototipos(dicionario: Dicionario) -> int:
    """Grava só o que é seu. O dicionário base não é reescrito nunca."""
    meus = dicionario.apenas(FONTE_USUARIO)
    if len(meus) == 0:
        return 0

    meus.salvar(config.PROTOTIPOS_USUARIO)
    return len(meus)


def executar() -> int:
    try:
        dicionario = carregar_dicionario()
    except (FileNotFoundError, ValueError) as erro:
        print(erro, file=sys.stderr)
        return 1

    try:
        camera = Camera()
    except CameraIndisponivel as erro:
        print(erro, file=sys.stderr)
        return 1

    segmentador = Segmentador(
        limiar_movimento=config.LIMIAR_MOVIMENTO,
        frames_para_iniciar=config.FRAMES_PARA_INICIAR,
        segundos_repouso=config.SEGUNDOS_REPOUSO,
        segundos_minimo=config.SEGUNDOS_MINIMO_SINAL,
        segundos_maximo=config.SEGUNDOS_MAXIMO_SINAL,
    )

    print(f"{len(dicionario.vocabulario)} sinais no dicionário "
          f"({len(dicionario)} protótipos, métrica {dicionario.metrica})")
    print(AJUDA)

    inicio = time.perf_counter()
    instante_anterior = inicio
    fps = 0.0

    candidatos: list = []
    ultima_consulta = None       # a sequência preparada, para re-ancorar
    instante_resultado = 0.0
    aprendidos = 0

    codigo = 0

    # O `finally` não é zelo: o que a pessoa ensinou nesta sessão só existe em
    # memória até `salvar_prototipos`. Sem ele, um Ctrl-C — ou a câmera falhando
    # no meio — jogaria fora a correção manual, que é justamente a mitigação do
    # viés dos três articuladores.
    try:
        with camera, DetectorSinais() as detector:
            while True:
                leitura = camera.ler()
                if leitura is None:
                    print("Falha ao ler o frame da camera.", file=sys.stderr)
                    codigo = 1
                    break

                frame_bgr, frame_rgb = leitura

                agora = time.perf_counter()
                delta = agora - instante_anterior
                instante_anterior = agora
                if delta > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / delta)

                deteccao = detector.detectar(
                    frame_rgb,
                    timestamp_ms=int((agora - inicio) * 1000),
                    video_espelhado=config.ESPELHAR_VIDEO,
                )

                gravacao = segmentador.oferecer(deteccao.frame, agora - inicio)

                if gravacao is not None:
                    try:
                        ultima_consulta = sequencia.preparar(gravacao)
                        candidatos = dicionario.buscar(
                            ultima_consulta.vetores, k=config.CANDIDATOS_NA_TELA
                        )
                        instante_resultado = agora
                    except ValueError:
                        # Gravação sem âncora em frame nenhum: nada a consultar.
                        ultima_consulta, candidatos = None, []

                if agora - instante_resultado > config.SEGUNDOS_MOSTRANDO:
                    candidatos = []

                ui.desenhar_esqueleto(
                    frame_bgr, deteccao.maos_cruas, deteccao.corpo_cru
                )
                ui.desenhar_estado(
                    frame_bgr,
                    segmentador.estado,
                    segmentador.progresso,
                    segmentador.velocidade,
                )

                if candidatos:
                    ui.desenhar_candidatos(frame_bgr, candidatos, AJUDA)
                elif not deteccao.tem_corpo:
                    ui.desenhar_aviso(frame_bgr, "corpo fora de quadro — afaste-se")

                desenho.desenhar_fps(frame_bgr, fps)
                cv2.imshow("libras-live — dicionário", frame_bgr)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ESC:
                    break

                if tecla == TECLA_LIMPAR:
                    candidatos = []
                    segmentador.reiniciar()
                elif tecla in TECLAS_CANDIDATO and candidatos and ultima_consulta:
                    escolhido = tecla - ord("1")
                    if escolhido < len(candidatos):
                        rotulo = candidatos[escolhido].rotulo
                        dicionario.ancorar(ultima_consulta.vetores, rotulo)
                        aprendidos += 1
                        print(f"aprendido: {rotulo} (com a sua mão)")
                        candidatos = []
                elif tecla == TECLA_NOVO and ultima_consulta:
                    rotulo = _perguntar_rotulo()
                    if rotulo:
                        dicionario.ancorar(ultima_consulta.vetores, rotulo)
                        aprendidos += 1
                        print(f"aprendido: {rotulo} (sinal novo)")
                        candidatos = []
    except KeyboardInterrupt:
        print()  # o ^C fica na linha do terminal; a mensagem abaixo merece a sua
    finally:
        cv2.destroyAllWindows()

        if aprendidos:
            total = salvar_prototipos(dicionario)
            print(f"\n{aprendidos} sinais ensinados nesta sessão "
                  f"({total} protótipos seus em {config.PROTOTIPOS_USUARIO})")

    return codigo


def _perguntar_rotulo() -> str:
    """Pergunta o nome no terminal. Trava a janela por um instante, e tudo bem:
    digitar dentro de uma janela do OpenCV custaria mais do que compra."""
    try:
        return input("nome do sinal: ").strip().upper()
    except (EOFError, KeyboardInterrupt):
        return ""


def main(argv: list[str] | None = None) -> int:
    analisador = argparse.ArgumentParser(
        prog="python -m libras.app --sinais",
        description="Dicionário reverso de sinais de Libras.",
    )
    analisador.parse_args(argv)
    return executar()


if __name__ == "__main__":
    raise SystemExit(main())
