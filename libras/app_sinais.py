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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config, desenho, nucleo, pose, sequencia, ui
from .camera import Camera, CameraIndisponivel
from .detector import DetectorSinais
from .dicionario import FONTE_USUARIO, Dicionario
from .segmenter import Estado, Segmentador

ESC = 27
TECLA_LIMPAR = ord("c")
TECLA_NOVO = ord("n")
TECLAS_CANDIDATO = tuple(ord(str(n)) for n in range(1, 10))

AJUDA = "ESC sair   1-5 corrigir e ensinar   N nomear   C limpar"


@dataclass(frozen=True)
class Busca:
    """O dicionário, o caminho de uma gravação até a representação dele, e onde
    guardar o que você ensinar.

    Os três andam juntos porque mudam juntos: com o encoder, a representação é um
    embedding de 256d, a métrica é cosseno e os seus protótipos vão para outro
    arquivo — um embedding não pode ser guardado no npz das sequências, e
    misturar os dois daria um dicionário que carrega e erra tudo.
    """

    dicionario: Dicionario
    representar: Callable[[sequencia.Sequencia], np.ndarray]
    prototipos: Path


def carregar_busca(apenas_nucleo: bool = True) -> Busca:
    """Prefere o encoder quando ele existe; cai na baseline DTW quando não.

    Quem rodou `training/train_sinais.py` ganha o encoder sem mexer em nada, e
    quem não rodou continua com o dicionário de sequências. O resto do loop não
    sabe qual dos dois está em uso.
    """
    if not config.DICIONARIO_EMBEDDINGS.exists():
        return _busca_dtw(apenas_nucleo)

    try:
        from . import encoder
    except ModuleNotFoundError:
        print(
            "encoder encontrado mas torch não está instalado — usando a baseline "
            "DTW.\nPara usar o encoder: pip install torch",
            file=sys.stderr,
        )
        return _busca_dtw(apenas_nucleo)

    if not config.ENCODER_SINAIS.exists():
        raise FileNotFoundError(
            f"dicionário de embeddings em {config.DICIONARIO_EMBEDDINGS} mas "
            f"nenhum modelo em {config.ENCODER_SINAIS}.\n"
            "Os dois são gerados juntos: python training/train_sinais.py"
        )

    modelo = encoder.carregar(config.ENCODER_SINAIS)

    # A primeira passada pelo MPS custa ~170 ms de inicialização e as seguintes
    # 9 ms. Sem isto, esses 170 ms cairiam no primeiro sinal que a pessoa fizer —
    # um engasgo visível num loop de 33 ms. Aqui, com a câmera ainda abrindo,
    # eles não custam nada.
    encoder.codificar(
        modelo,
        np.zeros((sequencia.T_PADRAO, pose.TAMANHO_VETOR), dtype=np.float32),
    )

    dicionario = carregar_dicionario(
        config.DICIONARIO_EMBEDDINGS,
        config.PROTOTIPOS_USUARIO_EMBEDDINGS,
        apenas_nucleo=apenas_nucleo,
    )

    return Busca(
        dicionario=dicionario,
        representar=lambda pronta: encoder.codificar(modelo, pronta.vetores),
        prototipos=config.PROTOTIPOS_USUARIO_EMBEDDINGS,
    )


def _busca_dtw(apenas_nucleo: bool = True) -> Busca:
    return Busca(
        dicionario=carregar_dicionario(apenas_nucleo=apenas_nucleo),
        representar=lambda pronta: pronta.vetores,
        prototipos=config.PROTOTIPOS_USUARIO,
    )


def carregar_dicionario(
    base: Path | None = None,
    meus: Path | None = None,
    apenas_nucleo: bool = True,
) -> Dicionario:
    """O dicionário base mais os seus protótipos, se existirem.

    O recorte do núcleo vale só para o que veio do V-LIBRASIL: o que você
    ensinou entra sempre, esteja ou não na lista curada. Ensinar um sinal que o
    núcleo não tem é justamente o caso em que o recorte não pode atrapalhar.
    """
    base = base or config.DICIONARIO_SINAIS
    meus = meus or config.PROTOTIPOS_USUARIO

    if not base.exists():
        raise FileNotFoundError(
            f"Dicionário de sinais não encontrado em {base}.\n"
            "Rode: python training/prepare_sinais.py --videos <raiz dos vídeos>\n"
            "A base V-LIBRASIL está em https://libras.cin.ufpe.br"
        )

    dicionario = Dicionario.carregar(base)
    if apenas_nucleo:
        dicionario = dicionario.restringir(nucleo.CHAVES)

    if meus.exists():
        seus = Dicionario.carregar(meus)
        dicionario = dicionario.juntar(seus)
        print(f"{len(seus)} protótipos seus carregados")

    return dicionario


def salvar_prototipos(dicionario: Dicionario, caminho: Path | None = None) -> int:
    """Grava só o que é seu. O dicionário base não é reescrito nunca."""
    meus = dicionario.apenas(FONTE_USUARIO)
    if len(meus) == 0:
        return 0

    meus.salvar(caminho or _prototipos_da_metrica(dicionario.metrica))
    return len(meus)


def _prototipos_da_metrica(metrica: str) -> Path:
    if metrica == "cosseno":
        return config.PROTOTIPOS_USUARIO_EMBEDDINGS
    return config.PROTOTIPOS_USUARIO


def executar(apenas_nucleo: bool = True) -> int:
    try:
        busca = carregar_busca(apenas_nucleo)
    except (FileNotFoundError, ValueError) as erro:
        print(erro, file=sys.stderr)
        return 1

    dicionario = busca.dicionario

    # O limiar foi calibrado com 163 sinais no índice. Com 1.364, o vizinho mais
    # próximo é sempre mais próximo — o mesmo corte passaria a aceitar quase
    # tudo, e um limiar frouxo é pior que nenhum: ele promete uma recusa que não
    # acontece. Em `--tudo` a rejeição fica desligada e assumida como desligada.
    limiar = config.limiar_de_rejeicao(dicionario.metrica) if apenas_nucleo else None

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
          f"({len(dicionario)} protótipos, métrica {dicionario.metrica}"
          f"{', vocabulário núcleo' if apenas_nucleo else ', vocabulário inteiro'})")
    print(AJUDA)

    inicio = time.perf_counter()
    instante_anterior = inicio
    fps = 0.0

    candidatos: list = []
    recusado = False             # a última gravação bateu longe demais de tudo
    ultima_consulta = None       # a representação da última gravação, para re-ancorar
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
                        ultima_consulta = busca.representar(
                            sequencia.preparar(gravacao)
                        )
                        candidatos = dicionario.buscar(
                            ultima_consulta, k=config.CANDIDATOS_NA_TELA
                        )
                        # Sem isto o dicionário sempre responde: um sinal que ele
                        # não tem devolve os cinco mais parecidos com cara de
                        # resposta. Com 163 sinais no índice, o que está fora é a
                        # maioria do que existe.
                        recusado = bool(candidatos) and (
                            limiar is not None and candidatos[0].distancia > limiar
                        )
                        if recusado:
                            candidatos = []
                        instante_resultado = agora
                    except ValueError:
                        # Gravação sem âncora em frame nenhum: nada a consultar.
                        ultima_consulta, candidatos, recusado = None, [], False

                if agora - instante_resultado > config.SEGUNDOS_MOSTRANDO:
                    candidatos = []
                    recusado = False

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
                elif recusado:
                    ui.desenhar_aviso(
                        frame_bgr, "não reconheci — N ensina este sinal"
                    )
                elif not deteccao.tem_corpo:
                    ui.desenhar_aviso(frame_bgr, "corpo fora de quadro — afaste-se")

                desenho.desenhar_fps(frame_bgr, fps)
                cv2.imshow("libras-live — dicionário", frame_bgr)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ESC:
                    break

                if tecla == TECLA_LIMPAR:
                    candidatos, recusado = [], False
                    segmentador.reiniciar()
                elif (
                    tecla in TECLAS_CANDIDATO
                    and candidatos
                    and ultima_consulta is not None
                ):
                    escolhido = tecla - ord("1")
                    if escolhido < len(candidatos):
                        rotulo = candidatos[escolhido].rotulo
                        dicionario.ancorar(ultima_consulta, rotulo)
                        aprendidos += 1
                        print(f"aprendido: {rotulo} (com a sua mão)")
                        candidatos = []
                elif tecla == TECLA_NOVO and ultima_consulta is not None:
                    rotulo = _perguntar_rotulo()
                    if rotulo:
                        dicionario.ancorar(ultima_consulta, rotulo)
                        aprendidos += 1
                        print(f"aprendido: {rotulo} (sinal novo)")
                        candidatos, recusado = [], False
    except KeyboardInterrupt:
        print()  # o ^C fica na linha do terminal; a mensagem abaixo merece a sua
    finally:
        cv2.destroyAllWindows()

        if aprendidos:
            total = salvar_prototipos(dicionario, busca.prototipos)
            print(f"\n{aprendidos} sinais ensinados nesta sessão "
                  f"({total} protótipos seus em {busca.prototipos})")

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
    analisador.add_argument(
        "--tudo",
        action="store_true",
        help=(
            "indexa as 1.364 palavras do V-LIBRASIL em vez do vocabulário núcleo. "
            "Mais cobertura e muito menos acerto: 10,3%% de recall@5 contra 37,3%%"
        ),
    )
    args = analisador.parse_args(argv)
    return executar(apenas_nucleo=not args.tudo)


if __name__ == "__main__":
    raise SystemExit(main())
