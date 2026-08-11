"""Grava suas próprias amostras de uma letra pela webcam.

Cada letra vira um arquivo em `data/coletados/<LETRA>.npy`, então dá para parar
no meio e continuar depois — o script pula o que já existe.

    python training/collect.py               # tudo que ainda falta
    python training/collect.py F G H         # só essas
    python training/collect.py --refazer K Q # regrava (as apertadas demais)
    python training/collect.py --revisar     # só mostra o diagnóstico e sai

Nada de imagem sai daqui: cada frame vira 21 pontos e é descartado. O arquivo
salvo tem só os 63 floats por amostra.

**Amostra repetida não conta.** Uma amostra só é aceita se for diferente o
bastante de todas as que já entraram, então mexer a mão não é conselho, é o que
faz a barra andar: gire o pulso, aproxime e afaste, incline. Se a barra travar,
é a mão que está parada.
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
from libras.sampling import ColetorDiverso

DIR_SAIDA = config.DIR_DADOS / "coletados"
ESC = 27
ESPACO = 32


def _novo_coletor() -> ColetorDiverso:
    return ColetorDiverso(
        total=config.AMOSTRAS_POR_LETRA,
        distancia_minima=config.DISTANCIA_MINIMA_AMOSTRA,
        paciencia=config.PACIENCIA_COLETA,
        decaimento=config.DECAIMENTO_COLETA,
    )


def dispersao(amostras: np.ndarray) -> float:
    """Raio médio da nuvem — mesma medida usada durante a gravação."""
    if len(amostras) < 2:
        return 0.0
    return float(np.linalg.norm(amostras - amostras.mean(0), axis=1).mean())


def diagnosticar(letras: list[str] | None = None) -> list[str]:
    """Mostra a dispersão do que já está gravado. Devolve as letras apertadas."""
    arquivos = sorted(DIR_SAIDA.glob("*.npy"))
    if letras:
        arquivos = [p for p in arquivos if p.stem in letras]
    if not arquivos:
        return []

    print("Gravações existentes (raio da nuvem, maior = mais variado):")
    apertadas = []
    for caminho in arquivos:
        amostras = np.load(caminho)
        raio = dispersao(amostras)
        if raio < config.DISPERSAO_ALVO:
            apertadas.append(caminho.stem)
            aviso = "  <- apertada, vale regravar"
        else:
            aviso = ""
        print(f"  {caminho.stem}: {raio:.3f}  ({len(amostras)} amostras){aviso}")

    if apertadas:
        print(
            f"\nAbaixo de {config.DISPERSAO_ALVO} a nuvem é estreita demais: o modelo\n"
            "decora a pose exata em vez da letra. Regrave com:\n"
            f"  python training/collect.py --refazer {' '.join(apertadas)}"
        )
    print()
    return apertadas


# --- telas ---


def _cabecalho(frame, letra: str, restantes: int) -> None:
    texto = f"{letra}"
    tamanho = cv2.getTextSize(texto, ui.FONTE, 3.0, 6)[0]
    origem = ((frame.shape[1] - tamanho[0]) // 2, 90)
    cv2.putText(frame, texto, origem, ui.FONTE, 3.0, ui.VERDE, 6, cv2.LINE_AA)

    if restantes:
        cv2.putText(
            frame, f"faltam {restantes}", (30, 50), ui.FONTE, 0.7, ui.CINZA, 2,
            cv2.LINE_AA,
        )


def _barra(frame, progresso: float, y: int, cor) -> None:
    largura = frame.shape[1]
    x0, x1 = 60, largura - 60
    cv2.rectangle(frame, (x0, y), (x1, y + 14), ui.CINZA, 1)
    cv2.rectangle(
        frame, (x0, y), (x0 + int((x1 - x0) * min(progresso, 1.0)), y + 14), cor, -1
    )


def _medidor(frame, coletor: ColetorDiverso, segundos: float) -> None:
    """Contagem, barra de progresso e o medidor de variedade."""
    altura = frame.shape[0]

    cv2.putText(
        frame,
        f"{len(coletor.amostras)}/{coletor.total}",
        (60, 130),
        ui.FONTE,
        0.9,
        ui.BRANCO,
        2,
        cv2.LINE_AA,
    )
    _barra(frame, coletor.progresso, 145, ui.AZUL)

    alvo = config.DISPERSAO_ALVO
    saudavel = coletor.dispersao >= alvo
    cv2.putText(
        frame,
        f"variedade {coletor.dispersao:.2f} / {alvo:.2f}",
        (60, 195),
        ui.FONTE,
        0.7,
        ui.VERDE if saudavel else ui.AMARELO,
        2,
        cv2.LINE_AA,
    )
    _barra(frame, coletor.dispersao / alvo, 205, ui.VERDE if saudavel else ui.AMARELO)

    if coletor.parado(segundos):
        aviso = "MEXA A MAO - gire o pulso, aproxime, incline"
        tamanho = cv2.getTextSize(aviso, ui.FONTE, 0.9, 2)[0]
        cv2.putText(
            frame,
            aviso,
            ((frame.shape[1] - tamanho[0]) // 2, altura // 2),
            ui.FONTE,
            0.9,
            ui.AMARELO,
            2,
            cv2.LINE_AA,
        )


def _rodape(frame, texto: str) -> None:
    altura = frame.shape[0]
    cv2.putText(
        frame, texto, (30, altura - 30), ui.FONTE, 0.8, ui.BRANCO, 2, cv2.LINE_AA
    )


def _centralizado(frame, texto: str, y: int, escala: float, cor, espessura: int) -> None:
    tamanho = cv2.getTextSize(texto, ui.FONTE, escala, espessura)[0]
    cv2.putText(
        frame,
        texto,
        ((frame.shape[1] - tamanho[0]) // 2, y),
        ui.FONTE,
        escala,
        cor,
        espessura,
        cv2.LINE_AA,
    )


# --- laços ---


def esperar_inicio(
    camera: Camera, detector: DetectorMaos, letra: str, restantes: int, inicio: float
) -> bool:
    """Pausa entre letras. Devolve False se o usuário desistiu.

    Emendar as letras sem pausa foi o que produziu a primeira coleta corrida:
    aqui você começa quando estiver pronto, não quando o cronômetro mandar.
    """
    while True:
        leitura = camera.ler()
        if leitura is None:
            return False

        frame_bgr, frame_rgb = leitura
        agora = time.perf_counter()
        deteccao = detector.detectar(frame_rgb, int((agora - inicio) * 1000))
        if deteccao is not None:
            ui.desenhar_mao(frame_bgr, deteccao.pontos)

        _cabecalho(frame_bgr, letra, restantes)
        _centralizado(
            frame_bgr, "ESPACO para comecar", frame_bgr.shape[0] // 2, 1.2, ui.BRANCO, 3
        )
        _rodape(frame_bgr, "posicione a mao - ESPACO comeca - ESC sai")

        cv2.imshow("coleta", frame_bgr)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ESC:
            return False
        if tecla == ESPACO:
            return True


def gravar_letra(
    camera: Camera, detector: DetectorMaos, letra: str, restantes: int, inicio: float
) -> np.ndarray | None:
    """Grava as amostras de uma letra. Retorna None se o usuário abortou."""
    coletor = _novo_coletor()
    inicio_contagem = time.perf_counter()
    preparando = True

    while True:
        leitura = camera.ler()
        if leitura is None:
            return None

        frame_bgr, frame_rgb = leitura
        agora = time.perf_counter()
        deteccao = detector.detectar(frame_rgb, int((agora - inicio) * 1000))

        if deteccao is not None:
            ui.desenhar_mao(frame_bgr, deteccao.pontos)

        _cabecalho(frame_bgr, letra, restantes)

        if preparando:
            restante = config.SEGUNDOS_PREPARACAO - (agora - inicio_contagem)
            if restante <= 0:
                preparando = False
                inicio_gravacao = time.perf_counter()
            else:
                _centralizado(
                    frame_bgr, str(int(restante) + 1), frame_bgr.shape[0] // 2,
                    2.5, ui.VERDE, 5,
                )
                _rodape(frame_bgr, "ja vai - ESC cancela")
        else:
            decorrido = agora - inicio_gravacao
            if deteccao is not None:
                coletor.oferecer(deteccao.vetor, decorrido)

            _medidor(frame_bgr, coletor, decorrido)
            _rodape(frame_bgr, f"gravando {decorrido:.0f}s - amostra repetida nao conta")

            if coletor.completo:
                cv2.imshow("coleta", frame_bgr)
                cv2.waitKey(300)
                return coletor.amostras

        cv2.imshow("coleta", frame_bgr)
        if (cv2.waitKey(1) & 0xFF) == ESC:
            return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("letras", nargs="*", help="letras a gravar")
    parser.add_argument(
        "--refazer", action="store_true", help="regrava letras já existentes"
    )
    parser.add_argument(
        "--revisar",
        action="store_true",
        help="só mostra a variedade do que já foi gravado e sai",
    )
    args = parser.parse_args()

    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    pedidas = [l.upper() for l in args.letras]

    desconhecidas = sorted(set(pedidas) - set(config.ALFABETO))
    if desconhecidas:
        parser.error(f"letras fora do alfabeto: {' '.join(desconhecidas)}")

    diagnosticar(pedidas or None)
    if args.revisar:
        return 0

    # Sem argumento, o alvo é o alfabeto inteiro: a base pública cobre 15 letras
    # com mãos de outras pessoas, e faltam amostras da sua para todas elas.
    alvo = pedidas or config.ALFABETO
    pendentes = [
        letra for letra in alvo
        if args.refazer or not (DIR_SAIDA / f"{letra}.npy").exists()
    ]

    if not pendentes:
        print("Todas as letras pedidas já foram gravadas.")
        print(f"Use --refazer para regravar. Arquivos em {DIR_SAIDA}")
        return 0

    print(f"A gravar ({len(pendentes)}): {' '.join(pendentes)}")
    print(
        f"{config.AMOSTRAS_POR_LETRA} amostras distintas por letra.\n"
        "ESPACO comeca cada letra, ESC sai.\n"
    )

    try:
        camera = Camera()
    except CameraIndisponivel as erro:
        print(erro, file=sys.stderr)
        return 1

    inicio = time.perf_counter()
    gravadas_agora: list[str] = []

    with camera, DetectorMaos() as detector:
        for indice, letra in enumerate(pendentes):
            restantes = len(pendentes) - indice - 1

            if not esperar_inicio(camera, detector, letra, restantes, inicio):
                print("Cancelado.")
                break

            print(f"Letra {letra}...", end=" ", flush=True)
            amostras = gravar_letra(camera, detector, letra, restantes, inicio)

            if amostras is None:
                print("cancelado.")
                break

            np.save(DIR_SAIDA / f"{letra}.npy", amostras)
            gravadas_agora.append(letra)
            print(f"{len(amostras)} amostras, variedade {dispersao(amostras):.2f}")

    cv2.destroyAllWindows()

    if gravadas_agora:
        print()
        diagnosticar(gravadas_agora)

    prontas = sorted(p.stem for p in DIR_SAIDA.glob("*.npy"))
    faltam = sorted(set(config.ALFABETO) - set(prontas))
    print(f"Gravadas: {' '.join(prontas) or '(nenhuma)'}")
    if faltam:
        print(f"Ainda sem gravação sua: {' '.join(faltam)}")
    print("\nDepois de gravar, retreine: python training/train.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
