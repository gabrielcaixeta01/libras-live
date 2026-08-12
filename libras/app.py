"""O ponto de entrada: lê os argumentos e escolhe o modo.

    python -m libras.app              # soletrar: sua mão vira texto
    python -m libras.app --praticar   # praticar: ele pede a letra, você faz
    python -m libras.app --sinais     # dicionário: você sinaliza, ele traduz

Os dois primeiros são o alfabeto e moram em `libras/alfabeto/app.py`. O terceiro
é outro problema — trajetória, duas mãos, localização no corpo — e mora em
`app_sinais.py`. Cada um tem o seu laço; este arquivo não tem nenhum.

O loop escolhido é importado só na hora, porque cada um carrega modelos
diferentes do MediaPipe. Importar os dois no topo faria o modo alfabeto exigir
o `pose_landmarker.task`, que ele não usa.
"""

from __future__ import annotations

import argparse

from . import config


def main(argv: list[str] | None = None) -> int:
    analisador = argparse.ArgumentParser(
        prog="python -m libras.app",
        description="Reconhecimento de Libras em tempo real: alfabeto e sinais.",
    )
    analisador.add_argument(
        "--sinais",
        action="store_true",
        help="dicionário reverso de sinais: você sinaliza, ele mostra 5 candidatos",
    )
    analisador.add_argument(
        "--tudo",
        action="store_true",
        help=(
            "com --sinais: indexa as 1.364 palavras em vez do vocabulário núcleo. "
            "Mais cobertura, muito menos acerto"
        ),
    )
    analisador.add_argument(
        "--praticar",
        action="store_true",
        help="modo treino do alfabeto: o app pede uma letra e confere",
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
    if argumentos.sinais and argumentos.praticar:
        analisador.error("--sinais e --praticar são modos diferentes; escolha um")
    if argumentos.tudo and not argumentos.sinais:
        analisador.error("--tudo só vale com --sinais; o alfabeto tem 26 letras")

    if argumentos.sinais:
        from .app_sinais import executar

        return executar(apenas_nucleo=not argumentos.tudo)

    from .alfabeto.app import executar

    return executar(praticando=argumentos.praticar, rodadas=argumentos.rodadas)


if __name__ == "__main__":
    raise SystemExit(main())
