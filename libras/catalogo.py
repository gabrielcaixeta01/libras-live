"""De onde vêm o rótulo e o articulador de cada vídeo.

O V-LIBRASIL é distribuído em três downloads, um por articulador, e o nome do
sinal está no nome do arquivo. Isso torna a leitura do dataset uma questão de
interpretar caminhos — e interpretar caminhos erra em silêncio: um sufixo de
repetição não removido transforma `CASA` e `CASA_2` em dois sinais diferentes,
o vocabulário incha, e a avaliação despenca sem que nada pareça quebrado.

Por isso a regra mora aqui, separada da leitura de vídeo, e é testada.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

EXTENSOES = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# Sufixos de repetição: casa_2, casa-02, casa (3), casa_v2.
_REPETICAO = re.compile(r"[\s_\-]*(?:\(|\[)?(?:v|rep|take)?\s*\d{1,3}(?:\)|\])?$", re.I)
_SEPARADORES = re.compile(r"[_\-\.]+")
_ESPACOS = re.compile(r"\s+")


def rotulo_do_arquivo(caminho: str | Path) -> str:
    """Nome do arquivo → rótulo do sinal, em maiúsculas e sem sufixo de repetição.

    Acentos são preservados: PÃO e PAO são a mesma palavra para um leitor, mas
    quem escreve o rótulo na tela é este projeto, e escrever errado seria pior
    que agrupar errado. A comparação entre rótulos usa `chave`.
    """
    nome = Path(caminho).stem
    nome = _SEPARADORES.sub(" ", nome)
    nome = _REPETICAO.sub("", nome)
    return _ESPACOS.sub(" ", nome).strip().upper()


def chave(rotulo: str) -> str:
    """Forma canônica para comparar rótulos: sem acento, sem caixa, sem espaço extra.

    Serve para juntar `AMANHÃ` e `AMANHA` vindos de arquivos escritos por
    pessoas diferentes, sem que nenhum dos dois vire o rótulo exibido.
    """
    sem_acento = unicodedata.normalize("NFKD", rotulo)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return _ESPACOS.sub(" ", sem_acento).strip().upper()


def articulador_do_caminho(caminho: str | Path, raiz: str | Path) -> str:
    """A pasta logo abaixo da raiz — é assim que o V-LIBRASIL separa as pessoas.

    Vídeo solto na própria raiz não tem como dizer de quem é, e vira
    `desconhecido`. Isso não é um erro fatal: quem avalia com
    leave-one-articulator-out é que vai reclamar, e é lá que a reclamação
    significa alguma coisa.
    """
    caminho, raiz = Path(caminho).resolve(), Path(raiz).resolve()
    try:
        relativo = caminho.relative_to(raiz)
    except ValueError:
        return "desconhecido"

    return relativo.parts[0] if len(relativo.parts) > 1 else "desconhecido"


def listar_videos(raiz: str | Path) -> list[Path]:
    """Todos os vídeos sob a raiz, em ordem estável.

    Ordem estável importa: a extração é retomável, e retomar depende de o
    percurso ser o mesmo entre duas execuções.
    """
    raiz = Path(raiz)
    if not raiz.exists():
        raise FileNotFoundError(f"diretório não encontrado: {raiz}")

    return sorted(
        p for p in raiz.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSOES
    )
