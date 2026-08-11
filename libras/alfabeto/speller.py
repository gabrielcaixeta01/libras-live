"""Acumula letras confirmadas em texto.

O tempo entra por parâmetro em vez de ser lido do relógio aqui dentro. Isso
mantém a classe pura e os testes determinísticos.
"""

from __future__ import annotations

from .. import config


class Soletrador:
    """Monta o texto a partir das letras confirmadas.

    Tirar a mão de cena por um tempo insere um espaço — é assim que você separa
    palavras sem tocar no teclado. Só um espaço por ausência, por mais longa que
    ela seja.
    """

    def __init__(self, segundos_para_espaco: float = config.SEGUNDOS_PARA_ESPACO):
        self.segundos_para_espaco = segundos_para_espaco
        self._texto: list[str] = []
        self._ausencia = 0.0
        self._espaco_emitido = False

    @property
    def texto(self) -> str:
        return "".join(self._texto)

    @property
    def palavra_atual(self) -> str:
        """Trecho depois do último espaço — a UI destaca isso."""
        return self.texto.rsplit(" ", 1)[-1]

    def adicionar(self, letra: str) -> None:
        self._texto.append(letra)
        self._ausencia = 0.0
        self._espaco_emitido = False

    def registrar_presenca(self) -> None:
        """Há mão no frame: zera o contador de ausência."""
        self._ausencia = 0.0
        self._espaco_emitido = False

    def registrar_ausencia(self, delta_segundos: float) -> bool:
        """Não há mão no frame. Retorna True se isto inseriu um espaço."""
        self._ausencia += delta_segundos

        if self._espaco_emitido or self._ausencia < self.segundos_para_espaco:
            return False

        self._espaco_emitido = True

        # Nada a separar ainda, ou já termina em espaço.
        if not self._texto or self._texto[-1] == " ":
            return False

        self._texto.append(" ")
        return True

    def apagar(self) -> None:
        if self._texto:
            self._texto.pop()

    def limpar(self) -> None:
        self._texto.clear()
        self._ausencia = 0.0
        self._espaco_emitido = False
