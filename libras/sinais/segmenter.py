"""Quando um sinal começa e quando ele acaba.

No alfabeto essa pergunta não existia: cada frame era uma letra, e o
estabilizador só decidia quando acreditar. Um sinal é um trecho — tem começo e
fim, e errar qualquer um dos dois muda a palavra.

A decisão é a mesma do "espaço por ausência" da fase 1, levada a sério: o app
continua sem exigir que você toque no teclado. **Repouso delimita.** A mão sai
do descanso, o sinal começa; a mão para, o sinal acaba e vai para a busca.

Três guardas contra disparo à toa, e cada uma existe por um motivo diferente:

- `frames_para_iniciar` — um tranco isolado não é uma consulta. Coçar o nariz
  passa por aqui e não vira busca.
- `segundos_minimo` — o que ficou curto demais foi ruído, não sinal. É
  descartado em silêncio, sem ocupar a tela com cinco palavras erradas.
- `segundos_maximo` — sem ele, alguém gesticulando numa conversa faria o
  segmentador gravar para sempre. Corta e busca.

A velocidade é medida em **larguras de ombro por segundo**, e não em pixels.
Isso é a mesma invariante da normalização, pela mesma razão: em pixels, chegar
perto da câmera dispararia sinais sozinho, e ficar longe deixaria o app surdo.

Lógica pura — o tempo entra por parâmetro, nunca de um relógio. É o que permite
testar um sinal de quatro segundos em microssegundos.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from . import pose


class Estado(Enum):
    REPOUSO = "repouso"
    SINALIZANDO = "sinalizando"


class Segmentador:
    """Recebe frames crus e devolve a gravação quando um sinal termina.

    Uso:

        resultado = segmentador.oferecer(frame, segundos)
        if resultado is not None:
            sequencia.preparar(resultado)  # ...e busca no dicionário

    Os frames entram **crus** (saída de `pose.montar_frame`, com NaN onde
    faltou) e saem crus. A normalização acontece aqui só para medir velocidade;
    quem prepara a sequência para valer é `sequencia.preparar`, depois.
    """

    def __init__(
        self,
        limiar_movimento: float,
        frames_para_iniciar: int,
        segundos_repouso: float,
        segundos_minimo: float,
        segundos_maximo: float,
    ):
        if limiar_movimento <= 0:
            raise ValueError("limiar_movimento deve ser > 0")
        if frames_para_iniciar < 1:
            raise ValueError("frames_para_iniciar deve ser >= 1")
        if segundos_repouso <= 0:
            raise ValueError("segundos_repouso deve ser > 0")
        if segundos_maximo <= segundos_minimo:
            raise ValueError("segundos_maximo deve ser > segundos_minimo")

        self.limiar_movimento = limiar_movimento
        self.frames_para_iniciar = frames_para_iniciar
        self.segundos_repouso = segundos_repouso
        self.segundos_minimo = segundos_minimo
        self.segundos_maximo = segundos_maximo

        self.reiniciar()

    # --- entrada ---

    def oferecer(self, frame: np.ndarray | None, segundos: float) -> np.ndarray | None:
        """Consome um frame. Devolve a gravação no instante em que o sinal fecha.

        Args:
            frame: (49, 3) cru, ou None quando não há ninguém no quadro.
            segundos: instante do frame.

        Returns:
            (T, 49, 3) quando um sinal acabou de terminar e é longo o bastante,
            None em todos os outros frames.
        """
        velocidade = self._medir(frame, segundos)
        self._velocidade = velocidade
        self._anterior = frame
        self._instante_anterior = segundos

        em_movimento = velocidade >= self.limiar_movimento

        if self._estado is Estado.REPOUSO:
            self._acumular_pre_buffer(frame, segundos, em_movimento)
            return None

        return self._continuar_sinal(frame, segundos, em_movimento)

    # --- repouso: esperando o sinal começar ---

    def _acumular_pre_buffer(self, frame, segundos, em_movimento: bool) -> None:
        """Guarda os últimos frames para não perder o começo do sinal.

        Quando o início é confirmado, o movimento já aconteceu há
        `frames_para_iniciar` frames. Sem esse buffer, todo sinal sairia
        decapitado — e o começo é onde muitos deles se distinguem.
        """
        if not em_movimento or frame is None:
            self._pre_buffer.clear()
            self._instantes_pre.clear()
            return

        self._pre_buffer.append(frame)
        self._instantes_pre.append(segundos)

        if len(self._pre_buffer) >= self.frames_para_iniciar:
            self._estado = Estado.SINALIZANDO
            self._buffer = list(self._pre_buffer)
            self._instantes = list(self._instantes_pre)
            self._ultimo_movimento = len(self._buffer) - 1
            self._pre_buffer.clear()
            self._instantes_pre.clear()

    # --- sinalizando: esperando o sinal acabar ---

    def _continuar_sinal(self, frame, segundos, em_movimento: bool):
        if frame is not None:
            self._buffer.append(frame)
            self._instantes.append(segundos)
            if em_movimento:
                self._ultimo_movimento = len(self._buffer) - 1

        parou_ha = segundos - self._instantes[self._ultimo_movimento]
        estourou = segundos - self._instantes[0] >= self.segundos_maximo

        if parou_ha >= self.segundos_repouso or estourou:
            return self._fechar()
        return None

    def _fechar(self) -> np.ndarray | None:
        """Apara o repouso do fim, decide se vale, e volta ao estado inicial."""
        fim = self._ultimo_movimento + 1
        gravacao = self._buffer[:fim]
        duracao = self._instantes[fim - 1] - self._instantes[0] if fim > 1 else 0.0

        self.reiniciar()

        if duracao < self.segundos_minimo:
            return None
        return np.stack(gravacao)

    # --- medição ---

    def _medir(self, frame: np.ndarray | None, segundos: float) -> float:
        """Deslocamento médio das mãos, em larguras de ombro por segundo.

        Frame sem os dois ombros não tem escala, então não tem velocidade
        confiável — vale zero. Inventar um número aqui faria o app disparar
        sozinho toda vez que a pose piscasse.
        """
        anterior = self._anterior
        dt = segundos - self._instante_anterior

        if frame is None or anterior is None or dt <= 0:
            return 0.0

        atual_n = pose.normalizar_frame(frame)
        anterior_n = pose.normalizar_frame(anterior)

        maos_atual = atual_n[: 2 * pose.NUM_PONTOS_MAO]
        maos_anterior = anterior_n[: 2 * pose.NUM_PONTOS_MAO]

        medidos = np.isfinite(maos_atual).all(1) & np.isfinite(maos_anterior).all(1)
        if not medidos.any():
            return 0.0

        deslocamento = np.linalg.norm(
            maos_atual[medidos] - maos_anterior[medidos], axis=1
        )
        return float(deslocamento.mean() / dt)

    # --- estado ---

    def reiniciar(self) -> None:
        self._estado = Estado.REPOUSO
        self._buffer: list[np.ndarray] = []
        self._instantes: list[float] = []
        self._pre_buffer: list[np.ndarray] = []
        self._instantes_pre: list[float] = []
        self._ultimo_movimento = 0
        self._anterior: np.ndarray | None = None
        self._instante_anterior = 0.0
        self._velocidade = 0.0

    @property
    def estado(self) -> Estado:
        return self._estado

    @property
    def velocidade(self) -> float:
        """Última velocidade medida, em larguras de ombro por segundo."""
        return self._velocidade

    @property
    def duracao(self) -> float:
        """Há quanto tempo o sinal em andamento começou. Zero em repouso."""
        if self._estado is Estado.REPOUSO or not self._instantes:
            return 0.0
        return self._instantes[-1] - self._instantes[0]

    @property
    def progresso(self) -> float:
        """Fração de `segundos_maximo` já gravada — para a barra na tela."""
        return min(self.duracao / self.segundos_maximo, 1.0)
