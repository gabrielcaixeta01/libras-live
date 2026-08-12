"""A extração dos vídeos: o que ela aceita, o que ela recusa, e o relógio.

O detector é falso de propósito. O que importa aqui não é o MediaPipe achar
mãos — é a extração sobreviver a um arquivo ruim no meio de 4.089, e o relógio
de timestamps continuar crescendo de um vídeo para o outro. Esse relógio já
derrubou a extração inteira uma vez: no modo VIDEO o MediaPipe exige
monotonicidade ao longo da vida do landmarker, não de cada arquivo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import prepare_sinais  # noqa: E402

from libras import sequencia  # noqa: E402

from . import apoio  # noqa: E402


class DetectorFalso:
    """Devolve sempre o mesmo frame e anota os timestamps que recebeu."""

    def __init__(self, frame: np.ndarray | None = None):
        self._frame = apoio.frame() if frame is None else frame
        self.marcas: list[int] = []

    def detectar(self, frame_rgb, timestamp_ms: int):
        self.marcas.append(timestamp_ms)
        return SimpleNamespace(frame=self._frame)


def escrever_video(caminho: Path, n_frames: int = 6) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(
        str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48)
    )
    for _ in range(n_frames):
        escritor.write(np.zeros((48, 64, 3), dtype=np.uint8))
    escritor.release()
    return caminho


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    escrever_video(tmp_path / "articulador_1" / "casa.mp4")
    return tmp_path


def test_video_bom_vira_resultado_aproveitavel(raiz: Path):
    video = raiz / "articulador_1" / "casa.mp4"

    resultado, _ = prepare_sinais.processar(
        video, raiz, passo=1, canhotos=set(), detector=DetectorFalso(), relogio_ms=0
    )

    assert resultado.problema is None
    assert resultado.rotulo == "CASA"
    assert resultado.articulador == "articulador_1"
    assert resultado.vetores.shape[0] == sequencia.T_PADRAO


def test_arquivo_ilegivel_vira_problema_e_nao_excecao(tmp_path: Path):
    quebrado = tmp_path / "articulador_1" / "nada.mp4"
    quebrado.parent.mkdir(parents=True)
    quebrado.write_bytes(b"isto nao e um video")

    resultado, _ = prepare_sinais.processar(
        quebrado, tmp_path, passo=1, canhotos=set(),
        detector=DetectorFalso(), relogio_ms=0,
    )

    assert resultado.vetores is None
    assert resultado.problema == "vídeo ilegível ou vazio"


def test_gravacao_sem_ombros_e_recusada_com_motivo(raiz: Path):
    """Sem os dois ombros não há âncora — recusar é melhor que normalizar no ar."""
    sem_corpo = apoio.frame(corpo=None)

    resultado, _ = prepare_sinais.processar(
        raiz / "articulador_1" / "casa.mp4", raiz, passo=1, canhotos=set(),
        detector=DetectorFalso(sem_corpo), relogio_ms=0,
    )

    assert resultado.vetores is None
    assert "ombros" in resultado.problema


def test_relogio_continua_crescendo_entre_videos(raiz: Path):
    """O segundo vídeo não pode reusar os timestamps do primeiro."""
    escrever_video(raiz / "articulador_1" / "agua.mp4")
    detector = DetectorFalso()

    _, relogio = prepare_sinais.processar(
        raiz / "articulador_1" / "casa.mp4", raiz, 1, set(), detector, 0
    )
    prepare_sinais.processar(
        raiz / "articulador_1" / "agua.mp4", raiz, 1, set(), detector, relogio
    )

    assert detector.marcas == sorted(set(detector.marcas))


def test_passo_pula_frames(raiz: Path):
    detector = DetectorFalso()

    prepare_sinais.processar(
        raiz / "articulador_1" / "casa.mp4", raiz, 2, set(), detector, 0
    )

    assert len(detector.marcas) == 3  # 6 frames, 1 a cada 2


def test_jobs_padrao_cabe_na_maquina():
    """Cada worker carrega dois modelos: o teto é a RAM, não os núcleos."""
    assert 1 <= prepare_sinais.jobs_padrao() <= 8


def test_parcial_sobrevive_ao_ida_e_volta(tmp_path: Path):
    """A retomada inteira depende disto: o que foi salvo volta igual.

    Se `feitos` voltasse diferente, a retomada reprocessaria vídeos já feitos
    (caro) ou pularia vídeos que faltam (silenciosamente incompleto).
    """
    parcial = tmp_path / "dic.parcial.npz"
    reps = [np.zeros((sequencia.T_PADRAO, 147), dtype=np.float32) for _ in range(3)]
    rotulos = ["CASA", "ÁGUA", "LIVRO"]
    fontes = ["articulador_1", "articulador_2", "articulador_1"]
    feitos = {"/a/casa.mp4", "/b/agua.mp4", "/a/livro.mp4"}

    prepare_sinais._salvar_parcial(parcial, reps, rotulos, fontes, feitos)
    voltou_reps, voltou_rotulos, voltou_fontes, voltou_feitos = prepare_sinais._retomar(
        parcial
    )

    assert voltou_rotulos == rotulos  # acento incluído
    assert voltou_fontes == fontes
    assert voltou_feitos == feitos
    assert len(voltou_reps) == 3


def test_retomar_sem_arquivo_comeca_do_zero(tmp_path: Path):
    reps, rotulos, fontes, feitos = prepare_sinais._retomar(tmp_path / "nao_existe.npz")

    assert (reps, rotulos, fontes, feitos) == ([], [], [], set())
