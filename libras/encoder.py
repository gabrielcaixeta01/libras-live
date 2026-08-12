"""O encoder neural: a mesma palavra, feita por pessoas diferentes, no mesmo lugar.

A baseline DTW mede **7,5% de recall@5** em leave-one-articulator-out, e a
verificação mostrou que o problema não é a busca: o mesmo sinal feito por dois
articuladores fica só 20% mais perto que dois sinais quaisquer. Com 1.363
classes, 20% de margem produz exatamente aquele número. É uma propriedade da
*representação*, e nenhuma métrica melhor por cima dela conserta isso.

Daí este módulo. A distância deixa de ser uma regra fixa e passa a ser
**aprendida com o objetivo certo**: aproximar o mesmo sinal feito por pessoas
diferentes e afastar sinais diferentes. É a razão 0,80 atacada de frente.

    (32, 147) sequência → GRU bidirecional → (256,) embedding L2 → cosseno

**Por que GRU e não Transformer.** Trinta e dois frames por sinal e duas
gravações por classe no treino. Atenção brilha em sequência longa e dado
abundante; aqui não há nem um nem outro, e a recorrência já carrega o viés
indutivo de "isto é uma trajetória no tempo" de graça. O Transformer teria que
aprender essa noção com o dado que falta.

**Por que ArcFace e não triplet.** Triplet precisa de mineração de negativos
difíceis para não estagnar, e com 2 exemplos por classe a mineração é frágil.
ArcFace transforma o problema num classificador com margem angular sobre as
1.363 classes, treina estável, e o que sobra dele — o embedding L2 — é
exatamente o que a busca por cosseno consome. O classificador é o andaime; o
encoder é o produto.

**O que este módulo não decide.** Ele não sabe de rodízio, de vocabulário aberto
nem de veredito: quem treina é `training/train_sinais.py`, porque é lá que o
protocolo honesto mora. Aqui fica a arquitetura, o aumento de dados e a
codificação — as partes que o app também precisa em runtime.

torch entra por aqui e só por aqui. Ninguém no caminho do alfabeto importa este
módulo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch import nn
from torch.nn import functional as F

from . import config, pose, sequencia


@dataclass(frozen=True)
class Hiperparametros:
    """A arquitetura, num objeto que viaja junto com os pesos.

    Fica guardado no `.pt` porque o embedding só significa alguma coisa se a
    entrada for montada do mesmo jeito no treino e no app. Um `com_velocidade`
    divergente entre os dois não daria erro — daria um dicionário silenciosamente
    inútil, que é pior.
    """

    dimensao: int = config.DIMENSAO_EMBEDDING
    oculto: int = config.ENC_OCULTO
    camadas: int = config.ENC_CAMADAS
    dropout: float = config.ENC_DROPOUT
    com_velocidade: bool = True
    z: bool = False

    @property
    def dim_entrada(self) -> int:
        return pose.TAMANHO_VETOR * (2 if self.com_velocidade else 1)


class Codificador(nn.Module):
    """Sequência (T, 147) → embedding (256,) na esfera unitária.

    A leitura da GRU é **média e máximo juntos** sobre o tempo, não o último
    estado. O último estado favorece o fim do sinal, e em Libras a informação
    está espalhada: a configuração de mão aparece no começo, o movimento no meio,
    a localização em toda parte. A média descreve o gesto como um todo e o máximo
    guarda o instante mais marcante dele.

    A saída é L2-normalizada porque a busca é por cosseno: sem isso, um protótipo
    com norma grande ganharia consultas por tamanho e não por forma.
    """

    def __init__(self, hp: Hiperparametros | None = None):
        super().__init__()
        self.hp = hp or Hiperparametros()

        self.entrada = nn.LayerNorm(self.hp.dim_entrada)
        self.gru = nn.GRU(
            input_size=self.hp.dim_entrada,
            hidden_size=self.hp.oculto,
            num_layers=self.hp.camadas,
            batch_first=True,
            bidirectional=True,
            dropout=self.hp.dropout if self.hp.camadas > 1 else 0.0,
        )
        self.dropout = nn.Dropout(self.hp.dropout)
        self.projecao = nn.Linear(4 * self.hp.oculto, self.hp.dimensao)

    def forward(self, entrada: torch.Tensor) -> torch.Tensor:
        """(B, T, dim_entrada) → (B, dimensao), com norma 1."""
        saida, _ = self.gru(self.entrada(entrada))
        resumo = torch.cat([saida.mean(dim=1), saida.max(dim=1).values], dim=1)
        return F.normalize(self.projecao(self.dropout(resumo)), dim=1)


class ArcFace(nn.Module):
    """A cabeça de treino: classificação com margem angular. Descartada depois.

    Um classificador comum aprende a separar as classes e para aí — fronteiras
    justas o suficiente para acertar o treino não deixam folga para uma pessoa
    nova. A margem exige que o embedding fique `margem` radianos *dentro* do seu
    setor, e é essa folga que sobra como generalização entre articuladores.

    Escala e margem são o par que decide a dificuldade. A margem entra por
    parâmetro no `forward` para poder subir devagar: começar com ela cheia sobre
    1.363 classes de 2 exemplos deixa a perda num platô do qual ela não sai.
    """

    def __init__(
        self,
        dimensao: int,
        classes: int,
        escala: float = config.ARCFACE_ESCALA,
    ):
        super().__init__()
        if classes < 2:
            raise ValueError(f"ArcFace precisa de >= 2 classes, recebido {classes}")

        self.peso = nn.Parameter(torch.empty(classes, dimensao))
        nn.init.xavier_uniform_(self.peso)
        self.escala = escala

    def forward(
        self, embeddings: torch.Tensor, alvos: torch.Tensor, margem: float
    ) -> torch.Tensor:
        """(B, D) + (B,) → logits (B, classes) para entropia cruzada."""
        cosseno = F.linear(embeddings, F.normalize(self.peso, dim=1))
        # acos explode a derivada nas pontas; o clamp é o que mantém o treino de
        # pé quando um embedding cai quase colado no seu centro de classe.
        cosseno = cosseno.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        angulo = torch.acos(cosseno)
        alvo = F.one_hot(alvos, num_classes=self.peso.shape[0]).to(angulo.dtype)
        return torch.cos(angulo + margem * alvo) * self.escala


# --- aumento de dados -------------------------------------------------------
#
# Duas gravações por classe no treino de cada rodízio. Sem aumento, a rede tem
# 2.724 amostras para 1.163 classes e decora as pessoas — que é literalmente o
# problema que ela existe para resolver.


def aumentar(
    vetores: np.ndarray,
    rng: np.random.Generator,
    rotacao_graus: float = config.ENC_AUG_ROTACAO_GRAUS,
    escala: float = config.ENC_AUG_ESCALA,
    ruido: float = config.ENC_AUG_RUIDO,
    tempo: float = config.ENC_AUG_TEMPO,
) -> np.ndarray:
    """Uma variação plausível da mesma gravação. (T, 147) → (T, 147).

    Quatro transformações, todas escolhidas por corresponderem a algo que muda
    de verdade entre duas gravações do mesmo sinal: o ângulo da câmera, o
    tamanho aparente da pessoa, o erro do landmark e o ritmo.

    **Sem espelhamento**, de propósito. No alfabeto trocar de mão dá a mesma
    letra; aqui o lado é fonema, e espelhar geraria um sinal diferente com o
    rótulo antigo. A `--canhoto` do `prepare_sinais` existe para canonizar
    lateralidade uma vez, não para inventá-la aqui.
    """
    pontos = np.asarray(vetores, dtype=np.float32).reshape(
        len(vetores), pose.NUM_PONTOS, 3
    )

    girado = pontos @ _rotacao(
        rng.uniform(-rotacao_graus, rotacao_graus),
        rng.uniform(-rotacao_graus, rotacao_graus),
    ).T
    girado *= 1.0 + rng.uniform(-escala, escala)

    if ruido > 0:
        girado += rng.normal(0.0, ruido, size=girado.shape).astype(np.float32)

    return _deformar_tempo(girado, rng, tempo).reshape(len(vetores), -1)


def _rotacao(graus_y: float, graus_z: float) -> np.ndarray:
    """Giro em torno do eixo vertical e do eixo da câmera, nesta ordem.

    Só estes dois. A origem é o meio dos ombros, então o giro em Y é a pessoa
    virando de lado para a câmera e o giro em Z é a câmera torta — as duas
    variações que aparecem em gravação real. Inclinar em X moveria o corpo para
    frente e para trás, o que o z fraco do MediaPipe não sustenta.
    """
    ay, az = np.radians([graus_y, graus_z])
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry


def _deformar_tempo(
    pontos: np.ndarray, rng: np.random.Generator, forca: float
) -> np.ndarray:
    """Reamostra o tempo por uma grade irregular: parte do sinal acelera, parte freia.

    O começo e o fim ficam onde estão. Um sinal é delimitado pelas suas pontas —
    a mesma razão pela qual `sequencia.reamostrar` as preserva — e deslocá-las
    seria recortar o gesto, não variá-lo.

    É o aumento que corresponde ao que o DTW já fazia de graça: a mesma palavra
    dita devagar e rápido. A rede não tem alinhamento temporal embutido, então
    ela precisa ver essa variação para aprender a ignorá-la.
    """
    t = len(pontos)
    if forca <= 0 or t < 3:
        return pontos

    passos = 1.0 + rng.uniform(-forca, forca, size=t - 1)
    grade = np.concatenate([[0.0], np.cumsum(passos)])
    grade /= grade[-1]

    destino = np.linspace(0.0, 1.0, t)
    deformado = interp1d(grade, pontos, axis=0, kind="linear")(destino)
    return deformado.astype(np.float32)


# --- codificação ------------------------------------------------------------


def dispositivo_padrao() -> torch.device:
    """MPS quando existe, CPU quando não. Sem CUDA — o projeto vive num laptop."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def entrada_do_modelo(vetores: np.ndarray, hp: Hiperparametros) -> np.ndarray:
    """(..., T, 147) → (..., T, dim_entrada), do jeito que este modelo espera."""
    return sequencia.caracteristicas(
        vetores, com_velocidade=hp.com_velocidade, z=hp.z
    )


@torch.no_grad()
def codificar(
    modelo: Codificador,
    vetores: np.ndarray,
    dispositivo: torch.device | None = None,
    lote: int = 256,
) -> np.ndarray:
    """(N, T, 147) ou (T, 147) → embeddings (N, 256) ou (256,), em numpy.

    Sempre em modo de avaliação: com dropout ligado, a mesma gravação daria
    embeddings diferentes a cada chamada e a re-ancoragem viraria loteria.

    `dispositivo` é onde a entrada é colocada, e o padrão é onde o modelo já
    está — quem passar outro precisa ter movido o modelo antes.
    """
    vetores = np.asarray(vetores, dtype=np.float32)
    unico = vetores.ndim == 2
    if unico:
        vetores = vetores[None]

    dispositivo = dispositivo or next(modelo.parameters()).device
    treinando = modelo.training
    modelo.eval()

    try:
        saidas = []
        for inicio in range(0, len(vetores), lote):
            pedaco = entrada_do_modelo(vetores[inicio : inicio + lote], modelo.hp)
            tensor = torch.from_numpy(np.ascontiguousarray(pedaco)).to(dispositivo)
            saidas.append(modelo(tensor).float().cpu().numpy())
    finally:
        modelo.train(treinando)

    embeddings = np.concatenate(saidas).astype(np.float32)
    return embeddings[0] if unico else embeddings


# --- persistência -----------------------------------------------------------


def salvar(modelo: Codificador, caminho: str | Path) -> None:
    """Pesos e hiperparâmetros no mesmo arquivo, por definição do problema."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"hiperparametros": asdict(modelo.hp), "pesos": modelo.state_dict()}, caminho
    )


def carregar(
    caminho: str | Path, dispositivo: torch.device | None = None
) -> Codificador:
    """Devolve o modelo em modo de avaliação, no dispositivo pedido."""
    dispositivo = dispositivo or dispositivo_padrao()
    dados = torch.load(Path(caminho), map_location=dispositivo, weights_only=True)

    modelo = Codificador(Hiperparametros(**dados["hiperparametros"]))
    modelo.load_state_dict(dados["pesos"])
    return modelo.to(dispositivo).eval()
