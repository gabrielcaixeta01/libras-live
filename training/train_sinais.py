"""Treina o encoder neural dos sinais e mede, no mesmo protocolo, se ele paga o torch.

    python training/train_sinais.py --treino-nucleo     # o que o app usa
    python training/train_sinais.py                    # vocabulário inteiro
    python training/train_sinais.py --epocas 80        # mais treino
    python training/train_sinais.py --limite-sinais 40 # ensaio rápido
    python training/train_sinais.py --treino-nucleo --so-rodizios --sem-maos  # ablação

Entra o dicionário de protótipos já extraído (`data/sinais/dicionario.npz`,
4.086 sequências de 32×147) e saem três coisas:

    models/encoder_sinais.pt            os pesos
    data/sinais/dicionario_embeddings.npz  os mesmos protótipos, como embedding
    models/relatorio_encoder.txt        o número, e o veredito sobre o torch

**Os vídeos não são reprocessados.** A extração custou 55 minutos e o resultado
está salvo; o encoder trabalha sobre ele.

O protocolo é o que torna o número comparável: **treina com dois articuladores e
valida no terceiro, girando**. Treinar com os três e avaliar depois mediria
memorização de pessoa e reproduziria, num lugar novo, o erro do L↔G da fase 1.
Nenhuma decisão de treino é tomada olhando o rodízio de fora — não há early
stopping nem escolha de época; o que o relatório mostra é a época final.

Uma segunda pergunta vem de graça: **vocabulário aberto**. Duzentos sinais ficam
fora do treino do encoder e entram só como protótipo. É o que mede a promessa
"adiciono um sinal novo com uma gravação" — sem isso ela é conversa.

**`--treino-nucleo` é o modo que o app usa.** Ele treina e mede nos 163 sinais de
`libras.nucleo` em vez das 1.363 classes, e foi medido em 37,3% de recall@5
contra 29,8% de um encoder treinado no vocabulário inteiro e consultado no mesmo
núcleo: concentrar a capacidade nas classes que serão respondidas venceu ter 8x
mais gravações. O índice em disco continua com os 4.086 protótipos, para que
`--sinais --tudo` não precise de um segundo treino.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libras import avaliacao, catalogo, config, encoder, nucleo, rejeicao  # noqa: E402
from libras.dicionario import Dicionario  # noqa: E402

# Quanto o encoder precisa multiplicar a baseline para que o torch entre por
# mérito e não por otimismo. 1,5x é folga suficiente para não ser ruído de
# semente sobre 4.086 consultas.
MARGEM_CLARA = 1.5


@dataclass(frozen=True)
class Amostras:
    """Os protótipos extraídos, com tudo o que o rodízio precisa saber deles."""

    vetores: np.ndarray  # (N, T, 147)
    rotulos: list[str]
    fontes: list[str]

    @property
    def chaves(self) -> list[str]:
        """O rótulo sem acento e sem caixa — a classe de verdade. `Avó` e `Avô`
        colapsam aqui, e o dicionário aceita esse custo desde a extração."""
        return [catalogo.chave(r) for r in self.rotulos]

    def __len__(self) -> int:
        return len(self.vetores)


def carregar_amostras(caminho: Path) -> Amostras:
    dicionario = Dicionario.carregar(caminho)
    if dicionario.metrica != "dtw":
        raise ValueError(
            f"esperado o dicionário de sequências (métrica 'dtw'), "
            f"recebido métrica {dicionario.metrica!r} em {caminho}"
        )

    return Amostras(
        vetores=np.asarray(dicionario.representacoes, dtype=np.float32),
        rotulos=dicionario.rotulos,
        fontes=dicionario.procedencias,
    )


def limitar_sinais(amostras: Amostras, quantos: int) -> Amostras:
    """Só os `quantos` primeiros sinais em ordem alfabética. Serve para ensaio.

    Um recorte de vocabulário **não é comparável** com o número de 4.086
    consultas: recall sobe quando há menos candidatos para confundir. O relatório
    avisa quando roda assim.
    """
    chaves = amostras.chaves
    escolhidas = set(sorted(set(chaves))[:quantos])
    indices = [i for i, c in enumerate(chaves) if c in escolhidas]

    return Amostras(
        vetores=amostras.vetores[indices],
        rotulos=[amostras.rotulos[i] for i in indices],
        fontes=[amostras.fontes[i] for i in indices],
    )


def restringir_ao_nucleo(amostras: Amostras) -> Amostras:
    """Só as gravações do vocabulário núcleo. Serve para treinar nele.

    É diferente de `--nucleo`, que recorta o **índice** na hora de avaliar e
    deixa o treino ver o vocabulário inteiro. Qual dos dois é melhor não é
    óbvio: treinar só no núcleo concentra a capacidade em 163 classes, mas com
    2 gravações de cada; treinar em tudo dá 8x mais dado para aprender o que é
    trajetória de mão, e o vocabulário aberto sugere que isso transfere.
    """
    chaves = amostras.chaves
    indices = [i for i, c in enumerate(chaves) if c in nucleo.CHAVES]

    return Amostras(
        vetores=amostras.vetores[indices],
        rotulos=[amostras.rotulos[i] for i in indices],
        fontes=[amostras.fontes[i] for i in indices],
    )


def escolher_abertos(chaves: list[str], quantos: int) -> set[str]:
    """Os sinais que ficam fora do treino do encoder, espalhados pelo vocabulário.

    Espalhados e não os primeiros: pegar um bloco alfabético concentraria o teste
    numa vizinhança do vocabulário, e o passo fixo dá uma amostra reprodutível
    sem depender de semente.

    Só entram sinais com pelo menos duas gravações — um sinal com uma gravação
    não pode estar no índice e na consulta ao mesmo tempo, então ele não mediria
    nada.
    """
    if quantos <= 0:
        return set()

    elegiveis = sorted(c for c, n in Counter(chaves).items() if n >= 2)
    if quantos >= len(elegiveis):
        raise ValueError(
            f"vocabulário aberto de {quantos} sinais não deixa nada para treinar: "
            f"há {len(elegiveis)} sinais elegíveis"
        )

    passo = len(elegiveis) / quantos
    return {elegiveis[int(i * passo)] for i in range(quantos)}


# --- treino -----------------------------------------------------------------


def treinar(
    vetores: np.ndarray,
    alvos: np.ndarray,
    classes: int,
    hp: encoder.Hiperparametros,
    dispositivo: torch.device,
    epocas: int = config.ENC_EPOCAS,
    lote: int = config.ENC_LOTE,
    taxa: float = config.ENC_TAXA,
    margem: float = config.ARCFACE_MARGEM,
    aumentar: bool = True,
    oclusao: float = config.ENC_AUG_OCLUSAO,
    semente: int = 0,
    rotulo: str = "treino",
    inicial: encoder.Codificador | None = None,
) -> encoder.Codificador:
    """As sequências de treino → um encoder treinado. Nada mais entra aqui.

    A cabeça ArcFace é criada, usada e jogada fora: o que sobra é o encoder, que
    é a única parte que o app precisa. Um classificador de 1.163 saídas não teria
    o que fazer num dicionário onde você pode acrescentar um sinal novo.

    `inicial` continua de um encoder já treinado em vez de partir do zero. É o
    que permite pré-treinar no vocabulário inteiro e especializar no núcleo: a
    cabeça é sempre nova, porque o conjunto de classes mudou, mas os pesos que
    interessam — os que aprenderam o que é trajetória de mão — atravessam.
    """
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)

    modelo = (inicial if inicial is not None else encoder.Codificador(hp)).to(
        dispositivo
    )
    cabeca = encoder.ArcFace(hp.dimensao, classes).to(dispositivo)

    otimizador = torch.optim.AdamW(
        list(modelo.parameters()) + list(cabeca.parameters()),
        lr=taxa,
        weight_decay=config.ENC_DECAIMENTO,
    )
    agenda = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas)

    alvos_torch = torch.from_numpy(alvos.astype(np.int64))
    epocas_de_rampa = max(1, int(0.3 * epocas))
    inicio = time.time()

    modelo.train()
    for epoca in range(1, epocas + 1):
        # A margem sobe de 0 até o alvo no primeiro terço. Com 2 exemplos por
        # classe, começar com ela cheia deixa a perda num platô do qual ela não
        # sai — o gradiente empurra todos os embeddings para longe de todos os
        # centros antes de qualquer estrutura existir.
        margem_agora = margem * min(1.0, epoca / epocas_de_rampa)
        ordem = rng.permutation(len(vetores))
        soma, passos = 0.0, 0

        for comeco in range(0, len(ordem), lote):
            indices = ordem[comeco : comeco + lote]
            brutos = vetores[indices]

            if aumentar:
                brutos = np.stack(
                    [encoder.aumentar(v, rng, oclusao=oclusao) for v in brutos]
                )

            entrada = torch.from_numpy(
                np.ascontiguousarray(encoder.entrada_do_modelo(brutos, hp))
            ).to(dispositivo)
            alvo = alvos_torch[indices].to(dispositivo)

            logits = cabeca(modelo(entrada), alvo, margem_agora)
            perda = F.cross_entropy(logits, alvo, label_smoothing=0.1)

            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(
                list(modelo.parameters()) + list(cabeca.parameters()), 5.0
            )
            otimizador.step()

            soma += float(perda.detach())
            passos += 1

        agenda.step()
        _progresso(rotulo, epoca, epocas, soma / max(passos, 1), inicio)

    print()
    return modelo.eval()


def _progresso(rotulo: str, epoca: int, epocas: int, perda: float, inicio: float) -> None:
    decorrido = time.time() - inicio
    restante = (decorrido / epoca) * (epocas - epoca)
    print(
        f"\r{rotulo}: época {epoca}/{epocas} — perda {perda:.3f} — "
        f"faltam ~{restante / 60:.1f} min",
        end="",
        flush=True,
    )


# --- avaliação --------------------------------------------------------------


@dataclass(frozen=True)
class Rodizio:
    """O resultado de um rodízio: quem ficou de fora e como a busca se saiu."""

    fonte: str
    posicoes: list[int | None]
    posicoes_abertas: list[int | None]
    # A matéria-prima do limiar de rejeição: as duas melhores distâncias de cada
    # consulta, com "a resposta estava na lista?" ao lado. Ver `libras.rejeicao`.
    # Vazia quando o índice não foi recortado: sem vocabulário fora dele, não há
    # o que rejeitar.
    consultas: list[rejeicao.Consulta] = field(default_factory=list)

    @property
    def metricas(self) -> avaliacao.Metricas:
        return avaliacao.agregar(self.posicoes)

    @property
    def metricas_abertas(self) -> avaliacao.Metricas:
        return avaliacao.agregar(self.posicoes_abertas)


def avaliar(
    modelo: encoder.Codificador,
    amostras: Amostras,
    fonte_fora: str,
    abertos: set[str],
    dispositivo: torch.device,
    k: int = config.CANDIDATOS_NA_TELA,
    chaves_indice: frozenset[str] | None = None,
    tta: int = 0,
) -> Rodizio:
    """Codifica tudo, indexa com os outros articuladores e consulta com este.

    A busca é a mesma classe que o app usa, só com a métrica trocada para
    cosseno — é isso que garante que o número medido aqui é o número que a
    pessoa vai ver na tela.

    `chaves_indice` recorta o vocabulário **depois** do treino: é o que mede o
    app como ele é entregue, indexando o núcleo. O recorte vale para o índice e
    para as consultas, porque a pergunta que ele responde é "quando você faz um
    sinal que o dicionário promete, ele acha?".
    """
    embeddings = (
        encoder.codificar_medio(modelo, amostras.vetores, tta, dispositivo=dispositivo)
        if tta > 1
        else encoder.codificar(modelo, amostras.vetores, dispositivo)
    )
    completo = Dicionario(
        representacoes=embeddings,
        rotulos=amostras.rotulos,
        fontes=amostras.fontes,
        metrica="cosseno",
    )
    indexavel = (
        completo if chaves_indice is None else completo.restringir(chaves_indice)
    )

    base, consultas = indexavel.separar_fonte(fonte_fora)

    posicoes: list[int | None] = []
    posicoes_abertas: list[int | None] = []
    medidas: list[rejeicao.Consulta] = []

    for representacao, rotulo in consultas:
        candidatos = base.buscar(representacao, k=k)
        onde = avaliacao.posicao(candidatos, rotulo)
        posicoes.append(onde)
        if candidatos:
            medidas.append(_medir_consulta(candidatos, acertou=onde is not None))
        if catalogo.chave(rotulo) in abertos:
            posicoes_abertas.append(onde)

    medidas += _consultas_de_fora(completo, base, fonte_fora, chaves_indice, k)

    return Rodizio(fonte_fora, posicoes, posicoes_abertas, medidas)


def _medir_consulta(candidatos: list, acertou: bool) -> rejeicao.Consulta:
    """As duas melhores distâncias de uma busca, que é o que a calibração come."""
    return rejeicao.Consulta(
        primeira=candidatos[0].distancia,
        segunda=candidatos[1].distancia if len(candidatos) > 1 else math.inf,
        acertou=acertou,
    )


def _consultas_de_fora(
    completo: Dicionario,
    base: Dicionario,
    fonte_fora: str,
    chaves_indice: frozenset[str] | None,
    k: int,
) -> list[rejeicao.Consulta]:
    """As gravações do articulador de teste que o índice **não** contém.

    São 1.200 sinais que o app promete não conhecer, e nenhum deles tem resposta
    certa possível: toda lista devolvida aqui é uma lista errada. É a metade
    negativa da calibração do limiar, e sem ela o corte seria escolhido olhando
    só para consultas que tinham resposta.
    """
    if chaves_indice is None:
        return []

    de_fora = [
        representacao
        for representacao, rotulo, fonte in zip(
            completo.representacoes, completo.rotulos, completo.procedencias
        )
        if fonte == fonte_fora and catalogo.chave(rotulo) not in chaves_indice
    ]

    medidas = []
    for representacao in de_fora:
        candidatos = base.buscar(representacao, k=k)
        if candidatos:
            medidas.append(_medir_consulta(candidatos, acertou=False))
    return medidas


def pre_treinar(
    amostras_amplas: Amostras,
    fonte_fora: str | None,
    abertos: set[str],
    hp: encoder.Hiperparametros,
    dispositivo: torch.device,
    epocas: int,
    **treino,
) -> encoder.Codificador | None:
    """Um encoder aquecido no vocabulário inteiro, para depois especializar no núcleo.

    Treinar só nos 163 sinais do núcleo venceu treinar nos 1.363 (37,3% contra
    29,8%), mas as duas opções desperdiçam metade da resposta: a primeira joga
    fora 8x mais gravações do que usa, e a segunda gasta capacidade em classes
    que nunca vão ser respondidas. O que aprende "isto é uma trajetória de mão"
    não precisa saber quais palavras existem; o que separa 163 palavras precisa.

    O articulador de fora **também fica de fora daqui**. Pré-treinar com ele
    seria mostrar a pessoa de teste à rede por uma porta lateral, e o número
    deixaria de ser honesto.
    """
    if epocas <= 0:
        return None

    dentro = [
        i
        for i, f in enumerate(amostras_amplas.fontes)
        if fonte_fora is None or f != fonte_fora
    ]
    vetores, alvos, classes = _preparar_treino(amostras_amplas, dentro, abertos)

    print(f"\n  pré-treino: {len(vetores)} gravações, {classes} classes")
    return treinar(
        vetores,
        alvos,
        classes,
        hp,
        dispositivo,
        epocas=epocas,
        rotulo="  pré-treino",
        **treino,
    )


def rodiziar(
    amostras: Amostras,
    abertos: set[str],
    hp: encoder.Hiperparametros,
    dispositivo: torch.device,
    chaves_indice: frozenset[str] | None = None,
    amostras_amplas: Amostras | None = None,
    pre_epocas: int = 0,
    tta: int = 0,
    **treino,
) -> list[Rodizio]:
    """Um treino por articulador de fora. É o número honesto, e ele custa 3 treinos."""
    resultados = []

    for fonte in sorted(set(amostras.fontes)):
        dentro = [i for i, f in enumerate(amostras.fontes) if f != fonte]
        vetores, alvos, classes = _preparar_treino(amostras, dentro, abertos)

        print(
            f"\nrodízio sem {fonte}: {len(vetores)} gravações, {classes} classes"
            f"{f', {len(abertos)} sinais fora do treino' if abertos else ''}"
        )
        inicial = (
            pre_treinar(
                amostras_amplas,
                fonte,
                abertos,
                hp,
                dispositivo,
                pre_epocas,
                **{k: v for k, v in treino.items() if k != "epocas"},
            )
            if amostras_amplas is not None
            else None
        )
        modelo = treinar(
            vetores,
            alvos,
            classes,
            hp,
            dispositivo,
            rotulo=f"sem {fonte}",
            inicial=inicial,
            **treino,
        )

        rodizio = avaliar(
            modelo,
            amostras,
            fonte,
            abertos,
            dispositivo,
            chaves_indice=chaves_indice,
            tta=tta,
        )
        print(f"  {rodizio.metricas}")
        if abertos:
            print(f"  vocabulário aberto: {rodizio.metricas_abertas}")

        resultados.append(rodizio)

    return resultados


def _preparar_treino(
    amostras: Amostras, indices: list[int], abertos: set[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Separa o que a rede pode ver e mapeia as chaves para índices de classe.

    O que fica de fora, fica de fora de verdade: as gravações dos sinais do
    vocabulário aberto não entram nem como exemplo de outra classe. Elas voltam
    depois, só como protótipo no índice.
    """
    chaves = amostras.chaves
    usaveis = [i for i in indices if chaves[i] not in abertos]

    classes = sorted({chaves[i] for i in usaveis})
    indice_da_classe = {chave: i for i, chave in enumerate(classes)}

    return (
        amostras.vetores[usaveis],
        np.array([indice_da_classe[chaves[i]] for i in usaveis], dtype=np.int64),
        len(classes),
    )


# --- relatório --------------------------------------------------------------


def montar_relatorio(
    amostras: Amostras,
    rodizios: list[Rodizio],
    hp: encoder.Hiperparametros,
    abertos: set[str],
    epocas: int,
    dispositivo: torch.device,
    segundos: float,
    limite_sinais: int | None,
    no_nucleo: bool = False,
    treino_nucleo: bool = False,
) -> str:
    todas = [p for r in rodizios for p in r.posicoes]
    abertas = [p for r in rodizios for p in r.posicoes_abertas]
    total = avaliacao.agregar(todas)

    linhas = [
        "=" * 72,
        "ENCODER NEURAL DE SINAIS — leave-one-articulator-out",
        "=" * 72,
        "",
        f"arquitetura .............. GRU bidirecional {hp.camadas}x{hp.oculto}"
        f" → embedding {hp.dimensao}d",
        f"entrada .................. 32×{hp.dim_entrada}"
        f" ({'posição + velocidade' if hp.com_velocidade else 'posição'}"
        f"{', z-normalizada' if hp.z else ''})",
        f"perda .................... ArcFace (margem {config.ARCFACE_MARGEM}, "
        f"escala {config.ARCFACE_ESCALA})",
        f"épocas por rodízio ....... {epocas}",
        f"protótipos ............... {len(amostras)}",
        f"sinais distintos ......... {len(set(amostras.chaves))}",
        "vocabulário do índice .... "
        + (
            f"núcleo, {len(nucleo.CHAVES)} sinais"
            + (" (e o treino também)" if treino_nucleo else " (treino no vocabulário inteiro)")
            if no_nucleo
            else "inteiro"
        ),
        f"vocabulário aberto ....... {len(abertos)} sinais fora do treino",
        f"dispositivo .............. {dispositivo.type}",
        f"tempo total .............. {segundos / 60:.1f} min",
        "",
        "-" * 72,
        "RODÍZIOS",
        "-" * 72,
        "",
    ]

    for rodizio in rodizios:
        linhas.append(f"  sem {rodizio.fonte:<12s} {rodizio.metricas}")
    linhas += ["", f"  {'TOTAL':<16s} {total}"]

    if abertas:
        linhas += [
            "",
            "-" * 72,
            "VOCABULÁRIO ABERTO",
            "-" * 72,
            "",
            f"  {len(abertos)} sinais que o encoder nunca viu no treino, presentes",
            "  no índice apenas como protótipo:",
            "",
            f"  {'':<16s} {avaliacao.agregar(abertas)}",
            "",
            "  É a medida da promessa do dicionário: um sinal novo entra com uma",
            "  gravação e passa a ser encontrável, sem retreinar nada.",
        ]

    medidas = [c for r in rodizios for c in r.consultas]
    if medidas and no_nucleo:
        linhas += _secao_de_rejeicao(medidas)

    linhas += [
        "",
        "-" * 72,
        "VEREDITO SOBRE O TORCH",
        "-" * 72,
        "",
        f"  baseline DTW (mesmo protocolo) .... recall@5 {_baseline(no_nucleo):.1%}"
        + (f"  [núcleo, {len(nucleo.CHAVES)} sinais]" if no_nucleo else ""),
        f"  encoder .......................... recall@5 {total.recall_5:.1%}",
        "",
        _veredito(total, limite_sinais, no_nucleo),
        "",
        "-" * 72,
        "COMO LER",
        "-" * 72,
        "",
        "Os números acima vêm dos rodízios: três encoders, cada um treinado sem",
        "um articulador e consultado com ele. Nenhuma época foi escolhida olhando",
        "esse resultado.",
        "",
        "O modelo salvo é um quarto treino, com os três articuladores, porque é",
        "ele que vai ser usado no app — mas ele não tem número honesto próprio, e",
        "é por isso que o placar não é dele.",
        "",
        "Referência da área para busca em vocabulário grande: 50,8% de MRR em",
        "10.235 sinais (arXiv:2502.20171).",
        "",
    ]
    return "\n".join(linhas)


def _secao_de_rejeicao(medidas: list[rejeicao.Consulta]) -> list[str]:
    """Os dois critérios de recusa, medidos lado a lado nas mesmas consultas.

    A escolha entre eles não é de gosto: a coluna "invenção cortada" diz quanto
    cada um pega das consultas que não tinham resposta possível, na mesma
    cobertura de acertos. Quem pegar mais, ganha.
    """
    de_fora = sum(1 for c in medidas if not c.acertou)
    linhas = [
        "",
        "-" * 72,
        "LIMIAR DE REJEIÇÃO",
        "-" * 72,
        "",
        "  Quando o app deve dizer 'não reconheci' em vez de mostrar cinco",
        "  palavras. Sem limiar ele sempre responde, e com 163 sinais no índice",
        f"  quase tudo que existe está fora dele — as {de_fora} consultas sem",
        "  resposta possível abaixo são o caso que ele existe para pegar.",
        "",
        "  DISTÂNCIA: recusa quando o melhor candidato está longe.",
        "  MARGEM:    recusa quando o primeiro e o segundo estão empatados —",
        "             não 'parece com alguma coisa?' e sim 'parece com UMA?'.",
    ]

    escolhas: dict[str, rejeicao.Corte] = {}
    for criterio in ("distancia", "margem"):
        linhas += [
            "",
            f"  {criterio.upper()}",
            f"  {'cobertura':>10s}  {'corte':>7s}  {'listas com resposta':>19s}"
            f"  {'invenção cortada':>16s}",
        ]
        for alvo in (0.99, 0.95, 0.90, 0.80):
            opcao = rejeicao.calibrar(medidas, criterio, alvo)
            if opcao is None:
                continue
            marca = " <-" if abs(alvo - rejeicao.COBERTURA_MINIMA) < 1e-9 else ""
            if marca:
                escolhas[criterio] = opcao
            linhas.append(
                f"  {opcao.cobertura:>9.1%}  {opcao.limiar:>7.4f}"
                f"  {opcao.precisao:>19.1%}  {opcao.recusa_correta:>16.1%}{marca}"
            )

    if not escolhas:
        return linhas

    vencedor = max(escolhas.values(), key=lambda c: c.recusa_correta)
    linhas += [
        "",
        f"  Na cobertura de {rejeicao.COBERTURA_MINIMA:.0%}, quem corta mais invenção é a",
        f"  **{vencedor.criterio}**: {vencedor.recusa_correta:.1%} contra "
        f"{min(escolhas.values(), key=lambda c: c.recusa_correta).recusa_correta:.1%}.",
        "  A coluna que decide não é essa, e sim a cobertura: num dicionário uma",
        "  lista errada custa um olhar, e um acerto apagado custa a resposta.",
        "",
        "  Ponha em config:",
    ]
    for criterio, corte in escolhas.items():
        chave = "REJEICAO_COSSENO" if criterio == "distancia" else "REJEICAO_MARGEM"
        linhas.append(f"    {chave} = {corte.limiar:.4f}")

    return linhas


def _baseline(no_nucleo: bool) -> float:
    """O número que o encoder tem que bater, no mesmo vocabulário em que ele roda.

    Comparar o encoder no núcleo com a baseline de 1.363 sinais seria creditar ao
    torch um ganho que veio de recortar o vocabulário.
    """
    return (
        config.BASELINE_DTW_NUCLEO_RECALL5 if no_nucleo else config.BASELINE_DTW_RECALL5
    )


def _veredito(
    total: avaliacao.Metricas, limite_sinais: int | None, no_nucleo: bool = False
) -> str:
    if limite_sinais:
        return (
            f"  SEM VEREDITO: rodou com vocabulário recortado em {limite_sinais}\n"
            "  sinais. Menos candidatos é menos confusão, e o número não se\n"
            "  compara com a baseline de 1.363 sinais. Rode sem --limite-sinais."
        )

    baseline = _baseline(no_nucleo)
    if total.recall_5 <= baseline:
        return (
            "  O TORCH NÃO ENTRA. O encoder não bateu a baseline DTW, e uma\n"
            "  dependência desse tamanho precisa comprar alguma coisa. O que o\n"
            "  projeto ganha aqui é a resposta medida, não o modelo."
        )
    if total.recall_5 < baseline * MARGEM_CLARA:
        return (
            f"  DECISÃO DO DONO. Ganho real ({total.recall_5 / baseline:.1f}x sobre\n"
            "  a baseline) mas abaixo da margem clara de "
            f"{MARGEM_CLARA:.1f}x que justificaria\n"
            "  o torch sozinho. Vale tentar mais épocas ou mais aumento antes de\n"
            "  decidir."
        )
    return (
        f"  O TORCH ENTRA. {total.recall_5 / baseline:.1f}x a baseline DTW no mesmo\n"
        "  protocolo, com o mesmo dado e a mesma busca. A representação era o\n"
        "  gargalo, e aprender a distância foi o que o destravou."
    )


# --- CLI --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dicionario", type=Path, default=config.DICIONARIO_SINAIS)
    ap.add_argument("--modelo", type=Path, default=config.ENCODER_SINAIS)
    ap.add_argument("--embeddings", type=Path, default=config.DICIONARIO_EMBEDDINGS)
    ap.add_argument("--saida", type=Path, default=config.RELATORIO_ENCODER)
    ap.add_argument("--confiancas", type=Path, default=config.CONFIANCAS_REJEICAO)
    ap.add_argument("--epocas", type=int, default=config.ENC_EPOCAS)
    ap.add_argument("--lote", type=int, default=config.ENC_LOTE)
    ap.add_argument("--taxa", type=float, default=config.ENC_TAXA)
    ap.add_argument("--dropout", type=float, default=config.ENC_DROPOUT)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument(
        "--aberto",
        type=int,
        help=(
            "sinais fora do treino, medidos como vocabulário aberto (0 desliga). "
            f"Padrão: {config.VOCABULARIO_ABERTO}, e 0 com --treino-nucleo, onde "
            "há 163 classes e reservar 200 não deixaria nada para treinar"
        ),
    )
    ap.add_argument(
        "--limite-sinais",
        type=int,
        help="recorta o vocabulário para ensaiar rápido; invalida o veredito",
    )
    ap.add_argument(
        "--nucleo",
        action="store_true",
        help="avalia indexando só o vocabulário núcleo, que é o que o app usa",
    )
    ap.add_argument(
        "--treino-nucleo",
        action="store_true",
        help="além de indexar, treina só no núcleo; implica --nucleo",
    )
    ap.add_argument(
        "--dispositivo",
        choices=("mps", "cpu"),
        help="padrão: mps quando existe",
    )
    ap.add_argument(
        "--z",
        action="store_true",
        help="z-normaliza a posição na entrada; medido em 7,1%% contra 10,3%% sem",
    )
    ap.add_argument("--sem-velocidade", action="store_true", help="só posição na entrada")
    ap.add_argument("--sem-aumento", action="store_true", help="desliga o aumento de dados")
    ap.add_argument(
        "--oclusao",
        type=float,
        default=config.ENC_AUG_OCLUSAO,
        help="chance de apagar uma mão por um trecho no aumento; 0 desliga",
    )
    ap.add_argument(
        "--sem-maos",
        action="store_true",
        help="tira os canais de configuração de mão da entrada (ablação)",
    )
    ap.add_argument(
        "--pre-epocas",
        type=int,
        default=0,
        help=(
            "pré-treina no vocabulário inteiro por N épocas antes de especializar "
            "no núcleo; 0 desliga. Medido em 42,9%% contra 46,7%% sem: as 8x mais "
            "gravações atrapalham em vez de ajudar. Só faz sentido com --treino-nucleo"
        ),
    )
    ap.add_argument(
        "--tta",
        type=int,
        default=config.ENC_TTA,
        help="versões aumentadas por gravação na hora de codificar; 1 desliga",
    )
    ap.add_argument(
        "--sem-rodizios",
        action="store_true",
        help="pula a medição honesta e só treina o modelo final",
    )
    ap.add_argument(
        "--so-rodizios",
        action="store_true",
        help="só mede: não treina o modelo final e não grava nada (ablação)",
    )
    args = ap.parse_args()

    if not args.dicionario.exists():
        print(
            f"dicionário não encontrado em {args.dicionario}\n"
            "Rode antes: python training/prepare_sinais.py --videos <raiz>"
        )
        raise SystemExit(1)

    amostras = carregar_amostras(args.dicionario)
    if args.limite_sinais:
        amostras = limitar_sinais(amostras, args.limite_sinais)

    no_nucleo = args.nucleo or args.treino_nucleo
    if no_nucleo:
        faltando = nucleo.ausentes(amostras.rotulos)
        if faltando:
            print(
                f"{len(faltando)} sinais do núcleo não existem neste dicionário: "
                f"{', '.join(faltando)}",
                file=sys.stderr,
            )
            raise SystemExit(1)
    # O índice em disco guarda **todos** os protótipos, mesmo quando o treino vê
    # só o núcleo: é o que mantém `--sinais --tudo` funcionando sem um segundo
    # treino. O recorte do que o app indexa acontece na hora de carregar.
    todas_as_amostras = amostras
    if args.treino_nucleo:
        amostras = restringir_ao_nucleo(amostras)

    hp = encoder.Hiperparametros(
        dropout=args.dropout,
        com_velocidade=not args.sem_velocidade,
        z=args.z,
        com_maos=config.ENC_MAOS_LOCAIS and not args.sem_maos,
    )
    dispositivo = (
        torch.device(args.dispositivo)
        if args.dispositivo
        else encoder.dispositivo_padrao()
    )
    treino = {
        "epocas": args.epocas,
        "lote": args.lote,
        "taxa": args.taxa,
        "aumentar": not args.sem_aumento,
        "oclusao": args.oclusao,
        "semente": args.semente,
    }

    print(
        f"{len(amostras)} protótipos, {len(set(amostras.chaves))} sinais, "
        f"articuladores {', '.join(sorted(set(amostras.fontes)))}"
    )
    print(f"entrada 32×{hp.dim_entrada} → embedding {hp.dimensao}d em {dispositivo.type}")

    # Com --treino-nucleo há 163 classes: reservar as 200 do padrão não deixaria
    # nada para treinar, e o script morria pedindo um --aberto 0 que ele mesmo
    # podia ter deduzido.
    quantos_abertos = (
        args.aberto
        if args.aberto is not None
        else (0 if args.treino_nucleo else config.VOCABULARIO_ABERTO)
    )
    abertos = escolher_abertos(amostras.chaves, quantos_abertos)
    chaves_indice = nucleo.CHAVES if no_nucleo else None
    inicio = time.time()

    rodizios = (
        []
        if args.sem_rodizios
        else rodiziar(
            amostras,
            abertos,
            hp,
            dispositivo,
            chaves_indice=chaves_indice,
            amostras_amplas=todas_as_amostras if args.pre_epocas > 0 else None,
            pre_epocas=args.pre_epocas,
            tta=args.tta,
            **treino,
        )
    )

    if args.so_rodizios:
        # Ablação: o número honesto é o dos rodízios, e treinar o modelo final
        # custaria um quarto treino que ninguém vai usar. Nada é gravado — nem
        # o modelo, nem o índice, nem o relatório —, para que uma experiência
        # não substitua em silêncio o que o app está usando.
        todas = [p for r in rodizios for p in r.posicoes]
        print(f"\n{'ABLAÇÃO':<16s} {avaliacao.agregar(todas)}")
        return

    # O modelo que vai ser usado vê os três articuladores e todas as classes: no
    # app não existe articulador de fora, e deixar dado no chão só para manter
    # simetria com a medição seria desperdício. O placar continua sendo o dos
    # rodízios.
    print(
        "\nmodelo final: todos os articuladores, "
        + ("vocabulário núcleo" if args.treino_nucleo else "vocabulário inteiro")
    )
    vetores, alvos, classes = _preparar_treino(
        amostras, list(range(len(amostras))), abertos=set()
    )
    final = treinar(
        vetores,
        alvos,
        classes,
        hp,
        dispositivo,
        rotulo="final",
        inicial=pre_treinar(
            todas_as_amostras,
            None,
            set(),
            hp,
            dispositivo,
            args.pre_epocas,
            **{k: v for k, v in treino.items() if k != "epocas"},
        )
        if args.pre_epocas > 0
        else None,
        **treino,
    )

    encoder.salvar(final, args.modelo)
    Dicionario(
        representacoes=encoder.codificar_medio(
            final, todas_as_amostras.vetores, args.tta, dispositivo=dispositivo
        ),
        rotulos=todas_as_amostras.rotulos,
        fontes=todas_as_amostras.fontes,
        metrica="cosseno",
    ).salvar(args.embeddings)

    segundos = time.time() - inicio

    medidas = [c for r in rodizios for c in r.consultas]
    if medidas:
        args.confiancas.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.confiancas,
            distancias=np.array([c.primeira for c in medidas], dtype=np.float32),
            segundas=np.array([c.segunda for c in medidas], dtype=np.float32),
            acertou=np.array([c.acertou for c in medidas], dtype=bool),
        )
        print(f"confianças dos rodízios salvas em {args.confiancas}")

    if rodizios:
        relatorio = montar_relatorio(
            amostras,
            rodizios,
            hp,
            abertos,
            args.epocas,
            dispositivo,
            segundos,
            args.limite_sinais,
            no_nucleo=no_nucleo,
            treino_nucleo=args.treino_nucleo,
        )
        print("\n" + relatorio)
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(relatorio, encoding="utf-8")
        print(f"relatório salvo em {args.saida}")
    else:
        print(
            "\nsem rodízios: nenhum número honesto foi medido, e o relatório não "
            "foi escrito"
        )

    print(f"modelo salvo em {args.modelo}")
    print(f"dicionário de embeddings salvo em {args.embeddings}")
    print("\no app usa o encoder automaticamente: python -m libras.app --sinais")


if __name__ == "__main__":
    main()
