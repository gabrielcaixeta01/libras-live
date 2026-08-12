# Fase 2 — Dicionário reverso de sinais

**Data:** 2026-08-11
**Estado:** aprovado, em implementação

Reconhecer **sinais** (palavras) de Libras, não só letras. O produto é um
**dicionário reverso**: você faz um sinal, ele mostra os cinco candidatos mais
prováveis em português.

---

## O problema que isso resolve

Um dicionário de Libras é indexado por português. Se você vê um sinal que não
conhece, não tem como procurar: você não sabe o nome dele — é justamente o que
está tentando descobrir. Buscar pela *forma* é o uso que não tem solução boa
hoje.

Isso também é o que torna o top-5 honesto em vez de uma desculpa: num dicionário,
cinco candidatos é uma resposta boa. Num tradutor, é uma resposta errada.

---

## O dataset e a restrição que ele impõe

[V-LIBRASIL](https://libras.cin.ufpe.br/) (UFPE/CIn): **1.364 sinais, 4.089
vídeos, 1440×1080, chroma key, ~10,5 GB** divididos em três downloads.

O número que decide tudo:

```
4.089 vídeos ÷ 1.364 sinais = 3 vídeos por sinal — um por articulador
```

Três exemplos por classe não é "dataset pequeno", é **few-shot por definição**.
E o único protocolo de avaliação que não mente com três pessoas é
**leave-one-articulator-out**, que deixa **dois** exemplos de treino por sinal.

Consequências aceitas:

- Um classificador de 1.364 saídas não é viável. A arquitetura é
  **representação + busca vetorial**.
- Com 3 articuladores, o modelo tem todo incentivo para aprender *as pessoas* em
  vez dos sinais. É o mesmo mecanismo do L↔G da fase 1, agora com número. A
  mitigação está desenhada (ver *Re-ancoragem*), não deixada para depois.

Alternativas consideradas e por que não:

| alternativa | por que não |
|---|---|
| MINDS-Libras (12 sinalizantes) | 20 sinais só; ótimo para medir, insuficiente para o produto |
| Recortar para 50–150 sinais | joga fora 90% do dataset; deixa de ser vocabulário grande |
| Classificador plano de 1.364 classes | 3 amostras/classe; decoraria os articuladores |

---

## A inversão central

A fase 1 normaliza colocando **o pulso na origem**. Foi certo para o alfabeto e é
**errado para sinais**: em Libras a *localização* (testa, peito, queixo, ombro) é
fonema — é o que separa PAI de MÃE. A normalização atual apaga exatamente essa
informação.

| | alfabeto (fase 1) | sinais (fase 2) |
|---|---|---|
| origem | pulso | ponto médio dos ombros |
| escala | maior distância ao pulso | distância entre ombros |
| preserva | forma da mão | forma **+ localização + trajetória** |
| unidade | 1 frame | janela de T frames |
| mãos | 1 | 2 |

`landmarks.normalizar` fica **intacto**. O app do alfabeto continua funcionando
sem alteração; ele foi recolhido em `libras/alfabeto/` e os sinais ocupam o topo
do pacote.

---

## Representação

**Detecção:** `HandLandmarker(num_hands=2)` + `PoseLandmarker` da API Tasks. Não
o Holistic legado, removido no MediaPipe 1.0.

**Subconjunto de landmarks — 49 pontos, não 543.** O
[paper do IEEE SAS 2026](https://arxiv.org/abs/2510.24887) mostra que o
subconjunto certo iguala ou bate o SOTA em Libras isolada com **5x menos tempo**
que OpenPose, e que usar tudo *piora*. Aqui:

```
2 × 21 pontos das mãos  +  7 da pose (nariz, ombros, cotovelos, punhos) = 49
49 pontos × 3 coordenadas = 147 floats por frame
```

Rosto fora nesta fase. É fonema em Libras, mas 468 pontos de face sobre 3
amostras por classe é ruído garantido.

**Sequência (T, 49, 3) → representação**, nesta ordem:

1. **Imputação por spline** dos landmarks faltantes (mão sai de quadro, oclusão),
   em espaço bruto. O mesmo paper mostra ganho substancial só com isso. É o
   problema de mão fechada da fase 1 voltando em outra forma.
2. **Máscara de validade** (T, 49) preservada como feature separada — o modelo
   precisa saber o que foi medido e o que foi inventado.
3. **Normalização ancorada no corpo**, por frame.
4. **Reamostragem para T = 32 frames** por interpolação linear. Sinais têm
   durações diferentes; cortar perderia o fim dos longos.

---

## Arquitetura: representação + busca

```
vídeo/webcam → 49 landmarks/frame → imputar → normalizar no corpo
             → reamostrar p/ T=32 → representação → busca no dicionário → top-5
```

### A baseline vem antes do encoder

O encoder neural (GRU bidirecional ou Transformer pequeno → vetor 256d, treinado
com ArcFace sobre os 1.364 rótulos) **traz torch para o projeto**, que hoje é
scikit-learn puro com modelo de menos de 1MB. É a maior mudança de dependência
da história do repo.

Então, no espírito do resto do projeto: **DTW-1NN primeiro**. Alinhamento
temporal contra os 4.092 protótipos, numpy puro, zero dependência nova. Ela
produz o número que o encoder precisa bater.

**Se o encoder não bater o DTW por margem clara em leave-one-articulator-out, o
torch não entra.** A decisão fica registrada em `models/relatorio_sinais.txt`.

Para que a troca seja barata, `Dicionario` recebe a métrica por parâmetro:

| representação | métrica | dependência |
|---|---|---|
| sequência (32, 147) | DTW com banda de Sakoe-Chiba | numpy |
| embedding (256,) | cosseno | torch (só no treino) |

O app não sabe a diferença.

> **Correção de 12/08/2026, depois de construir o encoder.** A troca saiu barata
> como previsto — `Dicionario` não mudou —, mas "torch só no treino" estava
> errado: o app precisa do modelo para codificar a *consulta*, então torch entra
> em runtime também. Sem ele, `app_sinais.carregar_busca` avisa e volta para o
> DTW. Resultado medido em `models/relatorio_encoder.txt`.

### Custo do DTW

DTW ingênuo em 4.092 candidatos seria lento em Python. A implementação
**vetoriza sobre os candidatos**, não sobre o tempo: o laço da programação
dinâmica é sequencial em (i, j) — 32×32 = 1.024 iterações — mas cada iteração
opera num array de 4.092 posições.

**Medido: 87 ms** para a busca completa (N=4.092, T=32, D=147, Apple M5), com
34 MB de pico no tensor de custos. E é uma consulta por sinal, não por frame —
a busca não entra no orçamento dos 33 ms do loop de vídeo.

### O dicionário

4.092 vetores guardados **separados por articulador**, não a média. A variação
entre pessoas é sinal, não ruído — a média inventaria um articulador que não
existe. A busca devolve a melhor distância por rótulo, depois os 5 melhores
rótulos.

### Re-ancoragem — a mitigação do viés

Você grava um sinal **uma vez** e ele vira protótipo adicional no dicionário,
**sem retreinar nada**. É a mesma lição do L↔G da fase 1, resolvida por
arquitetura em vez de por mais coleta. Também é o que permite adicionar sinal
fora das 1.364 palavras.

Protótipos seus são marcados com a fonte `voce` e ficam em
`data/sinais/meus_prototipos.npz`, fora do dicionário base.

---

## Segmentação ao vivo

Automática, por repouso — o app continua sem tocar no teclado, como o "espaço por
ausência" da fase 1.

```
REPOUSO --(velocidade > limiar por K frames)--> SINALIZANDO
SINALIZANDO --(velocidade < limiar por SEGUNDOS_REPOUSO, ou mãos somem)--> busca
```

Guardas contra falso disparo:

- **duração mínima** (0,4s): coçar o nariz não é consulta;
- **duração máxima** (4,0s): corta e busca, em vez de gravar para sempre;
- velocidade medida em **larguras de ombro por segundo**, então é indiferente à
  distância da câmera — mesma invariante da normalização.

`Segmentador` é lógica pura: recebe pontos e um instante, devolve a sequência
quando um sinal termina. Sem relógio e sem câmera dentro.

---

## Avaliação

**Protocolo: leave-one-articulator-out.** Treina/indexa com dois articuladores,
consulta com o terceiro, roda os três rodízios. É o único protocolo que mede
generalização entre pessoas — e mede exatamente o viés que preocupa.

Métricas, todas por rodízio e agregadas:

| métrica | o que responde |
|---|---|
| recall@1 | acertou de primeira? |
| **recall@5** | **a resposta está na tela?** ← a métrica do produto |
| MRR | quão alto na lista? |

**Avaliação de vocabulário aberto:** 200 sinais são retirados do treino do
encoder e entram só como protótipos. É o que valida a promessa "adiciono sinal
novo com uma gravação" — sem isso ela é conversa.

Comparação de referência: o SOTA de one-shot ISLR é 50,8% MRR em 10.235 sinais
([arXiv 2502.20171](https://arxiv.org/abs/2502.20171)). Um recall@5 alto aqui é
resultado bom, não um consolo.

---

## Módulos

Os sinais ocupam o topo do pacote e o alfabeto se recolhe num subpacote — o
layout diz qual dos dois é o projeto:

```
libras/
  app.py          só o CLI: lê argumentos e escolhe o modo
  app_sinais.py   o laço do dicionário
  pose.py         subconjunto de 49 landmarks + normalização no corpo (pura)
  sequencia.py    imputação spline, reamostragem, validade (pura)
  segmenter.py    início/fim de sinal por repouso (pura)
  dtw.py          DTW em lote com banda (pura)
  dicionario.py   protótipos, busca top-5, re-ancoragem (pura + npz)
  avaliacao.py    recall@1/@5, MRR, leave-one-articulator-out (pura)
  catalogo.py     rótulo e articulador a partir do caminho (pura)
  detector.py     MediaPipe 2 mãos + pose (fino, não testado)
  ui.py           overlay do dicionário (fino)

  config.py       compartilhado
  camera.py       compartilhado (fino)
  desenho.py      cores, fonte e faixas — compartilhado (fino)
  mediapipe_io.py primitivas do MediaPipe — compartilhado

  alfabeto/       a fase 1, inteira e intocada no comportamento

training/
  prepare_sinais.py   extrai landmarks dos vídeos do V-LIBRASIL
  eval_sinais.py      leave-one-articulator-out, recall@1/@5/MRR
```

`mediapipe_io.py` existe porque três detectores (uma mão em vídeo, uma mão em
imagem, duas mãos mais pose) repetiam a mesma conferência de modelo, a mesma
montagem de array e a mesma leitura de lateralidade. Código que envolve
biblioteca externa envelhece mal em triplicata: o MediaPipe já quebrou a API uma
vez, na 1.0, levando o `mp.solutions` inteiro.

Tudo que decide alguma coisa é lógica pura e testado sem câmera, sem vídeo e sem
dataset — mesma regra da fase 1.

---

## Fora de escopo

- **Tradução contínua** (frase, gramática espacial, concordância). É o problema
  de pesquisa em aberto de verdade. Reconhecimento de sinal isolado com
  vocabulário fechado, que é o que esta fase faz, não é.
- **Expressão facial** como traço distintivo.
- **Soletração misturada com sinais** no mesmo fluxo.

---

## Riscos registrados

| risco | mitigação |
|---|---|
| 3 articuladores → modelo aprende as pessoas | leave-one-articulator-out mede; re-ancoragem corrige no uso |
| Chroma key ≠ sua sala | landmarks são invariantes a fundo — mesma defesa da fase 1 |
| V-LIBRASIL vem do ASLLVD (traduzido do ASL) | vocabulário pode ter sinais pouco usados no Brasil; o top-5 e a re-ancoragem absorvem |
| 10,5 GB e ~370 mil frames para extrair | passo offline único, ~2–5h em CPU, com retomada |
| Encoder neural pode não valer o torch | a baseline DTW decide, e a decisão fica escrita |
