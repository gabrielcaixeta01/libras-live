# libras-live — Design

Data: 2026-08-11
Status: aprovado

## Problema

Estudante de Libras quer praticar datilologia (soletração manual) com feedback
imediato. Abrir a câmera do notebook, sinalizar à vontade e ver o texto sendo
formado na tela em tempo real, de forma contínua — sem tirar foto, sem apertar
botão a cada letra.

Tradução contínua de Libras completa (frases, gramática espacial, expressão
facial) é problema de pesquisa em aberto. Este projeto não tenta isso. O escopo
é reconhecimento de um vocabulário definido.

## Escopo

**Fase 1 (este documento):** alfabeto de Libras, reconhecimento por pose de mão,
modo único de tradução livre.

**Fase 2 (planejada, fora deste documento):** sinais com movimento, trocando a
cabeça do classificador de "um frame" para "janela de frames" e reaproveitando
todo o resto do pipeline.

**Fora de escopo:** prática guiada com pontuação, narração por voz (TTS),
gramática de Libras, expressão facial, duas mãos simultâneas.

## Decisões

### D1 — Landmarks, não pixels

MediaPipe HandLandmarker extrai 21 pontos 3D da mão por frame. O classificador
recebe esses pontos, nunca a imagem.

Motivo: o modelo fica imune a fundo, roupa, iluminação e cor de pele — o
MediaPipe já absorveu essa variação. Treina em segundos numa CPU, o modelo tem
~1MB e roda a 30fps folgado. E o mais importante: é a mesma representação que a
fase 2 vai usar, então nada aqui é jogado fora.

Alternativa rejeitada: CNN sobre imagem crua (transfer learning). Precisaria de
GPU, generalizaria mal para a webcam do usuário, e nada dela se reaproveitaria
para sinais dinâmicos.

### D2 — MediaPipe Tasks API, não `mp.solutions`

MediaPipe 1.0.0 **removeu** `mp.solutions.hands`. Usamos
`mediapipe.tasks.python.vision.HandLandmarker` com o modelo
`hand_landmarker.task` versionado em `models/`.

Consequência prática: tutoriais de Libras disponíveis online usam a API legada e
não servem como referência de código.

### D3 — Normalização invariante a posição e escala

Vetor de features por frame: 21 pontos × 3 coordenadas = **63 floats**.

Transformações, nesta ordem:
1. **Espelhamento** — se a mão detectada for esquerda, inverte o eixo X. O
   modelo passa a ser agnóstico a qual mão você usa, e cada amostra de treino
   serve para as duas.
2. **Translação** — pulso (landmark 0) vira a origem.
3. **Escala** — divide pela maior distância do pulso a qualquer ponto.

Depois disso, distância da câmera e posição na tela deixam de importar; só a
forma da mão sobra. Sem esse passo o modelo decora a sua mesa.

### D4 — Estabilização temporal

Classificar frame a frame produz texto tremido (`AAAABAAAA`), porque uma fração
dos frames sempre erra. É o erro que torna a maioria das implementações
inutilizável na prática.

Solução — buffer circular dos últimos N frames (padrão 15, ~0,5s a 30fps). Uma
letra é confirmada quando:
- domina ≥ 80% do buffer, **e**
- a confiança média dessa classe ≥ limiar (padrão 0,70).

Após confirmar, o estabilizador entra em **período refratário**: a mesma letra
só pode ser emitida de novo depois que a mão sair desse estado (outra letra
dominar, ou a mão sumir). Sem isso, segurar a letra emite `AAAAAAA`.

Ausência de mão por ≥ 1,5s emite um espaço; um único espaço, não uma sequência.

Buffer, limiar e tempo de espaço ficam em `config.py` — são os parâmetros que
mais serão ajustados no uso real.

### D5 — Dados: base pública de Libras + coleta das faltantes

Base pública: [Brazilian Sign Language Alphabet
Dataset](https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset)
— 4.411 imagens do alfabeto **de Libras** (não ASL), cobrindo 15 letras:
A, B, C, D, E, I, L, M, N, O, R, S, U, V, W.

Faltam 11 letras: **F, G, H, J, K, P, Q, T, X, Y, Z**. O usuário grava essas
via `training/collect.py`.

Como ambos os caminhos terminam em landmarks normalizados (D3), imagem baixada
e frame de webcam viram o mesmo vetor de 63 floats. Misturar as duas fontes é
concatenar arrays, sem reconciliação de formato.

### D6 — Letras com movimento são classificadas pela pose final

H, J, K, X e Z envolvem movimento em Libras. Na fase 1 são tratadas pela pose
característica. É uma simplificação assumida: funciona para treino de
datilologia, e a fase 2 as trata corretamente.

## Arquitetura

```
frame → landmarks (21×3) → normalizar → classificar → estabilizar → soletrar → tela
```

| Módulo | Responsabilidade | Depende de |
|---|---|---|
| `libras/config.py` | Parâmetros ajustáveis | — |
| `libras/landmarks.py` | MediaPipe → 21 pontos; normalização | mediapipe, numpy |
| `libras/stabilizer.py` | Quando uma letra conta | numpy |
| `libras/speller.py` | Letras → texto | — |
| `libras/classifier.py` | Carrega modelo, prediz letra + confiança | sklearn |
| `libras/camera.py` | Captura da webcam | opencv |
| `libras/ui.py` | Janela e overlay | opencv |
| `libras/app.py` | Loop principal | todos |

`normalize_landmarks`, `Stabilizer` e `Speller` são lógica pura — entram
números, saem números. Toda a lógica que pode dar errado está neles, e os três
são testáveis sem abrir câmera. `camera.py` e `ui.py` são finos de propósito:
são a parte que não dá para testar automaticamente, então contêm o mínimo.

## Tratamento de erro

| Situação | Comportamento |
|---|---|
| Webcam indisponível | Erro claro nomeando o índice tentado, sem stacktrace |
| `models/classifier.joblib` ausente | Instrui a rodar `training/train.py` |
| `models/hand_landmarker.task` ausente | Instrui a rodar o download |
| Nenhuma mão no frame | Estado normal — UI mostra "sem mão", conta para o espaço |
| Confiança abaixo do limiar | Nada é emitido; UI mostra a predição em cinza |
| Imagem do dataset sem mão detectável | Pulada, contabilizada no relatório de preparação |

## Testes

Testes unitários, sem câmera nem modelo treinado:

- **normalização** — invariância a translação e escala (mesma mão em posições e
  distâncias diferentes gera o mesmo vetor); espelhamento de mão esquerda.
- **estabilizador** — confirma na maioria; não confirma sem confiança; não
  repete letra segurada; volta a permitir após troca de estado.
- **speller** — acumula, espaço único após ausência, backspace, clear.

`camera`/`ui` não têm teste automatizado; são validados rodando o app.

## Estrutura

```
libras-live/
├── libras/{config,landmarks,stabilizer,speller,classifier,camera,ui,app}.py
├── training/{prepare_dataset,collect,train}.py
├── models/          # hand_landmarker.task + classifier.joblib
├── data/            # landmarks extraídos (gitignored)
├── tests/
└── docs/superpowers/specs/
```

## Runtime

Python 3.13 em `.venv`. mediapipe 1.0.0, opencv 5.0.0, scikit-learn 1.9.0,
numpy 2.5.2 — combinação validada nesta máquina (macOS arm64) antes de escrever
código.
