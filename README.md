# libras-live

Reconhecimento do **alfabeto de Libras** em tempo real pela webcam. Você
sinaliza, ele vai formando o texto na tela — contínuo, sem apertar botão a cada
letra.

Feito como ferramenta de treino de datilologia. Roda a 30fps em CPU (33ms por
frame, medido num Apple M5 — o contador na tela mostra o seu), o classificador
tem menos de 1MB, e nenhuma imagem sai da sua máquina.

```
python -m libras.app              # soletrar: sua mão vira texto
python -m libras.app --praticar   # praticar: ele pede a letra, você faz
```

---

## Índice

- [O que ele faz e o que ele não faz](#o-que-ele-faz-e-o-que-ele-não-faz)
- [Como funciona](#como-funciona)
- [Instalação](#instalação)
- [Uso](#uso)
- [Os dados](#os-dados)
- [O treinamento](#o-treinamento)
- [Estado atual](#estado-atual)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Ajustes](#ajustes)
- [Testes](#testes)
- [Limitações conhecidas](#limitações-conhecidas)
- [Próximos passos](#próximos-passos)
- [Documentação](#documentação)
- [Créditos](#créditos)

---

## O que ele faz e o que ele não faz

Reconhece as **26 letras do alfabeto de Libras** (Língua Brasileira de Sinais —
não ASL) a partir da pose da mão, e monta palavras conforme você soletra.

Não traduz Libras como língua: frases, gramática espacial, expressão facial e
sinais compostos estão fora do escopo. Isso é problema de pesquisa em aberto, não
uma limitação a ser corrigida numa próxima versão.

Letras com movimento (H, J, K, X, Z) são reconhecidas pela pose característica,
não pela trajetória. Funciona para treinar, mas é uma simplificação — veja
[Limitações conhecidas](#limitações-conhecidas).

**Privacidade:** nenhuma imagem é gravada, transmitida ou salva em disco, em
nenhuma etapa. Nem no uso, nem na coleta de dados. Cada frame vira 21 pontos e é
descartado no mesmo instante; o que fica gravado são 63 números por amostra.

---

## Como funciona

```
frame → 21 landmarks da mão → normalizar → classificar → estabilizar → texto
```

O classificador **nunca vê a imagem**. Essa é a decisão de arquitetura que
sustenta todo o resto.

### 1. Landmarks — [`libras/landmarks.py`](libras/landmarks.py)

O [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
extrai 21 pontos 3D da mão a partir do frame. É um modelo pronto, treinado pelo
Google em milhões de mãos — não treinamos nada de visão computacional aqui.

### 2. Normalização — a função que faz o modelo funcionar longe da mesa onde nasceu

Os 21 pontos crus não servem: eles carregam onde a mão está na tela e a que
distância da câmera. Três passos, nesta ordem:

| passo | o que faz | o que elimina |
|---|---|---|
| espelhar X se for mão esquerda | usa a lateralidade que o MediaPipe reporta | um modelo por mão |
| mover o pulso para a origem | subtrai o ponto 0 de todos | a posição na tela |
| dividir pela maior distância ao pulso | escala unitária | a distância da câmera |

Saída: um vetor de **63 floats** (21 pontos × 3 coordenadas).

Como o classificador só vê esse vetor, ele é indiferente a fundo, roupa,
iluminação, cor de pele, posição na tela e distância da câmera. É também por
isso que ele é minúsculo e roda em CPU — o trabalho pesado já foi feito pelo
MediaPipe.

### 3. Classificação — [`libras/classifier.py`](libras/classifier.py)

Um classificador do scikit-learn mapeia os 63 floats numa das 26 letras, com
probabilidade. Detalhes do treino em [O treinamento](#o-treinamento).

O modelo é **fechado nas letras que viu treinando**: dada qualquer mão, ele
responde a mais parecida entre elas, nunca "não sei". Abaixo de
`CONFIANCA_REJEICAO` a predição vira um `?` amarelo e não entra no texto. A
faixa de letras no topo da tela mostra apagadas as letras sem dado — o app não
finge saber o que não sabe.

### 4. Estabilização — [`libras/stabilizer.py`](libras/stabilizer.py)

É a peça que separa um demo de algo usável. Classificar frame a frame produz
texto tremido (`AAAABAAAA`), porque uma fração dos frames sempre erra. E segurar
a mão parada numa letra emitiria a mesma letra 30 vezes por segundo.

Uma letra só é confirmada quando:

- **domina 80%** de uma janela de 15 frames (~meio segundo), **e**
- a confiança média dela nessa janela passa de **70%**

Depois de confirmada, ela fica **bloqueada** até a mão mudar de estado. É o que
impede a letra segurada de repetir.

### 5. Soletração — [`libras/speller.py`](libras/speller.py)

Acumula as letras confirmadas. Tirar a mão de cena por 1,5s insere um espaço —
é assim que você separa palavras sem tocar no teclado. Um espaço por ausência,
por mais longa que ela seja.

### Sobre o desenho do código

`landmarks.normalizar`, `stabilizer`, `speller`, `sampling` e `practice` são
**lógica pura**: numpy entra, numpy sai, nenhum relógio e nenhuma câmera lá
dentro. O tempo entra sempre por parâmetro. É por isso que 118 testes rodam em
meio segundo sem webcam e sem modelo treinado — e é por isso que a parte que
pode dar errado é a parte que está testada.

`camera.py` e `ui.py` são deliberadamente finos: um lê frames, o outro desenha o
que recebe pronto. Nenhum dos dois decide nada.

---

## Instalação

Requer **Python 3.13** e uma webcam.

```bash
git clone https://github.com/<você>/libras-live.git
cd libras-live

python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_model.sh   # baixa o hand_landmarker do MediaPipe (~7.8MB)
```

No macOS, o terminal precisa de permissão de câmera em **Ajustes do Sistema →
Privacidade e Segurança → Câmera**.

Nem o modelo do MediaPipe nem o classificador treinado são versionados no git
(são grandes e regeráveis), então um clone novo precisa passar pelos quatro
passos abaixo.

---

## Uso

### 1. Preparar a base pública (~5 min, baixa 57MB)

```bash
python training/prepare_dataset.py
```

### 2. Gravar as suas amostras

```bash
python training/collect.py            # tudo que ainda falta
python training/collect.py --revisar  # diagnóstico do que já foi gravado
python training/collect.py --refazer K Q
```

### 3. Treinar

```bash
python training/train.py
```

### 4. Rodar

```bash
python -m libras.app
```

| tecla | ação |
|---|---|
| `ESC` | sair |
| `BACKSPACE` | apagar a última letra |
| `C` | limpar o texto |
| — | tirar a mão de cena por 1,5s insere um espaço |

Ao sair, o texto vai para a área de transferência.

**Na tela:** a letra atual e sua confiança no canto superior esquerdo, com a
barra de progresso do buffer de estabilização; a faixa das 26 letras logo
abaixo, com as sem dado apagadas; o texto no rodapé, com a palavra em andamento
destacada em verde.

### 5. Praticar

```bash
python -m libras.app --praticar
python -m libras.app --praticar --rodadas 20
```

Inverte o exercício: em vez de você soletrar o que quiser, o app pede uma letra
e confere se você acertou, cronometrando. É a diferença entre ler e escrever, e
é o que realmente treina datilologia.

Errar não passa a rodada — de propósito. Pular no erro deixaria você treinar só
o que já sabe, e o placar mediria sorte. `P` pula a letra atual, `R` reinicia a
sessão.

---

## Os dados

O modelo é treinado com **duas fontes**, e entender por que ambas são
necessárias é entender o projeto.

### Fonte 1: a base pública

[Brazilian Sign Language Alphabet
Dataset](https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset)
— 4.411 imagens de Libras, cobrindo 15 letras: **A B C D E I L M N O R S U V W**.
É Libras mesmo, não ASL.

`prepare_dataset.py` baixa, extrai e roda o MediaPipe em cada imagem,
guardando os vetores de 63 floats. O formato é idêntico ao da coleta pela
webcam, o que permite misturar as duas fontes sem reconciliar nada.

#### A cascata de recuperação — [`libras/recovery.py`](libras/recovery.py)

O MediaPipe falha em imagens de mão fechada, onde os dedos se escondem atrás da
palma. Na primeira passagem ele achava mão em só **24% das imagens de N** e 53%
das de M — N terminava com 37 amostras contra 568 de B.

A imagem não é ruim; o detector é que é sensível a enquadramento, contraste e
inclinação. Então em vez de desistir na primeira falha, o script reapresenta a
mesma foto de outros jeitos, em ordem de custo e de fidelidade:

```
original → espelhada → ampliada 2x → contraste (CLAHE) →
ampliada+contraste → girada ±20° → girada ±40°
```

Quando uma variante encontra a mão, a transformação é **desfeita nos pontos**
(inversa da matriz afim para as rotações, `x → 1-x` para o espelho, e a
lateralidade reportada é invertida junto), de modo que os landmarks voltem à
geometria da imagem original. Isso é recuperação, não aumento de dados: o
resultado é a mão que estava lá, medida por outro caminho.

**377 imagens recuperadas.** N sobe de 37 para 138 amostras, M de 105 para 159,
O de 92 para 150, S de 104 para 151. A variante que mais resgata é `girada+20°`
(230 imagens) — a base tem uma inclinação sistemática.

### Fonte 2: as suas gravações

```bash
python training/collect.py
```

Onze letras (**F G H J K P Q T X Y Z**) não existem na base pública e só podem
vir daqui. Mas as outras quinze também precisam da sua mão, e essa foi uma
lição aprendida na marra:

> Com L existindo só como foto de estúdio (mãos de outras pessoas) e G existindo
> só como webcam (a sua mão), o modelo passou a ler L como G. Não por confusão
> de forma — as duas nuvens estão bem separadas no espaço dos landmarks — mas
> porque a região do G era construída a partir daquela mão exata, apertada e
> confiante, enquanto a do L era difusa e estrangeira. O empate não era justo.

Grave as 26. A base pública continua ajudando com variedade de mãos; suas
gravações dão a âncora.

#### O filtro de diversidade — [`libras/sampling.py`](libras/sampling.py)

**Amostra repetida não conta.** Uma amostra só entra se estiver a pelo menos
`DISTANCIA_MINIMA_AMOSTRA` de *todas* as que já entraram — amostragem por disco
de Poisson no espaço dos landmarks, onde cada amostra aceita queima uma
vizinhança ao redor de si.

Isso existe porque a primeira coleta não tinha o filtro. 200 amostras a 30fps
saem em **oito segundos**, antes de a mão ter tempo de mudar de ângulo. O
resultado, medido depois:

| letra | raio da nuvem (antes) | (depois do filtro) |
|---|---|---|
| K | 0,118 | 0,365 |
| Q | 0,123 | 0,472 |
| T | 0,178 | 0,439 |
| F | 0,258 | 0,397 |
| G | 0,211 | 0,396 |

Contra ~0,64 de média nas letras da base pública. O modelo não tinha aprendido a
letra: tinha decorado uma pose exata, e qualquer desvio dela na hora do uso
virava outra letra.

Com o filtro, mexer a mão não é conselho — é o que faz a barra andar. A tela
mostra um medidor de variedade ao vivo (amarelo abaixo da meta, verde acima) e
avisa quando você trava. O limiar afrouxa sozinho depois de 1,5s sem aceitar
nada, com piso, para que a coleta sempre termine.

`--revisar` mostra o raio de cada gravação e aponta quais vale regravar.

---

## O treinamento

```bash
python training/train.py
```

Compara três candidatos — Random Forest, MLP e SVM RBF — e salva o melhor, junto
com um relatório completo em `models/relatorio_treino.txt`.

Duas decisões mudam o que o número final significa.

### Macro-F1, não acurácia

As classes são desiguais (B tem 779 amostras, F tem 200). Acurácia global é
dominada pelas classes fartas: um modelo que erre uma letra rara inteira perde
menos de 1% de acurácia. **Macro-F1 dá o mesmo peso a cada letra**, então errar
N custa o mesmo que errar B. É a métrica que responde "ele reconhece o
alfabeto?" em vez de "ele acerta muito?".

Os candidatos que aceitam também recebem `class_weight="balanced"`.

### Aumento dentro de cada fold, nunca antes de separar

[`libras/augment.py`](libras/augment.py) fabrica variações a partir das amostras
reais, no espaço dos landmarks — três transformações, todas seguidas de
renormalização para que a amostra sintética obedeça às mesmas invariantes que
uma real:

| transformação | simula | limite padrão |
|---|---|---|
| rotação 3D | pulso torto, mão fora do eixo da câmera | ±12° por eixo |
| ruído gaussiano | o jitter que o MediaPipe tem frame a frame | σ = 0,02 |
| escala em z | o erro da estimativa de profundidade | ±30% |

Não há aumento de escala nem de translação: a normalização já as elimina, seriam
operações nulas. E os limites são modestos de propósito — rotação demais
transforma um M num W, e aí o aumento passa a ensinar a letra errada.

Cada classe do treino é completada até **600 amostras**. Classes acima disso
ficam intactas: o objetivo é levantar o piso, não rebaixar o teto.

**O aumento acontece dentro de cada fold da validação cruzada, e no ajuste final
só sobre a partição de treino.** As amostras sintéticas são derivadas das reais;
se o aumento viesse antes da separação, uma variação da mesma mão cairia no
treino e outra no teste, e a métrica subiria sem que o modelo tivesse melhorado.
**O conjunto de teste é sempre 100% amostras reais.**

Os folds rodam em paralelo — com o aumento, cada um treina em ~15 mil amostras,
e o SVM sozinho levaria meia hora em série.

---

## Estado atual

Modelo cobrindo **as 26 letras**, treinado com 9.479 amostras reais das duas
fontes.

```
Vencedor: mlp
macro-F1 (teste real): 96,9%
acurácia (teste real): 97,2%
```

Na prática, o alfabeto inteiro sai em ordem, sem erro.

### O que esse número é e o que ele não é

Ele mede o quão bem o modelo aprendeu **a distribuição em que foi treinado**.
Não é uma medida de uso real: treino e teste saem das mesmas sessões de
gravação, no mesmo dia, com a mesma luz e a mesma roupa. Falta um conjunto de
avaliação gravado separadamente — veja [Próximos passos](#próximos-passos).

Vale registrar que **este número caiu de 99,6% para 96,9% quando o projeto
melhorou**. Antes, as 15 letras públicas só existiam como foto de estúdio e as
11 restantes só como webcam, então o modelo podia acertar pela origem da amostra
em vez da forma da mão. Com todas as letras tendo as duas fontes, o atalho
sumiu. 96,9% num problema honesto vale mais que 99,6% num problema com gabarito.

### Calibração

O MLP vencedor está bem calibrado — erro de calibração esperado (ECE) de
**0,013** no conjunto de teste. Isso importa porque tanto o estabilizador quanto
a rejeição filtram por confiança, e um modelo que mente sobre a própria certeza
quebraria os dois.

Efeito colateral: 95% dos frames de teste saem na faixa de confiança 0,9–1,0,
com média 0,997. Os limiares atuais quase não rejeitam nada em dados dessa
distribuição — `CONFIANCA_REJEICAO=0,55` deixa passar 99,6% dos frames. O `?`
amarelo é raro na prática.

### Letras mais difíceis

| letra | F1 | confunde com |
|---|---|---|
| T | 88,0% | W, B |
| W | 92,3% | S |
| F | 92,7% | — |
| U | 93,5% | V |
| V | 94,8% | U |

Duas famílias distintas:

- **U↔V, B↔W** — mesma mão, diferença de um dedo ou do afastamento entre eles.
  Nas 63 coordenadas cruas isso é sutil; para um humano é óbvio.
- **T** — mão fechada com o polegar entre os dedos. Os dedos ficam escondidos e
  o MediaPipe estima mal o que não vê. Não se resolve com mais amostras: os
  landmarks é que são ruins. É o mesmo problema que M e N tinham na base
  pública.

---

## Estrutura do projeto

```
libras/
  app.py          loop principal: liga tudo e desenha
  camera.py       captura da webcam (fino)
  landmarks.py    MediaPipe + a normalização dos 21 pontos
  classifier.py   carrega o modelo treinado, prediz e rejeita
  stabilizer.py   decide quando uma predição vira letra
  speller.py      acumula letras em texto, espaço por ausência
  practice.py     sessão do modo prática (lógica pura)
  sampling.py     filtro de diversidade da coleta (lógica pura)
  augment.py      aumento de dados no espaço dos landmarks
  recovery.py     cascata de variantes para imagens difíceis
  ui.py           desenho do overlay (fino)
  config.py       todos os parâmetros ajustáveis

training/
  prepare_dataset.py   baixa a base pública e extrai landmarks
  collect.py           grava suas amostras pela webcam
  train.py             compara modelos, treina e salva o relatório

tests/            118 testes, sem câmera e sem modelo
scripts/          download do modelo do MediaPipe
models/           hand_landmarker.task, classifier.joblib, relatorio_treino.txt
data/             dataset_publico.npz, coletados/*.npy, raw/
```

`data/` e os modelos não são versionados — são regeráveis e pesados.

---

## Ajustes

Tudo em [`libras/config.py`](libras/config.py).

| parâmetro | padrão | efeito |
|---|---|---|
| `TAMANHO_BUFFER` | 15 | maior = mais estável e mais lento |
| `DOMINANCIA_MINIMA` | 0,80 | fração do buffer que a letra precisa ocupar |
| `CONFIANCA_MINIMA` | 0,70 | maior = erra menos e engole mais letras |
| `CONFIANCA_REJEICAO` | 0,55 | abaixo disso a predição vira `?` |
| `SEGUNDOS_PARA_ESPACO` | 1,5 | ausência de mão que insere espaço |
| `DISTANCIA_MINIMA_AMOSTRA` | 0,12 | quão distintas as amostras da coleta precisam ser |
| `DISPERSAO_ALVO` | 0,35 | raio de nuvem considerado saudável |
| `AUG_ROTACAO_GRAUS` | 12,0 | rotação máxima do aumento, por eixo |
| `AUG_ALVO_POR_CLASSE` | 600 | amostras por letra depois de equilibrar |
| `ESPELHAR_VIDEO` | `True` | efeito espelho na imagem |

---

## Testes

```bash
python -m pytest tests/ -q
```

118 testes, ~0,5s, sem câmera e sem modelo treinado. Cobrem normalização,
estabilização, soletração, aumento de dados, recuperação de imagens, rejeição,
diversidade da coleta e modo prática — toda a lógica que pode dar errado.

O que não está testado é o que não dá para testar sem hardware: a captura da
webcam e o desenho na tela. Ambos foram mantidos finos justamente por isso.

---

## Limitações conhecidas

**Letras com movimento são tratadas como pose.** H, J, K, X e Z têm trajetória
em Libras, mas aqui são classificadas por um frame só. A medida que expõe isso:
a nuvem do J tem raio 1,43 contra ~0,5 típico, e invade M, C, A e L — as 200
amostras não são uma pose, são uma trajetória inteira achatada num rótulo só.

**Mãos fechadas são mal detectadas.** Quando os dedos se escondem atrás da
palma, o MediaPipe estima posições ruins. Afeta T principalmente, e afetava M e
N antes da cascata de recuperação.

**A métrica não descreve o uso real.** Treino e teste saem das mesmas sessões.

**Uma mão só.** `num_hands=1` — sinais que usam as duas mãos estão fora.

---

## Próximos passos

Em ordem de valor:

1. **Conjunto de avaliação separado.** Uma sessão de gravação em outro dia,
   outra luz, outra roupa, usada só como teste e nunca no treino. É o que
   transforma "achei que melhorou" em evidência, e é pré-requisito para medir
   qualquer coisa abaixo.
2. **Features geométricas** — ângulos entre dedos e distâncias ponta-a-ponta
   somados às coordenadas cruas, mirando U/V e B/W especificamente.
3. **Modelo temporal** para as letras de movimento: classificar uma janela de
   frames em vez de um frame. É a mudança de arquitetura que o raio do J pede.
4. **Autocorreção com dicionário PT-BR** — um erro em dez letras arruína a
   palavra inteira; a palavra em andamento já aparece destacada e poderia ser
   corrigida.

---

## Documentação

[Design da fase 1](docs/superpowers/specs/2026-08-11-libras-live-design.md) — as
decisões de arquitetura, as alternativas rejeitadas e o porquê de cada uma.

---

## Créditos

- [Brazilian Sign Language Alphabet Dataset](https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset)
  — base pública de imagens de Libras.
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  — detecção dos 21 pontos da mão.

Dependências: `mediapipe`, `opencv-python`, `scikit-learn`, `numpy`, `joblib`,
`pytest`. Versões exatas em [`requirements.txt`](requirements.txt).
