# libras-live

Reconhecimento de **Libras** em tempo real pela webcam, em duas frentes:

- **Alfabeto** — você soletra, ele vai formando o texto na tela, contínuo, sem
  apertar botão a cada letra.
- **Dicionário reverso de sinais** — você faz um sinal, ele mostra as cinco
  palavras mais prováveis, dentro de um vocabulário núcleo de 163 sinais do dia
  a dia. É a busca que um dicionário de Libras não oferece: procurar pela
  *forma* quando você não sabe o nome.

Roda a 30fps em CPU (33ms por frame, medido num Apple M5 — o contador na tela
mostra o seu), o classificador do alfabeto tem menos de 1MB, e nenhuma imagem
sai da sua máquina.

```
python -m libras.app              # soletrar: sua mão vira texto
python -m libras.app --praticar   # praticar: ele pede a letra, você faz
python -m libras.app --sinais     # dicionário: você sinaliza, ele traduz
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
- [Sinais: o dicionário reverso](#sinais-o-dicionário-reverso)
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

Reconhece **sinais isolados** (palavras) por busca num dicionário de protótipos:
você sinaliza, ele devolve cinco candidatos ordenados. Vocabulário fechado, uma
palavra por consulta.

O vocabulário é **curado e pequeno de propósito**: 163 sinais do cotidiano
(saudações, cortesia, família, necessidades, tempo, perguntas), não as 1.364
palavras do V-LIBRASIL. A razão está medida em
[Sinais: o dicionário reverso](#sinais-o-dicionário-reverso) — com 1.364
candidatos a resposta certa aparece em 16,4% das consultas, e com 163 em 45,5%.
`--tudo` devolve o vocabulário inteiro para quem preferir cobertura a acerto.

Não traduz Libras como língua: **frases**, gramática espacial, concordância
verbal e expressão facial como traço distintivo estão fora do escopo. É aí que
mora o problema de pesquisa em aberto — reconhecimento de sinal isolado com
vocabulário fechado, que é o que a segunda frente faz, não é.

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

### 1. Landmarks — [`libras/alfabeto/landmarks.py`](libras/alfabeto/landmarks.py)

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

### 3. Classificação — [`libras/alfabeto/classifier.py`](libras/alfabeto/classifier.py)

Um classificador do scikit-learn mapeia os 63 floats numa das 26 letras, com
probabilidade. Detalhes do treino em [O treinamento](#o-treinamento).

O modelo é **fechado nas letras que viu treinando**: dada qualquer mão, ele
responde a mais parecida entre elas, nunca "não sei". Abaixo de
`CONFIANCA_REJEICAO` a predição vira um `?` amarelo e não entra no texto. A
faixa de letras no topo da tela mostra apagadas as letras sem dado — o app não
finge saber o que não sabe.

### 4. Estabilização — [`libras/alfabeto/stabilizer.py`](libras/alfabeto/stabilizer.py)

É a peça que separa um demo de algo usável. Classificar frame a frame produz
texto tremido (`AAAABAAAA`), porque uma fração dos frames sempre erra. E segurar
a mão parada numa letra emitiria a mesma letra 30 vezes por segundo.

Uma letra só é confirmada quando:

- **domina 80%** de uma janela de 15 frames (~meio segundo), **e**
- a confiança média dela nessa janela passa de **70%**

Depois de confirmada, ela fica **bloqueada** até a mão mudar de estado. É o que
impede a letra segurada de repetir.

### 5. Soletração — [`libras/alfabeto/speller.py`](libras/alfabeto/speller.py)

Acumula as letras confirmadas. Tirar a mão de cena por 1,5s insere um espaço —
é assim que você separa palavras sem tocar no teclado. Um espaço por ausência,
por mais longa que ela seja.

### Sobre o desenho do código

`landmarks.normalizar`, `stabilizer`, `speller`, `sampling` e `practice` são
**lógica pura**: numpy entra, numpy sai, nenhum relógio e nenhuma câmera lá
dentro. O tempo entra sempre por parâmetro. É por isso que 380 testes rodam em
segundos sem webcam e sem modelo treinado — e é por isso que a parte que pode dar
errado é a parte que está testada.

`camera.py`, `mediapipe_io.py`, `desenho.py` e os dois `ui.py` são
deliberadamente finos: uns leem frames e traduzem o que a biblioteca devolve, os
outros desenham o que recebem pronto. Nenhum deles decide nada — é por isso que
não estarem testados não é um buraco.

---

## Instalação

Requer **Python 3.13** e uma webcam.

```bash
git clone https://github.com/<você>/libras-live.git
cd libras-live

python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_model.sh   # baixa os modelos do MediaPipe (~17MB)
```

No macOS, o terminal precisa de permissão de câmera em **Ajustes do Sistema →
Privacidade e Segurança → Câmera**.

Nem os modelos do MediaPipe nem o classificador treinado são versionados no git
(são grandes e regeráveis), então um clone novo precisa passar pelos passos
abaixo. O alfabeto usa só o `hand_landmarker`; os sinais usam os dois.

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

### 6. Sinais (opcional, e independente dos passos acima)

O dicionário de sinais tem a sua própria base e o seu próprio pipeline — não
precisa do classificador do alfabeto, nem ele do dicionário.

```bash
bash scripts/download_vlibrasil.sh                # ~10,8 GB, baixa e extrai
python training/prepare_sinais.py --videos data/raw/v-librasil
python training/eval_sinais.py --nucleo           # a baseline DTW do núcleo
python training/train_sinais.py --treino-nucleo   # o encoder (~1,1 min)
python -m libras.app --sinais                     # usar
```

O script baixa do **espelho no Kaggle**, que serve o bundle sem exigir login. O
site oficial (`libras.cin.ufpe.br`) responde 502 desde antes desta escrita — o
proxy da UFPE está de pé, o backend não. O download é retomável.

Depois que a extração terminar, `data/raw/v-librasil.zip` pode ser apagado: ele é
o mesmo conteúdo de `data/raw/v-librasil/`, e são **10,8 GB** parados. O script
rebaixa se você precisar dele de novo. E depois do `prepare_sinais.py` nem os
vídeos são mais necessários para treinar — o que o encoder consome é o
`data/sinais/dicionario.npz`, de 70 MB.

A extração roda em 8 processos por padrão (`--jobs`), o que a traz de 2-5 horas
para a casa da meia hora num laptop de 10 núcleos. Cada worker carrega os dois
modelos do MediaPipe, então o teto é a RAM e não os núcleos.

`--limite 50` processa só os primeiros vídeos, para testar o caminho antes de
gastar o tempo todo. A extração também é retomável: interromper e rodar de novo
continua de onde parou.

O `train_sinais.py` **não reprocessa vídeo nenhum** — ele trabalha sobre os
protótipos já extraídos. Quando ele termina, o app passa a usar o encoder sozinho;
sem ele, continua na baseline DTW. `--limite-sinais 30 --epocas 3` ensaia o
caminho inteiro em segundos.

O `--treino-nucleo` treina e mede no [vocabulário
núcleo](#o-vocabulário-núcleo-a-promessa-que-cabe-no-dado), que é o que o app
indexa. Sem ele o treino usa as 1.363 classes, custa 9,1 min em vez de 1,1, e
entrega um recall@5 mais baixo **no vocabulário que o app responde** — ver a
tabela do núcleo. Ele também desliga sozinho a medição de vocabulário aberto,
que não cabe em 163 classes; era um `--aberto 0` que o script pedia e agora
resolve por conta própria.

Detalhes em [Sinais: o dicionário reverso](#sinais-o-dicionário-reverso).

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

#### A cascata de recuperação — [`libras/alfabeto/recovery.py`](libras/alfabeto/recovery.py)

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

#### O filtro de diversidade — [`libras/alfabeto/sampling.py`](libras/alfabeto/sampling.py)

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

[`libras/alfabeto/augment.py`](libras/alfabeto/augment.py) fabrica variações a partir das amostras
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

## Sinais: o dicionário reverso

```bash
python -m libras.app --sinais          # 163 sinais do dia a dia
python -m libras.app --sinais --tudo   # as 1.364 do V-LIBRASIL
```

Um dicionário de Libras é indexado por português. Se você vê um sinal que não
conhece, não tem como procurar — você não sabe o nome dele, que é exatamente o
que está tentando descobrir. Este modo inverte o índice: **a chave passa a ser a
forma**.

É também o que torna o top-5 honesto em vez de uma desculpa. Num tradutor, cinco
respostas é uma resposta errada. Num dicionário, cinco candidatos é uma resposta
boa — você reconhece a certa quando a vê.

### A inversão que faz isso funcionar

O alfabeto normaliza pondo **o pulso na origem**, o que apaga onde a mão está.
Para uma letra isso é o certo: a letra é só a forma da mão.

Para um sinal é errado. Em Libras a **localização é fonema** — PAI e MÃE têm a
mesma configuração de mão em lugares diferentes do rosto. Uma normalização que
apaga a localização apaga a diferença entre as duas palavras.

| | alfabeto | sinal |
|---|---|---|
| origem | pulso | ponto médio dos ombros |
| escala | maior distância ao pulso | distância entre os ombros |
| preserva | forma da mão | forma **+ localização + trajetória** |
| unidade | 1 frame | janela de 32 frames |
| mãos | 1 | 2 |
| pontos | 21 | 49 (2 mãos + 7 do corpo) |

`landmarks.normalizar` não foi tocado. O app do alfabeto continua idêntico; o
novo mora no topo de [`libras/`](libras/), com o alfabeto recolhido em [`libras/alfabeto/`](libras/alfabeto/).

### O pipeline

```
webcam → 49 landmarks → segmentar por repouso → imputar buracos
       → normalizar no corpo → reamostrar p/ 32 frames → buscar → top-5
```

**Segmentação automática** — [`sinais/segmenter.py`](libras/segmenter.py).
A mão sai do descanso, o sinal começa; a mão para, ele acaba e vai para a busca.
Mesmo espírito do "espaço por ausência" do alfabeto: sem tocar no teclado. A
velocidade é medida em **larguras de ombro por segundo**, não em pixels — senão
chegar perto da câmera dispararia sinais sozinho.

**Imputação PCHIP** — [`sinais/sequencia.py`](libras/sequencia.py).
O MediaPipe perde a mão quando ela cruza o corpo ou fecha. Num frame isolado
isso é fatal; numa sequência não, porque os vizinhos no tempo sabem onde ela
estava. Nunca por extrapolação: nas bordas o valor é travado, porque uma mão
inventada longe do corpo estraga toda distância que ela tocar. E **nunca por
spline cúbica**: ela também dispara *dentro* do intervalo, entre dois pontos
válidos distantes — foi o que produziu as coordenadas de ±400 larguras de ombro
que o projeto passou meses tratando como artefato de medida. PCHIP preserva a
forma e não pode ultrapassar os dados que interpola. Uma **máscara de validade**
acompanha a sequência dizendo o que foi medido e o que foi inventado.

**49 pontos, não 543.** O MediaPipe entrega 468 de rosto + 33 de pose + 21 por
mão. Usar tudo piora: sobre três amostras por sinal, cada coordenada a mais é
uma chance a mais de decorar ruído. O subconjunto segue
[arXiv:2510.24887](https://arxiv.org/abs/2510.24887) (IEEE SAS 2026), que bate o
SOTA em Libras isolada com 5x menos tempo justamente por selecionar landmarks.

### Os dados, e a restrição que eles impõem

[V-LIBRASIL](https://libras.cin.ufpe.br) (UFPE/CIn): **1.364 sinais, 4.086
vídeos aproveitáveis, ~10,8 GB**. O número que decide o desenho inteiro:

```
4.086 vídeos ÷ 1.364 sinais = 3 gravações por sinal — uma por articulador
```

São 4.089 gravações no catálogo original, mas três (`Congelar_Articulador3`,
`Criança_Articulador3`, `Poesia_Articulador2`) estão corrompidas na origem —
vêm com largura, altura e fps zerados, e o próprio dataset as lista em
`error.csv`. O espelho já as exclui.

Três exemplos por classe não é "dataset pequeno", é **few-shot por definição**.
Um classificador de 1.364 saídas decoraria os articuladores. Daí a arquitetura
ser **representação + busca vetorial**, não classificação.

```bash
bash scripts/download_vlibrasil.sh                # espelho do Kaggle
python training/prepare_sinais.py --videos data/raw/v-librasil
python training/eval_sinais.py                    # o placar honesto
```

A extração é retomável e nenhum frame é gravado — cada vídeo vira 32×147 números
e a imagem é descartada, como na fase 1.

### A baseline vem antes do encoder

A métrica de busca é **DTW** — alinhamento temporal, numpy puro, zero dependência
nova. O mesmo sinal feito devagar e feito rápido é a mesma palavra; uma distância
frame a frame diria que não.

O passo seguinte natural seria um encoder neural com metric learning. Mas ele
traz **torch** para um projeto que era scikit-learn puro com modelo de menos de
1MB — a maior mudança de dependência da história do repo. Antes de pagar essa
conta é preciso saber quanto ela compra, e a baseline é quem produz esse número.
**Se o encoder não bater o DTW por margem clara em leave-one-articulator-out, o
torch não entra.** A decisão fica escrita em `models/relatorio_sinais.txt`.

O encoder foi construído e medido depois disso — ele bate a baseline, e não por
margem clara. O número e a leitura estão em
[O encoder neural, e o que ele comprou](#o-encoder-neural-e-o-que-ele-comprou).

Para que a troca seja barata, a métrica entra por parâmetro no dicionário:
sequência (32, 147) com DTW hoje, embedding (256,) com cosseno depois. O app não
sabe a diferença.

Custo medido da busca completa: **87 ms** para 4.092 protótipos — e é uma
consulta por sinal, não por frame, então ela não entra no orçamento dos 33 ms do
loop de vídeo. Sem FAISS, sem índice: é um produto de matrizes.

### O protocolo de avaliação

**leave-one-articulator-out**: indexa com dois articuladores, consulta com o
terceiro, três rodízios. Com três pessoas gravando tudo, qualquer divisão
aleatória põe o mesmo articulador dos dois lados e o placar mede memorização de
pessoa — o mesmo erro que custou caro no L↔G.

| métrica | o que responde |
|---|---|
| **recall@5** | **a resposta está na tela?** ← a métrica do produto |
| recall@1 | acertou de primeira? |
| MRR | quão alto na lista? |

Referência da área para busca em vocabulário grande: 50,8% de MRR em 10.235
sinais ([arXiv:2502.20171](https://arxiv.org/abs/2502.20171)).

### Re-ancoragem: quando errar ensina

Três articuladores significam que o modelo tem todo incentivo para aprender *as
pessoas* em vez dos sinais. É o mecanismo do L↔G de novo, agora com número.

A mitigação é estrutural. Quando o resultado sai errado, você aperta o **número
do candidato certo** — e aquele sinal, feito pela sua mão, vira protótipo no
dicionário. Da próxima vez ele acerta. Sem retreinar nada.

| tecla | ação |
|---|---|
| `1`–`5` | o candidato certo era este — ensina com a sua mão |
| `N` | nomear um sinal que não está no dicionário |
| `C` | limpar o resultado |
| `ESC` | sair |

Seus protótipos ficam em `data/sinais/meus_prototipos.npz`, separados do
dicionário base, que nunca é reescrito. É a mesma lição da fase 1 — a sua mão é
a âncora — resolvida por arquitetura em vez de por mais uma sessão de coleta.

**Estado: o dicionário está construído, e o número da baseline é ruim.** 4.086
vídeos extraídos, 1.363 sinais, e o placar honesto do DTW:

```
leave-one-articulator-out, 4.086 consultas
recall@5  7,2%      recall@1  3,1%      MRR  4,5%
```

Acaso é 0,4%, então a busca aprendeu alguma coisa — mas 7,2% não é um
dicionário que serve. **A baseline DTW não sustenta o produto**, que é
exatamente a pergunta que esta fase existia para responder.

O número não é bug. O que foi verificado antes de aceitá-lo:

| verificação | resultado |
|---|---|
| auto-recuperação (consulta = protótipo do índice) | 6/6, similaridade ~1,000 |
| vetores degenerados (NaN, Inf, constantes) | nenhum |
| distância intra-sinal ÷ distância aleatória | 0,80 |
| razão por grupo (mão esq. / mão dir. / corpo) | 0,79 / 0,83 / 0,85 |
| remover as amostras com outlier (\|v\|>5) | não muda nada |

A busca está correta; a **representação** é que separa mal entre pessoas. O
mesmo sinal feito por dois articuladores fica só 20% mais perto que dois sinais
quaisquer, e com 1.363 classes isso produz os 7,2% medidos.

(A última linha da tabela envelheceu bem e mal ao mesmo tempo. Os outliers de
fato não mudavam o DTW — mas eles não eram artefato de medida, eram a spline
cúbica disparando entre pontos válidos distantes, e trocá-la por PCHIP levou a
amplitude máxima de **417,7 para 5,8** larguras de ombro. O DTW mal sentiu, o
encoder sentiu. Ver
[`docs/notas-de-construcao-do-dicionario.md`](docs/notas-de-construcao-do-dicionario.md).)

A fraqueza é uniforme em toda escala — o que muda com o tamanho do vocabulário
não é a qualidade da representação, é quantos candidatos existem para confundir:

| sinais no dicionário | 25 | 100 | 500 | 1.363 |
|---|---|---|---|---|
| recall@5 | 56,8% | 24,0% | 11,5% | 7,5% |
| acaso | 20% | 5% | 1% | 0,4% |

(Medida no dicionário anterior à troca da imputação, que mudou a última coluna
para 7,2%. A forma da curva é o que importa aqui, e ela não mudou.)

Essa tabela acabou sendo a coisa mais importante medida nesta fase, e não pelo
motivo esperado: ela é a receita do
[vocabulário núcleo](#o-vocabulário-núcleo-a-promessa-que-cabe-no-dado).

Engenharia de feature sobre DTW também não: z-normalizar as sequências leva o
recall@5 de 16,4% para 20,2% num vocabulário de 250 — ganho real, longe de
suficiente. (E que **não atravessa** para o encoder: lá a mesma z-normalização
derruba 10,3% para 7,1%. Um ganho medido sobre uma representação não é um ganho
da tarefa.)

É aqui, e só aqui, que o **encoder neural com metric learning** se justifica:
ele tem 7,2% para bater, medido no mesmo protocolo.

### O encoder neural, e o que ele comprou

[`libras/encoder.py`](libras/encoder.py) e
[`training/train_sinais.py`](training/train_sinais.py). GRU bidirecional 2×256
sobre as sequências de 32 frames, embedding de **256d** na esfera unitária,
treinado com **ArcFace** sobre as 1.363 classes. Entram os mesmos 4.086
protótipos já extraídos — **nenhum vídeo foi reprocessado** para trocar de
arquitetura.

```
leave-one-articulator-out, 4.086 consultas, três encoders (um por rodízio)
recall@5  16,4%     recall@1  7,2%     MRR  10,5%
```

| | DTW | encoder |
|---|---|---|
| recall@5 | 7,2% | **16,4%** |
| recall@1 | 3,1% | 7,2% |
| MRR | 4,5% | 10,5% |

**2,3x a baseline**, no mesmo protocolo, com o mesmo dado e a mesma busca. E
ainda assim 16,4% em 1.363 palavras não é um dicionário que se usa — é a
diferença entre "a representação era o gargalo" (era) e "a representação era o
único gargalo" (não era). Com três gravações por sinal, o que falta a partir
daqui é dado.

**Vocabulário aberto: 19,3% de recall@5** em 200 sinais que o encoder nunca viu
no treino e que entram no índice só como protótipo. É *acima* dos 16,4% do
vocabulário inteiro, e isso diz uma coisa importante: o ganho não vem de decorar
as classes vistas. A promessa "adiciono um sinal novo com uma gravação e ele
passa a ser encontrável" se sustenta — no patamar atual, que ainda não é o de um
dicionário utilizável.

#### A entrada importou mais que a arquitetura

O encoder original media 10,3%. Nada mudou na GRU, no ArcFace ou no protocolo
para ele chegar a 16,4% — **as duas mudanças foram no que entra nele**:

| mudança | efeito |
|---|---|
| imputação PCHIP no lugar da spline cúbica | tirou as coordenadas de até ±417 larguras de ombro que a cúbica inventava dentro dos buracos |
| configuração de mão em escala própria | **+9,4 pontos** de recall@5 no núcleo, três sementes de cada lado |

A segunda merece a explicação. A normalização dos sinais ancora tudo nos ombros,
que é o certo para localização e trajetória — mas sob essa escala a diferença
entre punho fechado e mão espalmada é uma fração de largura de ombro, ordens de
grandeza abaixo da variação do braço. A rede não estava aprendendo pouco sobre a
configuração de mão: não estava aprendendo nada. `sequencia.maos_locais`
acrescenta as duas mãos ancoradas no **pulso** e divididas pela **própria
escala**, 120 números ao lado dos 294 de posição e velocidade. É a mesma
informação, numa escala em que ela consegue competir.

**Cinco outras hipóteses foram medidas e rejeitadas** — oclusão, TTA, velocidade
de mão, pré-treino, dropout —, e o que permitiu rejeitá-las foi medir antes o
piso de ruído: três sementes do mesmo experimento variam ±1,3 ponto neste
protocolo. Quatro delas teriam entrado no projeto sem esse número, duas piorando
o produto. O registro completo está nas
[notas de construção](docs/notas-de-construcao-do-dicionario.md).

Três decisões de arquitetura, e o porquê de cada uma:

| decisão | alternativa | por quê |
|---|---|---|
| GRU | Transformer | 32 frames e 2 exemplos por classe: atenção não tem onde brilhar, e a recorrência já traz "isto é trajetória" de graça |
| ArcFace | triplet | triplet precisa de mineração de negativos difíceis, que é frágil com 2 exemplos por classe |
| posição + velocidade | posição z-normalizada | medido: 10,3% contra **7,1%**. Centrar cada canal apaga a **localização**, que é fonema em Libras — PAI e MÃE são a mesma mão em lugares diferentes |

O protocolo é o que custa: **três treinos, um por articulador de fora**, mais um
quarto com os três para o modelo que o app usa. 9,1 min no total num M5 com MPS
no vocabulário inteiro, 1,1 min no núcleo. Nenhuma época foi escolhida olhando o
rodízio — não há early stopping, e o número é o da última época.

Dois efeitos colaterais bem-vindos: a busca cai de **90 ms para 1,4 ms** (produto
de matrizes contra 1.024 passos de programação dinâmica) e o dicionário em disco
cai de **70 MB para 3 MB**. Nenhum dos dois era problema, e nenhum dos dois é
argumento a favor do torch — só param de ser considerações.

O app troca sozinho: quando `data/sinais/dicionario_embeddings.npz` existe, a
representação passa a ser embedding e a métrica passa a ser cosseno. Sem ele,
continua na baseline DTW. A re-ancoragem (`1`–`5`) funciona nos dois — os seus
protótipos vão para um arquivo por métrica, porque um embedding não cabe no npz
das sequências.

### O vocabulário núcleo: a promessa que cabe no dado

16,4% de recall@5 não é um dicionário utilizável, e o diagnóstico já estava
escrito acima: com três gravações por sinal, nenhuma arquitetura responde "qual
destas 1.363?". O que sobrou foi mudar a pergunta. **Parar de prometer 1.364
palavras.**

[`libras/nucleo.py`](libras/nucleo.py) é uma lista curada de **163 sinais** do
cotidiano — saudações, cortesia, pessoas e família, necessidades e saúde,
lugares, tempo, perguntas, verbos do dia a dia, sentimentos, cores, comida,
objetos. É o que o app indexa por padrão. Os `LETRA A`…`LETRA Z` do V-LIBRASIL
ficam de fora: o alfabeto manual é o outro modo do projeto, e ele acerta.

O recorte foi medido nos dois caminhos, no mesmo leave-one-articulator-out:

| | índice | recall@5 | recall@1 | MRR |
|---|---|---|---|---|
| DTW | 1.363 sinais | 7,2% | 3,1% | 4,5% |
| encoder treinado em tudo | 1.363 sinais | 16,4% | 7,2% | 10,5% |
| DTW | núcleo | 20,8% | 8,2% | 12,6% |
| encoder treinado em tudo | núcleo | 36,1% | 20,6% | 26,0% |
| **encoder treinado no núcleo** | **núcleo** | **45,5%** | **26,5%** | **33,6%** |

**6,3x a baseline original**, e o caminho até lá tem três partes que se somam:
recortar o índice vale 2,9x sozinho (7,2% → 20,8%, sem tocar em nada), aprender a
distância vale mais 1,7x (20,8% → 36,1%), e treinar o encoder no vocabulário que
ele vai responder vale mais 1,3x.

A última linha foi a surpresa. Treinar em 1.363 classes dá ao encoder **8x mais
gravações** para aprender o que é trajetória de mão, e a medida de vocabulário
aberto (19,3%, acima dos 16,4% do vocabulário inteiro) sugeria que essa
representação genérica transferiria bem. Transfere —
36,1% é bem melhor que 16,4% —, mas perde para concentrar a capacidade nas 163
classes que importam, com 2 gravações de cada. Mais dado genérico não bateu menos
dado específico.

O treino do núcleo custa **1,1 min** contra 9,1 min do vocabulário inteiro, e o
modelo do app é o mesmo arquivo de sempre:

```
python training/eval_sinais.py --nucleo          # a baseline DTW do núcleo
python training/train_sinais.py --treino-nucleo
python -m libras.app --sinais                    # usa o núcleo
python -m libras.app --sinais --tudo             # as 1.364, se você preferir
```

O `--tudo` continua funcionando porque o índice em disco guarda os 4.086
protótipos mesmo quando o treino vê só o núcleo. O recorte acontece ao carregar,
e vale só para o que veio do V-LIBRASIL: **o que você ensina entra sempre**,
esteja ou não na lista curada. Ensinar um sinal de fora é justamente o caso em
que o recorte não pode atrapalhar.

### O limiar de rejeição, e o pouco que ele compra

Com 163 sinais no índice, quase tudo que você pode sinalizar está fora dele — e
sem limiar a busca responde assim mesmo, devolvendo os cinco mais parecidos com
cara de resposta. É a versão em sinais do problema que `CONFIANCA_REJEICAO`
resolve no alfabeto.

O corte saiu do mesmo protocolo, de graça: as gravações dos **1.200 sinais fora
do núcleo**, feitas pelo articulador de teste, são consultas que não têm resposta
possível. Foram 267 delas contra as 490 de dentro.

Há dois critérios possíveis, e eles perguntam coisas diferentes. **Distância**:
"isto parece com alguma coisa?". **Margem** entre o primeiro e o segundo
candidato: "isto parece com *uma* coisa?" — um sinal que o dicionário não tem
cai no meio de vários protótipos parecidos e não destaca nenhum. Os dois foram
calibrados na mesma cobertura de 95% dos acertos:

| critério | corte | listas com resposta | invenção cortada |
|---|---|---|---|
| **distância** | **0,5384** | **47,0%** | **10,5%** |
| margem | 0,0087 | 46,5% | 8,6% |

A distância ganha, e é ela que fica ligada. Ligar as duas não soma: cada uma
gasta seus 5% de cobertura por conta própria, e juntas jogariam fora ~10% dos
acertos para cortar pouco mais que a melhor delas sozinha.

A tabela completa da distância, para quem quiser outro ponto de operação:

| cobertura dos acertos | corte | listas com resposta | invenção cortada |
|---|---|---|---|
| 99,1% | 0,5803 | 45,9% | 2,6% |
| **95,1%** | **0,5384** | **47,0%** | **10,5%** |
| 90,1% | 0,5038 | 48,4% | 19,9% |
| 80,3% | 0,4603 | 51,0% | 35,6% |

**As duas distribuições se sobrepõem, e a tabela não esconde isso.** No corte
escolhido o app recusa 10,5% das consultas sem resposta e paga pouco mais de um
ponto de recall@5 por isso. A distância do cosseno não é confiança, e enquanto
ela for o único sinal disponível a rejeição fica nesse patamar — presente,
honesta, e pequena.

A escolha de 95% vem do custo assimétrico dos dois erros: num dicionário, uma
lista errada custa um olhar (você não reconhece nenhuma das cinco e refaz o
sinal), e um acerto recusado custa a resposta que a pessoa procurava. Maximizar
o J de Youden ignora essa assimetria: ele trata os dois erros como iguais e
escolhe um corte que joga fora a maior parte dos acertos para ganhar recusas que
custam pouco.

Em `--tudo` a rejeição fica **desligada**: o limiar foi calibrado com 163 sinais
no índice, e com 1.364 o vizinho mais próximo é sempre mais próximo. O mesmo
corte passaria a aceitar quase tudo, e um limiar frouxo é pior que nenhum — ele
promete uma recusa que não acontece.

As distâncias medidas ficam em `models/confiancas_rejeicao.npz`, para que
escolher outro corte não custe três treinos.

---

## Estrutura do projeto

Os **sinais ocupam o topo do pacote** e o alfabeto se recolhe num subpacote. O
layout responde qual dos dois é o projeto sem que ninguém precise perguntar.

```
libras/
  app.py           só o CLI: lê os argumentos e escolhe o modo
  app_sinais.py    o laço do dicionário reverso
  pose.py          os 49 pontos e a normalização ancorada no corpo
  sequencia.py     imputação PCHIP, reamostragem, validade, canais de mão
  segmenter.py     quando um sinal começa e quando acaba
  dtw.py           distância com alinhamento temporal, em lote
  encoder.py       GRU bidirecional → embedding 256d (único lugar com torch)
  dicionario.py    protótipos, busca top-5, re-ancoragem, recorte de vocabulário
  rejeicao.py      quando dizer "não reconheci", e a calibração dos dois cortes
  nucleo.py        os 163 sinais que o dicionário promete acertar
  avaliacao.py     recall@1/@5, MRR, leave-one-articulator-out
  catalogo.py      rótulo e articulador a partir do caminho do vídeo
  detector.py      MediaPipe 2 mãos + pose (fino)
  ui.py            overlay do dicionário (fino)

  config.py        parâmetros dos dois modos           ─┐
  camera.py        captura da webcam (fino)             │ compartilhado
  desenho.py       cores, fonte e faixas (fino)         │
  mediapipe_io.py  primitivas do MediaPipe            ─┘

  alfabeto/        a fase 1, inteira
    app.py         o laço da datilologia
    landmarks.py   MediaPipe + a normalização dos 21 pontos
    classifier.py  carrega o modelo treinado, prediz e rejeita
    stabilizer.py  decide quando uma predição vira letra
    speller.py     acumula letras em texto, espaço por ausência
    practice.py    sessão do modo prática (lógica pura)
    sampling.py    filtro de diversidade da coleta (lógica pura)
    augment.py     aumento de dados no espaço dos landmarks
    recovery.py    cascata de variantes para imagens difíceis
    ui.py          overlay do alfabeto (fino)

training/
  prepare_dataset.py   baixa a base pública e extrai landmarks
  collect.py           grava suas amostras pela webcam
  train.py             compara modelos, treina e salva o relatório
  prepare_sinais.py    extrai os vídeos do V-LIBRASIL
  eval_sinais.py       leave-one-articulator-out da baseline DTW
  train_sinais.py      treina o encoder e mede se ele paga o torch

tests/            407 testes, espelhando a mesma divisão
scripts/
  download_model.sh       modelos do MediaPipe (~17 MB)
  download_vlibrasil.sh   base de sinais do espelho no Kaggle (~10,8 GB)
  organizar_vlibrasil.py  layout plano do espelho → pasta por articulador
models/           hand_landmarker.task, pose_landmarker.task, classifier.joblib,
                  encoder_sinais.pt
data/             dataset_publico.npz, coletados/*.npy, sinais/, raw/
```

**Por que `mediapipe_io.py` existe.** Três detectores vivem aqui — uma mão em
vídeo (alfabeto), uma mão em foto solta (base pública) e duas mãos mais o corpo
(sinais) — e os três repetiam a mesma conferência de modelo, a mesma montagem de
array e a mesma leitura de lateralidade. Código que envolve biblioteca externa
envelhece mal em triplicata: o MediaPipe já quebrou a API uma vez, na 1.0,
levando junto o `mp.solutions` inteiro. Da próxima, a correção acontece num
lugar.

`data/` e os pesos treinados não são versionados — são regeráveis e pesados. Os
**relatórios** ficam (`models/relatorio_*.txt`, `models/confiancas_rejeicao.npz`):
eles são a medida, e a medida é a parte do trabalho que não se regenera de graça.

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

Do modo sinais:

| parâmetro | padrão | efeito |
|---|---|---|
| `LIMIAR_MOVIMENTO` | 0,35 | larguras de ombro/s para o sinal começar |
| `FRAMES_PARA_INICIAR` | 3 | frames em movimento antes de gravar |
| `SEGUNDOS_REPOUSO` | 0,5 | mão parada por este tempo fecha o sinal |
| `SEGUNDOS_MINIMO_SINAL` | 0,4 | abaixo disso foi ruído, descarta |
| `SEGUNDOS_MAXIMO_SINAL` | 4,0 | corta em vez de gravar para sempre |
| `CANDIDATOS_NA_TELA` | 5 | quantas palavras a busca devolve |
| `BANDA_DTW` | 0,2 | desvio máximo da diagonal no alinhamento |
| `REJEICAO_COSSENO` | ver `config.py` | acima desta distância o app diz "não reconheci"; `None` desliga |
| `REJEICAO_MARGEM` | ver `config.py` | abaixo desta diferença entre o 1º e o 2º candidato, a lista é um empate e o app recusa |

Do encoder neural:

| parâmetro | padrão | efeito |
|---|---|---|
| `DIMENSAO_EMBEDDING` | 256 | tamanho do vetor que substitui a sequência |
| `ENC_OCULTO` / `ENC_CAMADAS` | 256 / 2 | unidades por direção da GRU, e profundidade |
| `ENC_DROPOUT` | 0,3 | com 2 gravações por classe, sem isso ele decora |
| `ENC_EPOCAS` | 60 | épocas por rodízio |
| `ARCFACE_MARGEM` | 0,30 | folga angular exigida; sobe de 0 no primeiro terço |
| `ARCFACE_ESCALA` | 32,0 | dureza do softmax sobre a esfera |
| `ENC_MAOS_LOCAIS` | `True` | a configuração de mão em escala própria como canal extra |
| `ENC_AUG_*` | — | rotação, escala, ruído e deformação de ritmo do aumento |
| `ENC_AUG_OCLUSAO` | 0,0 | apagar uma mão por um trecho; medido e desligado |
| `ENC_TTA` | 1 | versões aumentadas por gravação ao codificar; medido e desligado |
| `VOCABULARIO_ABERTO` | 200 | sinais fora do treino, medidos à parte |
| `BASELINE_DTW_RECALL5` | ver `config.py` | o número a bater no vocabulário inteiro |
| `BASELINE_DTW_NUCLEO_RECALL5` | ver `config.py` | o número a bater no núcleo |

O vocabulário núcleo em si é uma lista, não um parâmetro: está em
[`libras/nucleo.py`](libras/nucleo.py), agrupado por tema. Acrescentar uma
palavra é acrescentar uma linha — e vale rodar o treino de novo, porque o
encoder é treinado nas classes que ele vai responder. `nucleo.ausentes` recusa
uma palavra que não exista no V-LIBRASIL em vez de deixá-la sumir em silêncio.

---

## Testes

```bash
python -m pytest tests/ -q
```

407 testes, ~5s, sem câmera, sem vídeo, sem dataset e sem modelo treinado.
Cobrem normalização (das duas), estabilização, soletração, aumento de dados,
recuperação de imagens, rejeição, diversidade da coleta, modo prática, imputação
temporal, segmentação de sinais, DTW, busca no dicionário, re-ancoragem, as
métricas de avaliação, e o encoder — toda a lógica que pode dar errado.

Os testes do encoder cobrem o que quebraria em silêncio: embedding sem norma 1
faria o cosseno ordenar por tamanho em vez de forma; hiperparâmetros perdidos no
`.pt` fariam a entrada ser montada de um jeito no treino e de outro no app; um
aumento que espelhasse o sinalizante treinaria a rede no rótulo errado; e o
vazamento do vocabulário aberto para dentro do treino daria um número bonito e
mentiroso. Os que dependem de torch se pulam sozinhos quando ele não está
instalado.

O que não está testado é o que não dá para testar sem hardware: a captura da
webcam, o desenho na tela e os dois wrappers do MediaPipe. Todos foram mantidos
finos justamente por isso.

---

## Limitações conhecidas

### No alfabeto

**Letras com movimento são tratadas como pose.** H, J, K, X e Z têm trajetória
em Libras, mas aqui são classificadas por um frame só. A medida que expõe isso:
a nuvem do J tem raio 1,43 contra ~0,5 típico, e invade M, C, A e L — as 200
amostras não são uma pose, são uma trajetória inteira achatada num rótulo só.

**Mãos fechadas são mal detectadas.** Quando os dedos se escondem atrás da
palma, o MediaPipe estima posições ruins. Afeta T principalmente, e afetava M e
N antes da cascata de recuperação.

**A métrica não descreve o uso real.** Treino e teste saem das mesmas sessões.

### Nos sinais

**Três articuladores.** O V-LIBRASIL tem uma gravação por pessoa por sinal, e o
modelo tem todo incentivo para aprender as pessoas. O leave-one-articulator-out
mede isso e a re-ancoragem corrige no uso — mas a base é essa.

**recall@5 de 45,5% no vocabulário núcleo.** Medido, não estimado: 490 consultas
em leave-one-articulator-out. A busca está correta (auto-recuperação 6/6) e os
vetores são sadios — o que falha é a representação generalizar entre pessoas, e
as três coisas que ajudaram foram recortar o vocabulário, treinar o encoder nele
e dar à rede a configuração de mão em escala própria.

**45,5% ainda não é um dicionário que você usa sem pensar**: em uma de cada duas
consultas a palavra certa não aparece. É utilizável para explorar e para
aprender, e é 6x o ponto de partida. Com três gravações por sinal o gargalo que
resta é dado e não objetivo — e a re-ancoragem é o que empurra esse número para
cima na sua própria mão, sem retreinar nada.

**O vocabulário é 163 palavras.** Tudo fora dele o app erra por construção, e o
limiar de rejeição só pega 10,5% desses casos. Sinalizar algo que não está na
lista quase sempre devolve cinco palavras erradas com aparência de resposta.

**Sem expressão facial.** É fonema em Libras, e sinais que só diferem por ela
vão colidir. 468 pontos de rosto sobre três amostras por classe seria ruído
garantido — fica para quando houver mais dado.

**Um sinal por vez.** Consulta isolada com repouso antes e depois; não há
segmentação de frases contínuas.

---

## Próximos passos

~~1. Construir e medir o dicionário de sinais.~~ **Feito.** 4.086 vídeos
extraídos, recall@5 de 7,2% registrado em `models/relatorio_sinais.txt`. Era o
número do qual tudo abaixo dependia.

~~2. Encoder neural com metric learning.~~ **Feito.** Bateu a baseline no mesmo
leave-one-articulator-out, registrado em `models/relatorio_encoder.txt`.

~~3. Vocabulário núcleo.~~ **Feito.** 163 sinais do dia a dia. Foi a resposta
para "o alfabeto funciona e os sinais não": a promessa era grande demais para o
dado.

~~4. Imputação que não inventa.~~ **Feito.** A spline cúbica disparava entre
pontos válidos distantes e produzia coordenadas de até 417 larguras de ombro;
PCHIP baixou o máximo para 5,8.

~~5. A configuração de mão como canal próprio.~~ **Feito.** Sob a normalização
ancorada no corpo, os dedos variam ordens de grandeza menos que a trajetória, e
a rede simplesmente não os via. Ancorados no pulso e em escala de mão, valem
**+9,4 pontos** de recall@5 — a única mudança de representação que passou do
ruído de semente (±1,3 ponto) por margem larga.

Cinco hipóteses foram medidas e **rejeitadas** no caminho, e ficam registradas
porque medir custou mais que implementar: aumento por oclusão, TTA, velocidade
da mão em escala própria, pré-treino no vocabulário inteiro seguido de
especialização, e variações de dropout. Nenhuma saiu do ruído; duas pioraram.

Em ordem de valor:

1. **Mais de três gravações por sinal — o gargalo continua aqui.** O recorte de
   vocabulário comprou 2,9x sem tocar no dado, e não compra de novo: encolher
   mais a lista começa a tirar palavras que a pessoa quer. Duas saídas: outra base de
   Libras isolada somada ao V-LIBRASIL, ou re-ancoragem em escala — as gravações
   que a própria pessoa faz no uso já são protótipos e já entram sem retreino.
2. **Pré-treino numa base com muitos sinalizadores.** A raiz do problema é
   invariância a pessoa, e nenhuma base de Libras isolada tem mais de três. Uma
   base grande de outra língua de sinais (AUTSL tem 43 sinalizadores) ensinaria
   isso, e o encoder transferiria: os 36,1% da linha "treinado em tudo" já
   provam que a representação atravessa vocabulários. É o item mais caro e o de
   maior teto.
3. **A máscara de validade na representação.** Ela é calculada, viaja em
   `Sequencia.validade` e é **descartada** por `vetores` — hoje um ponto imputado
   entra na distância como se tivesse sido medido. Usá-la exige guardar a máscara
   junto dos protótipos, o que muda o formato do `dicionario.npz` e custa uma
   re-extração de 55 min. É a última melhoria barata que sobrou na representação.
4. **Uma rejeição que funcione.** A atual corta 10,5% das consultas sem resposta
   porque a distância do cosseno não é confiança. A margem entre o primeiro e o
   segundo candidato foi medida e é *pior* (8,6%) — o que falta não é outro
   critério sobre a mesma distância, é um sinal de confiança de verdade, e o
   único caminho barato para ele é uma cabeça treinada para prever acerto.
5. **Conjunto de avaliação separado do alfabeto.** Uma sessão de gravação em
   outro dia, outra luz, outra roupa, usada só como teste. É o que transforma
   "achei que melhorou" em evidência.
6. **Features geométricas no alfabeto** — ângulos entre dedos e distâncias
   ponta-a-ponta somados às coordenadas cruas, mirando U/V e B/W.
7. **Autocorreção com dicionário PT-BR** — um erro em dez letras arruína a
   palavra inteira; a palavra em andamento já aparece destacada.

---

## Documentação

- [Design da fase 1 — alfabeto](docs/superpowers/specs/2026-08-11-libras-live-design.md)
  — as decisões de arquitetura, as alternativas rejeitadas e o porquê de cada uma.
- [Design da fase 2 — dicionário reverso de sinais](docs/superpowers/specs/2026-08-11-sinais-dicionario-design.md)
  — a restrição de três gravações por sinal e o que ela obriga.
- [Notas de construção do dicionário](docs/notas-de-construcao-do-dicionario.md)
  — o que quebrou entre "o pipeline está pronto" e um recall@5 medido: o site da
  UFPE fora do ar, o `unzip` que corrompe UTF-8, o layout que destruiria a
  avaliação em silêncio, e o bug que eu inventei medindo com média em vez de
  mediana. Traz também o que o encoder comprou, o barato que não atravessou do
  DTW para ele, e a armadilha de apontar a avaliação para o npz errado.

---

## Créditos

- [Brazilian Sign Language Alphabet Dataset](https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset)
  — base pública de imagens do alfabeto.
- [V-LIBRASIL](https://libras.cin.ufpe.br) (UFPE/CIn) — 1.364 sinais de Libras
  em vídeo.
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  e [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
  — detecção dos pontos da mão e do corpo.

Trabalhos que embasaram decisões da fase 2:

- [Proper Body Landmark Subset Enables More Accurate and 5X Faster Recognition
  of Isolated Signs in LIBRAS](https://arxiv.org/abs/2510.24887) (IEEE SAS 2026)
  — o subconjunto de landmarks e a imputação por spline.
- [Representing Signs as Signs: One-Shot ISLR](https://arxiv.org/abs/2502.20171)
  — busca vetorial em vocabulário grande, e a referência de 50,8% de MRR.
- [ArcFace: Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698)
  — a perda do encoder, e a razão de ela existir: margem angular em vez de
  fronteira justa.

Dependências: `mediapipe`, `opencv-python`, `scikit-learn`, `scipy`, `numpy`,
`joblib`, `pytest` e `torch` — este último só para o encoder dos sinais. Versões
exatas em [`requirements.txt`](requirements.txt).
