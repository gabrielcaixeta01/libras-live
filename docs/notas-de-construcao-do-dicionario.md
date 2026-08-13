# Notas de construção do dicionário de sinais

O que foi preciso para sair de "o pipeline está pronto" até um `recall@5`
medido, e o que cada obstáculo ensinou. Escrito depois de construir o
dicionário pela primeira vez, em 11 de agosto de 2026.

Isto não é o design — esse está em
[Design da fase 2](superpowers/specs/2026-08-11-sinais-dicionario-design.md).
Isto é o registro do que deu errado no caminho, para que a próxima pessoa não
pague de novo.

---

## O que está funcionando

**O alfabeto está pronto para uso.** macro-F1 de 0,969 em teste real, e 98,7%
de acerto quando reconferido contra as amostras em `data/coletados/`. As
predições abaixo do limiar viram `?` em vez de virar letra errada, que é o
comportamento desejado.

**O pipeline de sinais funciona ponta a ponta.** 4.086 vídeos entraram, 4.086
representações saíram — nenhum descarte, nenhum arquivo ilegível, nenhuma
gravação sem âncora. O caminho de runtime do app (`vídeo → detector →
sequência → busca`) foi verificado sobre vídeos reais e devolve candidatos.

**A busca é rápida o bastante.** 90 ms para varrer os 4.086 protótipos com DTW,
e ela roda uma vez por sinal e não por frame. Não há gargalo de latência aqui e
não há o que otimizar.

**A extração é paralela.** Oito processos trazem o trabalho de 2-5 horas para
~55 minutos num laptop de 10 núcleos. O limite é RAM e não núcleos: cada worker
carrega os dois modelos do MediaPipe.

---

## Onde a dificuldade estava

### A base não estava onde a documentação dizia

`libras.cin.ufpe.br` responde **502 em todas as rotas**. O DNS resolve e o proxy
da UFPE está de pé — é o backend do V-LIBRASIL que não está. O README mandava
baixar manualmente de um site fora do ar, e o projeto inteiro parou aí.

A saída foi o espelho no Kaggle, que serve o bundle de 10,8 GB **sem exigir
login**: a rota de download devolve 302 para uma URL assinada do Google Cloud
Storage. Vale saber, porque a intuição diz que Kaggle sempre exige credencial.

**Lição:** quando um passo manual depende de um serviço de terceiro, ele vai
quebrar em silêncio e o sintoma vai aparecer longe da causa — aqui, como "os
vídeos não baixaram".

### O `unzip` do macOS corrompe os nomes do pacote

Todos os 4.088 nomes do zip vêm marcados como UTF-8. O `unzip` do sistema
mesmo assim os destrói (`Abençoar` vira `Aben+?oar`) e **aborta a extração com
um falso "disk full"** — com 757 GB livres no disco.

A mensagem de erro aponta para o lugar errado, que é o que a torna cara. A saída
é extrair com o `zipfile` do Python, que respeita a flag de encoding.

### O espelho tem layout plano, e isso silenciosamente destruiria a avaliação

O pacote entrega tudo numa pasta só, com o articulador no nome do arquivo:

```
videos UFPE (V-LIBRASIL)/data/Abacaxi_Articulador1.mp4
```

Mas `catalogo.articulador_do_caminho` lê o articulador da **pasta**. Sem
reorganizar, os 4.086 vídeos cairiam todos em `desconhecido`, o
leave-one-articulator-out deixaria de separar pessoas, e o recall@5 passaria a
medir memorização de articulador — **exatamente o erro que a fase 1 cometeu com
o L↔G**, de novo, sem que nada parecesse quebrado.

Nada teria falhado. Os testes passariam, a extração terminaria, e o número
sairia bonito e mentiroso. Daí `scripts/organizar_vlibrasil.py` existir como
passo próprio e testado, em vez de uma linha de `mv`.

### Três vídeos estão corrompidos na origem

O catálogo do V-LIBRASIL fala em 4.089 gravações; o espelho traz 4.086. A
diferença está em `error.csv`, distribuído junto: `Congelar_Articulador3`,
`Criança_Articulador3` e `Poesia_Articulador2` vêm com largura, altura e fps
zerados. O README dizia 4.089 e agora diz 4.086.

### `Avó` e `Avô` são o mesmo sinal para o vocabulário

`catalogo.chave` remove acentos de propósito, para juntar `AMANHÃ` e `AMANHA`
escritos por pessoas diferentes. O efeito colateral é que dois sinais
genuinamente distintos colapsam numa chave só: 1.364 nomes viram 1.363 chaves.

Fica como está. O rótulo exibido na tela preserva o acento, e trocar a regra
para resolver 1 caso em 1.364 quebraria o caso que ela existe para resolver.

### Medir com média sobre dados com outlier inventa um bug

Ao diagnosticar o recall baixo, a decomposição da distância por grupo de canais
acusou razão **1,112** para a mão direita — o mesmo sinal ficando *mais longe*
que dois sinais aleatórios. Isso sugeria fortemente inconsistência de
lateralidade, e havia até uma flag (`--canhoto`) que eu não tinha usado.

Era artefato da medição. A média de normas estava sendo dominada por poucas
amostras com valores de ±400. Refeita com **mediana**, a anomalia sumiu:

| grupo | média | mediana |
|---|---|---|
| mão esquerda | 0,682 | 0,785 |
| mão direita | **1,112** | 0,830 |
| corpo | 0,888 | 0,850 |

O teste independente confirmou que não havia canhoto: os três pares de
articuladores têm distância intra-sinal quase igual (2,79 / 2,93 / 2,90).

**Lição:** num dado que você já sabe ter outliers, a primeira estatística
robusta vem antes da primeira hipótese. Quase gastei uma re-extração de uma hora
consertando um bug que não existia.

**Adendo, 12 de agosto.** A mediana escondeu o outlier, e por isso a pergunta
"de onde vêm os ±400?" ficou sem resposta por um dia. Vinham de um bug real —
ver a seção seguinte. A estatística robusta salvou o diagnóstico *daquela*
hipótese e adiou o diagnóstico da certa.

### Os ±400 eram a spline, e não o dado

A pergunta que a mediana adiou tem resposta simples. `sequencia.imputar`
preenchia os buracos com `scipy.interpolate.CubicSpline` — a cúbica **natural**,
que é global: ela casa a segunda derivada entre os trechos, e para conseguir isso
através de um buraco longo ela sai do intervalo dos valores que o delimitam.

O buraco longo é o caso comum, não o raro: a mão cruza o corpo, o HandLandmarker
a perde por meio segundo, e a spline preenche quinze frames com uma parábola
gigante. Um braço aberto chega a 1,5 largura de ombro. O dicionário tinha pontos
a **417**.

A troca por `PchipInterpolator` — a mesma família de cúbicas, mas preservando a
forma — resolve por construção: dentro de um buraco, PCHIP fica entre os dois
valores que o delimitam. Não há ultrapassagem possível. Medido sobre a mesma
base:

| | CubicSpline | PCHIP |
|---|---|---|
| amplitude máxima | 417,7 | **5,8** |
| gravações com algum \|x\| > 5 | 4,38% | **1,25%** |

O `LIMITE_PLAUSIVEL` de `caracteristicas` limitava a **amplitude** em ±5 e
salvava a rede de um valor absurdo dominar a entrada — mas não salvava a
**forma**: dentro do buraco a trajetória continuava sendo a de uma parábola
inventada, e os canais de configuração de mão, que dividem pela escala da
própria mão, herdavam a invenção inteira. A baseline DTW não tinha nem o corte:
ela consumia os 417 diretamente.

**Lição:** um corte de amplitude é um curativo sobre um valor, não sobre um
processo. Quando um número impossível aparece no dado, ele tem uma causa, e o
lugar de consertá-lo é onde ele nasce.

### Um Ctrl-C descartava trabalho em três lugares

Encontrados ao revisar os laços, todos da mesma família — estado que só existe
em memória até um passo final que a interrupção pula:

| onde | o que se perdia |
|---|---|
| `app_sinais.py` | todos os sinais ensinados na sessão |
| `alfabeto/app.py` | o texto soletrado, sem copiar para a área de transferência |
| `prepare_sinais.py` | até 99 vídeos já extraídos |

O caso do `app_sinais` é o mais caro: a re-ancoragem manual é a mitigação
documentada do viés dos três articuladores. Perdê-la em silêncio anula a
resposta do projeto ao seu próprio problema conhecido. Os três agora usam
`try/finally`.

---

## O número, e por que ele não é um bug

```
leave-one-articulator-out, 4.086 consultas
recall@5  7,5%     recall@1  2,9%     MRR  4,4%
```

Acaso é 0,4%, então a busca aprendeu algo — mas 7,5% não serve como dicionário.
Antes de aceitar, foi descartado defeito de implementação:

| verificação | resultado | o que descarta |
|---|---|---|
| auto-recuperação | 6/6, similaridade ~1,000 | busca e métrica DTW |
| NaN, Inf, vetores constantes | nenhum | extração e imputação |
| amostras sem variação temporal | nenhuma | reamostragem |
| intra-sinal ÷ aleatória | 0,80 | — |
| por grupo (esq./dir./corpo) | 0,79 / 0,83 / 0,85 | lateralidade |
| sem as amostras com \|v\|>5 | idêntico | outliers |

A busca está correta. O que falha é a **representação generalizar entre
pessoas**: o mesmo sinal feito por dois articuladores fica só 20% mais perto que
dois sinais quaisquer, e com 1.363 classes isso produz os 7,5% observados.

Duas saídas baratas foram medidas e recusadas:

**Vocabulário menor não salva** — a fraqueza é uniforme em toda escala:

| sinais | 25 | 50 | 100 | 250 | 500 | 1.363 |
|---|---|---|---|---|---|---|
| recall@5 | 56,8% | 41,9% | 24,0% | 21,1% | 11,5% | 7,5% |
| acaso | 20% | 10% | 5% | 2% | 1% | 0,4% |

**Engenharia de feature sobre DTW também não** (vocabulário de 250):

| variante | recall@5 |
|---|---|
| posição (atual) | 16,4% |
| velocidade | 9,6% |
| posição + velocidade | 15,9% |
| posição z-normalizada | 19,5% |
| z(posição) + z(velocidade) | **20,2%** |
| só as mãos | 15,8% |

O melhor ganho é real e insuficiente: +3,8 pontos sobre uma base que precisaria
multiplicar por quatro.

---

## O encoder, e o que ele comprou

Escrito em 12 de agosto de 2026, um dia depois do resto desta página. O plano da
seção anterior foi executado como estava documentado: GRU bidirecional 2×256
sobre as sequências de 32×147, embedding de 256d, ArcFace sobre o vocabulário —
1.163 classes e ~2.320 gravações em cada rodízio, 1.363 classes e 4.086 gravações
no modelo final —, treinado nos protótipos já extraídos: **nenhum vídeo
reprocessado**.

```
leave-one-articulator-out, 4.086 consultas, três encoders (um por rodízio)
recall@5  10,3%     recall@1  3,8%     MRR  6,1%     ← encoder
recall@5   7,5%     recall@1  2,9%     MRR  4,4%     ← baseline DTW
```

**1,4x.** O alvo era bater 7,5% e ele foi batido. A margem que justificaria o
torch sozinho — 1,5x, escrita no `train_sinais.py` antes de rodar — não foi
alcançada. Aprender a distância foi na direção certa e parou perto: o objetivo de
treino era o diagnóstico correto, e não era o gargalo maior.

O gargalo maior é o mesmo desde o começo: **duas gravações por classe em cada
rodízio**. Nenhuma arquitetura resolve few-shot com o objetivo já certo.

### O que o vocabulário aberto respondeu

Duzentos sinais ficaram fora do treino do encoder e entraram só como protótipo:
**12,3% de recall@5**, contra 10,3% do vocabulário inteiro.

Sinal que o encoder nunca viu se comporta tão bem quanto — melhor que — sinal que
ele treinou. Isso diz duas coisas. A boa: a promessa "adiciono um sinal novo com
uma gravação e ele passa a ser encontrável" se sustenta, e a re-ancoragem não é
conversa. A menos boa: o encoder não está extraindo das classes vistas quase nada
que não generalize também para as não vistas — ele aprendeu uma representação
genérica de gesto, não as 1.363 palavras.

### A z-normalização não atravessou

Era o barato que a seção anterior recomendava: +3,8 pontos sobre o DTW num
vocabulário de 250. Sobre o encoder ela **piora**, e não por pouco:

| entrada | rodízio 1 | rodízio 2 | rodízio 3 | total |
|---|---|---|---|---|
| posição + velocidade | 12,8% | 8,2% | 9,9% | **10,3%** |
| z(posição) + velocidade | 7,2% | 7,4% | 6,8% | 7,1% |

Sete e um décimo é *abaixo* da baseline DTW. O mesmo pré-processamento que dava o
melhor resultado com DTW faz o encoder perder para o DTW.

O motivo estava escrito antes de medir, e medir confirmou: z-normalizar por canal
tira de cada coordenada a sua própria média ao longo do tempo, e isso apaga a
**localização absoluta** do gesto. Localização é fonema em Libras — PAI e MÃE são
a mesma configuração de mão em lugares diferentes do rosto. O DTW não tinha como
aprender a compensar articulador, então trocar localização por robustez valia. O
encoder é treinado exatamente para compensar articulador, e começar jogando fora
a informação é pagar duas vezes.

**Lição:** um ganho medido sobre uma representação não é um ganho da tarefa. Ele
vale para o método que o mediu, e a razão de ele existir é o que decide se
atravessa.

A flag `--z` continua no `train_sinais.py`, agora com número ao lado dela.

### O que ficou de fora, e por quê

**A máscara de validade.** Era o segundo barato recomendado, e é o único item do
plano que não foi executado. Ela é calculada, viaja em `Sequencia.validade` e é
descartada por `vetores` — hoje um ponto imputado por spline entra na distância
como se tivesse sido medido pela câmera.

Usá-la não é uma linha: os protótipos são guardados como `(N, 32, 147)` em
`dicionario.npz` e a máscara não cabe ali. Seria preciso mudar o formato do
dicionário e **re-extrair os 4.086 vídeos** — 55 minutos. Não entrou porque a
fase existia para responder se o encoder paga o torch, e a resposta não dependia
disso. Fica como a próxima melhoria de representação, e agora com o custo
explícito.

### A armadilha que apareceu no caminho

Depois do treino há dois npz parecidos em `data/sinais/`: `dicionario.npz` com
sequências e `dicionario_embeddings.npz` com embeddings. Apontar o
`eval_sinais.py` para o segundo **roda sem erro** e imprime um número alto — o
encoder salvo é o que viu os três articuladores, então o rodízio deixa de separar
pessoas e o placar volta a medir memorização.

É o L↔G com roupa nova, e pela terceira vez a forma dele é a mesma: nada falha,
tudo passa, o número sai bonito. O `eval_sinais.py` agora detecta a métrica e
escreve o aviso no próprio relatório.

---

## O que vem depois

**Mais gravações por sinal.** É o que o encoder mediu, não o que ele supôs: com o
objetivo de treino correto e a representação aprendida, o teto de 10,3% é o teto
de três exemplos por classe. As duas saídas são outra base de Libras isolada
somada ao V-LIBRASIL, ou a re-ancoragem em escala — as gravações que a própria
pessoa faz no uso já são protótipos e já entram sem retreino.

**Depois dela, a máscara de validade** (acima, com o custo).

Enquanto isso o modo `--sinais` serve como demonstração do pipeline e como
ferramenta pessoal: a re-ancoragem (`1`–`5`) contorna o problema para a sua
própria mão, que é o caso de uso que funciona hoje — e agora ela contorna sobre
uma representação 1,4x melhor.
