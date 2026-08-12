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

## O que vem depois

**Encoder neural com metric learning — aprovado, agendado.**

É a conclusão que esta fase existia para produzir, e o `eval_sinais.py` já a
escreve sozinho quando o recall@5 fica abaixo de 80%. Fica agendado em vez de
improvisado porque torch é a maior mudança de dependência do projeto e merece
entrar por decisão, não por impulso de quem estava com o terminal aberto.

Quando for a hora:

- **Alvo:** bater 7,5% de recall@5 no mesmo leave-one-articulator-out. Sem esse
  protocolo, qualquer número novo é incomparável com este.
- **Entrada:** os 4.086 protótipos de `data/sinais/dicionario.npz`, já
  extraídos — **não é preciso reprocessar os vídeos nem rebaixar os 10,8 GB.**
  A extração custou 55 minutos e o resultado está salvo.
- **Arquitetura:** GRU ou Transformer pequeno sobre as sequências de 32×147,
  treinado com ArcFace ou triplet — o objetivo é aproximar o mesmo sinal feito
  por pessoas diferentes, que é precisamente a razão 0,80 que precisa cair.
- **Cuidado com o protocolo:** treinar com dois articuladores e validar no
  terceiro, girando. Treinar com os três e avaliar depois reproduziria o erro do
  L↔G num lugar novo.
- **Baratos que valem junto:** a z-normalização acima (+3,8 pontos sozinha) e
  usar a **máscara de validade**, que hoje é calculada, carregada em
  `Sequencia.validade` e descartada por `vetores` — pontos imputados entram na
  distância como se tivessem sido medidos.

Até lá o modo `--sinais` continua servindo como demonstração do pipeline e como
ferramenta pessoal: a re-ancoragem (`1`–`5`) contorna o problema para a sua
própria mão, que é o caso de uso que funciona hoje.
