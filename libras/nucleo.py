"""O vocabulário núcleo: os sinais que o dicionário promete acertar.

O V-LIBRASIL tem 1.364 sinais e três gravações de cada um. Medido no
leave-one-articulator-out, o dicionário inteiro dá **7,5% de recall@5** com DTW e
10,3% com o encoder — e a mesma medição, recortada por tamanho de vocabulário,
mostra de onde vem o problema:

    sinais no índice     25      100      500     1.363
    recall@5           56,8%    24,0%    11,5%     7,5%

O gargalo não é a busca nem a perda de treino: é responder "qual destes 1.363?"
com duas gravações por classe. Nenhuma arquitetura resolve isso; o que resolve é
**parar de prometer 1.364 sinais**.

Este módulo é essa promessa menor e cumprida. Um recorte curado de sinais do
cotidiano — saudações, cortesia, família, necessidades, tempo, perguntas — no
qual o dicionário funciona de verdade, em vez de um vocabulário completo no qual
ele erra quase sempre.

**O que fica de fora não some.** Os seus protótipos, os que você ensina com a
tecla `N` ou corrigindo um resultado, entram no índice mesmo fora do núcleo: o
recorte é do que veio do V-LIBRASIL, não do que é seu. E
`python -m libras.app --sinais --tudo` devolve as 1.364 palavras para quem
preferir cobertura a acerto.

Os `LETRA A`…`LETRA Z` do V-LIBRASIL ficam fora por outro motivo: o alfabeto
manual é o outro modo do projeto, com um classificador próprio que acerta,
e datilologia parada não é o que o segmentador de sinais foi feito para pegar.
"""

from __future__ import annotations

from . import catalogo

# As palavras, agrupadas pelo motivo de estarem aqui. A grafia é a do
# V-LIBRASIL, incluindo as desambiguações entre parênteses, porque a comparação
# passa por `catalogo.chave` mas o que aparece na tela é o rótulo do dicionário.

SAUDACOES = [
    "OI",
    "BEM",
    "TUDO CERTO",
    "OBRIGADO",
    "POR FAVOR",
    "COM LICENÇA",
    "DESCULPA",
    "PARABÉNS",
    "SIM",
    "NÃO",
    "TALVEZ",
    "VENHA",
    "ESPERE",
    "VAMOS",
]

PESSOAS = [
    "PESSOA",
    "HOMEM",
    "MULHER",
    "MENINA",
    "GAROTO (MENINO)",
    "CRIANÇA",
    "BEBÊ",
    "AMIGO",
    "FAMÍLIA",
    "MÃE",
    "PAI",
    "FILHO",
    "FILHA",
    "IRMÃ",
    "AVÓ",
    "TIO",
    "TIA",
]

IDENTIDADE = [
    "NOME",
    "SURDO",
    "USE LÍNGUA DE SINAIS",
    "LÍNGUA (IDIOMA)",
    "SOLETRAR",
    "INTERPRETAR",
    "APARELHO AUDITIVO",
    "COMUNICAR",
]

NECESSIDADES = [
    "ÁGUA",
    "COMER",
    "BEBIDA",
    "FOME",
    "COM SEDE",
    "BANHEIRO",
    "DORMIR",
    "BANHO",
    "AJUDAR",
    "DOENTE",
    "DOR DE CABEÇA",
    "MEDICAMENTO",
    "HOSPITAL",
    "MÉDICO (DOUTOR)",
    "PERIGO",
    "EMERGÊNCIA",
]

LUGARES = [
    "CASA",
    "ESCOLA",
    "UNIVERSIDADE",
    "EMPREGO (TRABALHO)",
    "CIDADE",
    "RESTAURANTE",
    "IGREJA",
    "CARRO",
    "AVIÃO",
]

TEMPO = [
    "AGORA",
    "ONTEM",
    "AMANHÃ",
    "DIA",
    "NOITE",
    "TARDE",
    "MANHÃ",
    "ANO",
    "HORA",
    "TEMPO",
    "SEMPRE",
    "NUNCA",
    "DEPOIS (MAIS TARDE)",
    "ANTES",
    "SEGUNDA FEIRA",
    "TERÇA FEIRA",
    "QUARTA FEIRA",
    "QUINTA FEIRA",
    "SEXTA FEIRA",
    "SÁBADO",
    "FIM DE SEMANA",
]

PERGUNTAS = [
    "O QUÊ?",
    "QUEM",
    "ONDE",
    "QUANDO",
    "POR QUE",
    "COMO",
    "QUAL",
    "QUANTOS",
]

VERBOS = [
    "SABER",
    "COMPREENDER",
    "APRENDER",
    "ENSINAR",
    "ESTUDO",
    "DIZER",
    "PEDIR",
    "RESPOSTA",
    "VER",
    "OUVIR",
    "GOSTAR",
    "QUER",
    "PODER",
    "TER",
    "IR",
    "DAR",
    "PEGAR",
    "COMPRAR",
    "PAGAR",
    "ESPERAR",
    "ESQUECER",
    "LEMBRAR (TER EM MENTE)",
    "ESCREVER",
    "SENTAR",
    "PARAR",
]

SENTIMENTOS = [
    "FELIZ",
    "TRISTE",
    "ÓDIO",
    "COM MEDO",
    "NERVOSO",
    "CANSADO",
    "CALMO",
    "BONITO",
    "FEIO",
    "RUIM",
    "MELHOR",
    "DIFÍCIL",
    "IMPORTANTE",
    "ENGRAÇADO",
    "SURPRESA",
    "EU AMO VOCÊ",
]

CORES = [
    "AZUL",
    "VERMELHO",
    "VERDE",
    "AMARELO",
    "PRETO",
    "BRANCO",
    "ROSA",
    "ROXO",
    "LARANJA",
    "MARROM",
]

COMIDA = [
    "PÃO",
    "LEITE",
    "CAFÉ",
    "CARNE",
    "FRUTA",
    "BANANA",
    "PIZZA",
    "CHOCOLATE",
    "SORVETE",
]

OBJETOS = [
    "TELEFONE",
    "COMPUTADOR",
    "INTERNET",
    "LIVRO",
    "DINHEIRO",
    "CHAVE",
    "JANELA",
    "CAMA",
    "SAPATO",
    "ÓCULOS",
]

GRUPOS = {
    "saudações e cortesia": SAUDACOES,
    "pessoas e família": PESSOAS,
    "identidade e Libras": IDENTIDADE,
    "necessidades e saúde": NECESSIDADES,
    "lugares": LUGARES,
    "tempo": TEMPO,
    "perguntas": PERGUNTAS,
    "verbos do dia a dia": VERBOS,
    "sentimentos e qualidades": SENTIMENTOS,
    "cores": CORES,
    "comida": COMIDA,
    "objetos": OBJETOS,
}

VOCABULARIO: list[str] = [palavra for grupo in GRUPOS.values() for palavra in grupo]

# A forma canônica, que é como o dicionário compara rótulos. `AVÓ` e `AVÔ`
# colapsam aqui, e é por isso que só um dos dois está na lista acima: os dois
# rótulos já são a mesma chave, e escrever os dois sugeriria duas classes.
CHAVES: frozenset[str] = frozenset(catalogo.chave(p) for p in VOCABULARIO)


def contem(rotulo: str) -> bool:
    """Se este rótulo faz parte do vocabulário núcleo."""
    return catalogo.chave(rotulo) in CHAVES


def ausentes(vocabulario_disponivel: list[str]) -> list[str]:
    """As palavras do núcleo que **não** existem no vocabulário dado.

    Existe para falhar alto: um erro de grafia nesta lista tiraria um sinal do
    índice em silêncio, e o dicionário ficaria menor sem que nada quebrasse.
    """
    disponiveis = {catalogo.chave(r) for r in vocabulario_disponivel}
    return sorted(p for p in VOCABULARIO if catalogo.chave(p) not in disponiveis)
