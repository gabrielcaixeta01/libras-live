"""Reconhecimento de Libras em tempo real, em duas frentes.

O topo do pacote é o **dicionário reverso de sinais**: você faz um sinal, ele
diz que palavra é. Trajetória, duas mãos, e a localização no corpo — que em
Libras é fonema.

O **alfabeto** (datilologia) mora em `libras/alfabeto/`. Foi a primeira fase e
continua inteira; o que ela tem de diferente é justamente o que não serve para
sinais, e por isso os dois não compartilham normalização.

Compartilhado pelos dois: `config`, `camera`, `desenho` e `mediapipe_io`.
"""
