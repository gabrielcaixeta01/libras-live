"""Reconhecimento do alfabeto de Libras — datilologia, letra a letra.

Foi a fase 1 do projeto e continua inteira: uma mão, um frame, uma letra,
normalizada pelo pulso. Ela ficou num subpacote quando os sinais viraram o
assunto principal, mas não mudou de comportamento — só de endereço.

A diferença que importa em relação ao pacote de cima: aqui a normalização
**apaga** onde a mão está, porque uma letra é só a forma da mão. Nos sinais isso
seria fatal, e é por isso que os dois têm normalizações separadas em vez de uma
função com um parâmetro.

Desenho em docs/superpowers/specs/2026-08-11-libras-live-design.md.
"""
