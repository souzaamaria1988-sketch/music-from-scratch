# Music From Scratch v2

Transformer treinado do zero. Auto-gera MIDIs em training_data/midi e renderiza WAV em training_data/audio com soundfonts.

## Estrutura

- training_data/midi/ - MIDIs auto-gerados
- training_data/audio/ - WAVs renderizados
- soundfonts/ - seus .sf2/.sf3
- models/best.pt - modelo treinado
- generated/ - musicas geradas

## Como rodar

1. Actions > Train From Scratch > Run workflow
2. Actions > Generate Music > Run workflow
3. Estilos: random, calm, tense, epic, chaotic