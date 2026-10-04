# Music From Scratch

Modelo de musica **treinado do zero**. Sem MusicGen, sem pesos pre-treinados, sem HuggingFace.

## Arquitetura

- Transformer decoder (~5M parametros) escrito em PyTorch puro
- Treinado em tokens MIDI (pitch, duracao, velocity, canal)
- Vocab de 149 tokens
- 6 camadas, d_model=256, 8 cabecas

## Pipeline

1. midi_gen.py gera 600 MIDIs sinteticos (varios estilos)
2. dataset.py tokeniza MIDI para sequencias
3. train.py treina o transformer do zero (loss cai de ~5 para <2)
4. generate.py amostra autoregressivamente e vira MIDI
5. fluidsynth + soundfont renderiza WAV

## Como rodar

1. **Actions > Train From Scratch > Run workflow**
2. Depois: **Actions > Generate Music > Run workflow**
3. Escolha estilo: random, calm, tense, epic, chaotic

## Soundfonts

Coloque .sf2 em soundfonts/. Se nao tiver, o script gera so MIDI.

## Estilos

- calm: temperatura 0.7, top_k 20
- tense: temperatura 0.9, top_k 30
- epic: temperatura 1.05, top_k 40
- chaotic: temperatura 1.3, top_k 80
