# Music From Scratch v3

Transformer 15M params com RoPE, treinado do zero.

## Pastas
- training_data/midi/ - MIDIs auto-gerados
- training_data/augmented/ - MIDIs aumentados (transpostos)
- training_data/audio/ - WAVs (opcional)
- music_input/ - SEUS MIDIs de exemplo (entram no treino)
- soundfonts/ - seus .sf2 (usados pra renderizar)
- models/best.pt - modelo treinado
- generated/ - musicas geradas

## Uso
1. Coloque seus MIDIs em music_input/ (opcional)
2. Coloque seus .sf2 em soundfonts/
3. Actions > Train From Scratch > Run workflow
4. Actions > Generate Music > Run workflow

## Melhorias v3
- Modelo 3x maior (15M params)
- RoPE (positional encoding rotativo)
- Augmentation por transposicao
- Repetition penalty na geracao
- Label smoothing
- LR com warmup + cosine
- 4 tracks separados (lead, pad, bass, drums)
- top_p (nucleus) sampling