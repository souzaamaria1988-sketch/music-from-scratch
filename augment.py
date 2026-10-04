#!/usr/bin/env python3
import argparse,shutil
from pathlib import Path
from mido import MidiFile,MidiTrack,Message,MetaMessage
def transpose(midi_in,midi_out,semitones):
    mf=MidiFile(str(midi_in));out=MidiFile(ticks_per_beat=mf.ticks_per_beat)
    for track in mf.tracks:
        new=MidiTrack()
        for msg in track:
            if msg.type in ('note_on','note_off'):
                new.append(msg.copy(note=max(0,min(127,msg.note+semitones))))
            else:new.append(msg.copy())
        out.tracks.append(new)
    out.save(str(midi_out))
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--src',nargs='+',default=['training_data/midi','music_input'])
    ap.add_argument('--out',default='training_data/augmented')
    ap.add_argument('--steps',type=int,default=6)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    files=[]
    for src in a.src:
        p=Path(src)
        if p.exists():files+=list(p.glob('*.mid'))+list(p.glob('*.midi'))
    print('Fonte: '+str(len(files))+' MIDIs')
    shifts=[-6,-4,-2,2,4,6][:a.steps]
    n=0
    for f in files:
        try:
            shutil.copy(f,out/(f.stem+'_orig.mid'));n+=1
            for s in shifts:
                transpose(f,out/(f.stem+'_t'+str(s)+'.mid'),s);n+=1
        except Exception as e:print('skip',f,e)
    print('Augmented: '+str(n)+' arquivos em '+str(out))