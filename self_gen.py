#!/usr/bin/env python3
import random,argparse
from pathlib import Path
from mido import MidiFile,MidiTrack,Message,MetaMessage

SCALES={'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],
        'dorian':[0,2,3,5,7,9,10],'phrygian':[0,1,3,5,7,8,10],
        'pent_min':[0,3,5,7,10],'blues':[0,3,5,6,7,10],'harm_min':[0,2,3,5,7,8,11]}
PROGS={'pop':[[0,4,7],[5,9,12],[7,11,14],[5,9,12]],
       'rock':[[0,4,7],[5,9,12],[7,11,14],[0,4,7]],
       'minor':[[0,3,7],[8,12,15],[3,7,10],[10,14,17]],
       'jazz':[[0,4,7,11],[5,9,12,16],[2,5,9,12],[7,11,14,17]],
       'blues':[[0,4,7],[5,9,12],[7,11,14],[0,4,7],[5,9,12],[7,11,14]],
       'cinematic':[[0,3,7],[8,12,15],[5,9,12],[7,11,14]]}
STYLES=list(PROGS.keys())

def make(path,seed,bars,style):
    r=random.Random(seed)
    scale=SCALES[r.choice(list(SCALES.keys()))]
    prog=PROGS[style]
    bpm=r.randint(70,180)
    root=r.choice([57,58,60,62,64,65])
    mf=MidiFile(ticks_per_beat=480);tp=480
    cond=MidiTrack();mf.tracks.append(cond)
    cond.append(MetaMessage('set_tempo',tempo=int(60000000/bpm),time=0))
    lead=MidiTrack();mf.tracks.append(lead)
    lead.append(Message('program_change',program=r.randint(0,7),channel=0,time=0))
    for b in range(bars):
        ch=prog[b%len(prog)]
        for beat in range(4):
            if r.random()<0.75:
                n=root+scale[r.randrange(len(scale))]+r.choice([0,12,12,24])
                n=max(40,min(96,n))
                d=r.choice([tp//2,tp,tp*2]);v=r.randint(70,110)
                lead.append(Message('note_on',note=n,velocity=v,channel=0,time=0))
                lead.append(Message('note_off',note=n,velocity=0,channel=0,time=d))
            else:
                lead.append(Message('note_off',note=0,velocity=0,channel=0,time=tp))
    pad=MidiTrack();mf.tracks.append(pad)
    pad.append(Message('program_change',program=48,channel=1,time=0))
    for b in range(bars):
        ch=prog[b%len(prog)]
        notes=[max(36,min(84,root-12+n)) for n in ch]
        for n in notes:pad.append(Message('note_on',note=n,velocity=60,channel=1,time=0))
        pad.append(Message('note_off',note=notes[0],velocity=0,channel=1,time=4*tp))
        for n in notes[1:]:pad.append(Message('note_off',note=n,velocity=0,channel=1,time=0))
    bass=MidiTrack();mf.tracks.append(bass)
    bass.append(Message('program_change',program=33,channel=2,time=0))
    for b in range(bars):
        ch=prog[b%len(prog)]
        for off in [ch[0],ch[1] if len(ch)>1 else ch[0]+7,ch[0],ch[2] if len(ch)>2 else ch[0]+12]:
            n=max(28,min(55,root-24+off))
            bass.append(Message('note_on',note=n,velocity=85,channel=2,time=0))
            bass.append(Message('note_off',note=n,velocity=0,channel=2,time=tp))
    drums=MidiTrack();mf.tracks.append(drums)
    for b in range(bars):
        for _ in range(4):
            drums.append(Message('note_on',note=36,velocity=110,channel=9,time=0))
            drums.append(Message('note_off',note=36,velocity=0,channel=9,time=tp//2))
            drums.append(Message('note_off',note=0,velocity=0,channel=9,time=tp//2))
            drums.append(Message('note_on',note=38,velocity=100,channel=9,time=0))
            drums.append(Message('note_off',note=38,velocity=0,channel=9,time=tp//2))
            drums.append(Message('note_off',note=0,velocity=0,channel=9,time=tp//2))
        for _ in range(8):
            drums.append(Message('note_on',note=42,velocity=75,channel=9,time=0))
            drums.append(Message('note_off',note=42,velocity=0,channel=9,time=tp//4))
    mf.save(str(path))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='training_data/midi')
    ap.add_argument('--count',type=int,default=600)
    ap.add_argument('--bars',type=int,default=8)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for i in range(a.count):
        st=STYLES[i%len(STYLES)]
        make(out/(str(i).zfill(4)+'_'+st+'.mid'),seed=i,bars=a.bars,style=st)
    print('Gerados '+str(a.count)+' MIDI em '+str(out))