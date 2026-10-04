#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from mido import MidiFile
DUR=[60,120,180,240,360,480,720,960,1440,1920,2880,3840]
VEL=[0,30,50,70,90,110,120]
def db(t):
    for i,b in enumerate(DUR):
        if t<=b:return i
    return len(DUR)-1
def vb(v):
    for i,b in enumerate(VEL):
        if v<=b:return i
    return len(VEL)-1
VOCAB=159
def midi_to_tokens(p):
    mf=MidiFile(str(p));toks=[1];active={}
    for track in mf.tracks:
        t=0
        for msg in track:
            t+=msg.time
            if msg.type=='note_on' and msg.velocity>0:
                active[(msg.note,msg.channel)]=(t,msg.velocity)
            elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                k=(msg.note,msg.channel)
                if k in active:
                    st,v=active.pop(k);d=t-st
                    toks.append(3+msg.note)
                    toks.append(131+db(d))
                    toks.append(143+vb(v))
                    ch=msg.channel
                    flag=2 if ch==9 else (1 if ch==1 else (0 if ch==0 else 3))
                    toks.append(150+flag)
    toks.append(2);return toks
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--midi',default='training_data/augmented')
    ap.add_argument('--out',default='tokens.jsonl')
    a=ap.parse_args()
    files=sorted(Path(a.midi).glob('*.mid'))+sorted(Path(a.midi).glob('*.midi'))
    n=0
    with open(a.out,'w') as f:
        for p in files:
            try:
                t=midi_to_tokens(p)
                if len(t)<20:continue
                f.write(json.dumps(t)+'\n');n+=1
            except Exception as e:print('skip',p,e)
    print('OK '+str(n)+' seq vocab='+str(VOCAB))