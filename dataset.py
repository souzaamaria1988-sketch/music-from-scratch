#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from mido import MidiFile
DUR_BINS=[60,120,240,360,480,720,960,1440,1920]
VEL_BINS=[0,40,60,80,100,120]
def dur_bin(t):
    for i,b in enumerate(DUR_BINS):
        if t<=b:return i
    return len(DUR_BINS)-1
def vel_bin(v):
    for i,b in enumerate(VEL_BINS):
        if v<=b:return i
    return len(VEL_BINS)-1
VOCAB=149
def midi_to_tokens(path):
    mf=MidiFile(str(path));toks=[1];active={}
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
                    toks.append(131+dur_bin(d))
                    toks.append(140+vel_bin(v))
                    toks.append(146+min(2,msg.channel))
    toks.append(2);return toks
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--midi',default='training_data/midi')
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
    print('OK '+str(n)+' seq -> '+a.out)