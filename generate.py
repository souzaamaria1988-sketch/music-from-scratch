#!/usr/bin/env python3
import argparse,subprocess,random
from pathlib import Path
from datetime import datetime
import torch
from mido import MidiFile,MidiTrack,Message,MetaMessage
from model import MusicTransformer,VOCAB
DUR=[60,120,180,240,360,480,720,960,1440,1920,2880,3840]
VEL=[30,50,70,90,110,127]
def tokens_to_midi(toks,path,bpm=120):
    mf=MidiFile(ticks_per_beat=480);tp=480
    c=MidiTrack();mf.tracks.append(c);c.append(MetaMessage('set_tempo',tempo=int(60000000/bpm),time=0))
    tracks={}
    for ch in [0,1,2,9]:
        t=MidiTrack();mf.tracks.append(t);tracks[ch]=t
    tracks[0].append(Message('program_change',program=0,channel=0,time=0))
    tracks[1].append(Message('program_change',program=48,channel=1,time=0))
    tracks[2].append(Message('program_change',program=33,channel=2,time=0))
    i=0
    while i<len(toks):
        t=toks[i]
        if 3<=t<=130 and i+3<len(toks):
            pitch=t-3;d_bin=toks[i+1]-131;v_bin=toks[i+2]-143;ch_flag=toks[i+3]-150 if 150<=toks[i+3]<=153 else 0
            d=DUR[d_bin] if 0<=d_bin<len(DUR) else 480
            v=VEL[v_bin] if 0<=v_bin<len(VEL) else 90
            ch=9 if ch_flag==2 else (1 if ch_flag==1 else (2 if ch_flag==3 else 0))
            trk=tracks[ch]
            trk.append(Message('note_on',note=pitch,velocity=v,channel=ch,time=0))
            trk.append(Message('note_off',note=pitch,velocity=0,channel=ch,time=d))
            i+=4
        else:i+=1
    mf.save(str(path))
def load_model(ckpt):
    d=torch.load(ckpt,map_location='cpu',weights_only=False)
    m=MusicTransformer(vocab=d['vocab'],d_model=d['d_model'],n_head=d['n_head'],n_layer=d['n_layer'],d_ff=d.get('d_ff',1536),max_len=d['max_len'])
    m.load_state_dict(d['model']);m.eval();return m
def pick_sf(d):
    p=Path(d)
    if not p.exists():return None
    fs=sorted(p.glob('*.sf2'))+sorted(p.glob('*.sf3'))
    return fs[0] if fs else None
def render_wav(midi,sf2,wav):
    try:
        subprocess.run(['fluidsynth','-F',str(wav),'-T','wav','-r','32000','-ni',str(sf2),str(midi)],check=True,capture_output=True,timeout=300)
        return True
    except Exception as e:print('fs:',e);return False
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',default='models/best.pt')
    ap.add_argument('--style',default='random',choices=['random','calm','tense','epic','chaotic','boss_fight'])
    ap.add_argument('--length',type=int,default=512)
    ap.add_argument('--temperature',type=float,default=1.0)
    ap.add_argument('--top-k',type=int,default=40)
    ap.add_argument('--seed',type=int,default=None)
    ap.add_argument('--bpm',type=int,default=120)
    ap.add_argument('--out',default='outputs')
    ap.add_argument('--soundfonts',default='soundfonts')
    a=ap.parse_args()
    if a.seed is not None:torch.manual_seed(a.seed);random.seed(a.seed)
    m=load_model(a.ckpt)
    styles={'calm':(0.7,20,1.05),'tense':(0.9,30,1.1),'epic':(1.05,40,1.15),'chaotic':(1.3,80,1.2),'boss_fight':(1.15,60,1.25),'random':(a.temperature,a.top_k,1.15)}
    temp,topk,rep=styles.get(a.style,(a.temperature,a.top_k,1.15))
    print('style='+a.style+' T='+str(temp)+' k='+str(topk)+' len='+str(a.length))
    start=torch.tensor([[1]],dtype=torch.long)
    out=m.generate(start,max_new_tokens=a.length,temperature=temp,top_k=topk,top_p=0.95,repetition_penalty=rep)
    toks=out[0].tolist()
    o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    base=ts+'_'+a.style
    midi_path=o/(base+'.mid')
    tokens_to_midi(toks,midi_path,bpm=a.bpm)
    print('MIDI: '+str(midi_path))
    sf2=pick_sf(a.soundfonts)
    if sf2:
        wav=o/(base+'.wav')
        print('SF: '+sf2.name)
        if render_wav(midi_path,sf2,wav):
            print('WAV: '+str(wav))
            try:
                subprocess.run(['ffmpeg','-y','-i',str(wav),'-codec:a','libmp3lame','-qscale:a','2',str(wav.with_suffix('.mp3'))],check=True,capture_output=True)
                print('MP3: '+str(wav.with_suffix('.mp3')))
            except Exception:pass
    else:print('Sem soundfont (adicione .sf2 em soundfonts/)')
if __name__=='__main__':main()