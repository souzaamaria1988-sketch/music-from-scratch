#!/usr/bin/env python3
import argparse,subprocess
from pathlib import Path
import soundfile as sf
import numpy as np

def pick_sf(sf_dir):
    p=Path(sf_dir)
    if not p.exists():return None
    fs=sorted(p.glob('*.sf2'))+sorted(p.glob('*.sf3'))
    return fs[0] if fs else None

def fallback_render(midi_path,wav_path,sr=32000):
    from mido import MidiFile
    mf=MidiFile(str(midi_path))
    tempo=500000
    for tr in mf.tracks:
        for msg in tr:
            if msg.type=='set_tempo':tempo=msg.tempo;break
    total=0
    for tr in mf.tracks:
        t=0
        for m in tr:t+=m.time
        total=max(total,t)
    sec=(tempo/1e6)*(total/mf.ticks_per_beat)
    n=int(max(1,sec)*sr)
    buf=np.zeros(n,dtype=np.float32)
    for tr in mf.tracks:
        ct=0
        active={}
        for msg in tr:
            ct+=msg.time
            samp=int((tempo/1e6)*(ct/mf.ticks_per_beat)*sr)
            if msg.type=='note_on' and msg.velocity>0:
                active[(msg.note,msg.channel)]=(samp,msg.velocity)
            elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                k=(msg.note,msg.channel)
                if k in active:
                    st,v=active.pop(k)
                    en=min(samp,n)
                    if en>st:
                        f=440.0*(2**((msg.note-69)/12))
                        t=np.arange(en-st)/sr
                        sig=np.sin(2*np.pi*f*t)*0.15*(v/127)
                        buf[st:en]+=sig.astype(np.float32)
    pk=np.max(np.abs(buf))
    if pk>0:buf=buf/pk*0.9
    sf.write(str(wav_path),buf,sr)

def render_one(midi,wav,sf2):
    if sf2:
        try:
            subprocess.run(['fluidsynth','-F',str(wav),'-T','wav','-r','32000','-ni',str(sf2),str(midi)],check=True,capture_output=True,timeout=120)
            return True
        except Exception as e:
            print('fluidsynth erro:',e)
    fallback_render(midi,wav)
    return False

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--midi',default='training_data/midi')
    ap.add_argument('--out',default='training_data/audio')
    ap.add_argument('--soundfonts',default='soundfonts')
    ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args()
    midi_dir=Path(a.midi);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sf2=pick_sf(a.soundfonts)
    print('SoundFont:',sf2.name if sf2 else 'nenhum (fallback)')
    files=sorted(midi_dir.glob('*.mid'))+sorted(midi_dir.glob('*.midi'))
    if a.limit:files=files[:a.limit]
    ok=0
    for i,p in enumerate(files):
        w=out/(p.stem+'.wav')
        if w.exists():continue
        render_one(p,w,sf2)
        ok+=1
        if i%50==0:print('  '+str(i)+'/'+str(len(files)))
    print('Render: '+str(ok)+' WAV em '+str(out))