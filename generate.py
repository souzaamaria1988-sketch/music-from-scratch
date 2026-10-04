#!/usr/bin/env python3
"""Gera MIDI com modelo proprio, renderiza WAV com soundfont."""
import argparse,subprocess,random
from pathlib import Path
from datetime import datetime
import torch
from mido import MidiFile,MidiTrack,Message,MetaMessage
from model import MusicTransformer,VOCAB

DUR_VALS=[60,120,240,360,480,720,960,1440,1920]
VEL_VALS=[30,50,70,90,110,127]

def tokens_to_midi(toks,path,bpm=120):
    mf=MidiFile(ticks_per_beat=480);tp=480
    cond=MidiTrack();mf.tracks.append(cond)
    cond.append(MetaMessage("set_tempo",tempo=int(60_000_000/bpm),time=0))
    trk=MidiTrack();mf.tracks.append(trk)
    trk.append(Message("program_change",program=0,channel=0,time=0))
    i=0
    while i<len(toks):
        t=toks[i]
        if 3<=t<=130 and i+3<len(toks):
            pitch=t-3
            d_bin=toks[i+1]-131
            v_bin=toks[i+2]-140
            ch_flag=toks[i+3]-146 if 146<=toks[i+3]<=148 else 0
            d=DUR_VALS[d_bin] if 0<=d_bin<len(DUR_VALS) else 480
            v=VEL_VALS[v_bin] if 0<=v_bin<len(VEL_VALS) else 90
            ch=9 if ch_flag==2 else (1 if ch_flag==1 else 0)
            trk.append(Message("note_on",note=pitch,velocity=v,channel=ch,time=0))
            trk.append(Message("note_off",note=pitch,velocity=0,channel=ch,time=d))
            i+=4
        else:
            i+=1
    mf.save(str(path))

def load_model(ckpt):
    d=torch.load(ckpt,map_location="cpu",weights_only=False)
    m=MusicTransformer(vocab=d["vocab"],d_model=d["d_model"],n_head=d["n_head"],n_layer=d["n_layer"],max_len=d["max_len"])
    m.load_state_dict(d["model"]);m.eval()
    return m,d["vocab"]

def render_wav(midi,sf2,wav):
    try:
        subprocess.run(["fluidsynth","-F",str(wav),"-T","wav","-r","32000","-ni",str(sf2),str(midi)],check=True,capture_output=True,timeout=180)
        return True
    except Exception as e:
        print("fluidsynth erro:",e);return False

def pick_sf(sf_dir):
    p=Path(sf_dir)
    if not p.exists():return None
    fs=list(p.glob("*.sf2"))+list(p.glob("*.sf3"))
    return fs[0] if fs else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",default="checkpoints/best.pt")
    ap.add_argument("--prompt-style",default="random",choices=["random","calm","tense","epic","chaotic"])
    ap.add_argument("--length",type=int,default=512)
    ap.add_argument("--temperature",type=float,default=1.0)
    ap.add_argument("--top-k",type=int,default=40)
    ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--bpm",type=int,default=120)
    ap.add_argument("--out",default="outputs")
    ap.add_argument("--soundfonts",default="soundfonts")
    a=ap.parse_args()

    if a.seed is not None:torch.manual_seed(a.seed);random.seed(a.seed)

    m,_=load_model(a.ckpt)
    style_temps={"calm":0.7,"tense":0.9,"epic":1.05,"chaotic":1.3,"random":a.temperature}
    temp=style_temps.get(a.prompt_style,a.temperature)
    topk={"calm":20,"tense":30,"epic":40,"chaotic":80,"random":a.top_k}.get(a.prompt_style,a.top_k)

    start=torch.tensor([[1]],dtype=torch.long)
    print(f"Gerando {a.length} tokens (style={a.prompt_style} T={temp} k={topk})")
    out=m.generate(start,max_new_tokens=a.length,temperature=temp,top_k=topk)
    toks=out[0].tolist()
    print(f"Gerado: {len(toks)} tokens")

    out_dir=Path(a.out);out_dir.mkdir(parents=True,exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    base=f"{ts}_{a.prompt_style}"

    midi_path=out_dir/f"{base}.mid"
    tokens_to_midi(toks,midi_path,bpm=a.bpm)
    print(f"MIDI: {midi_path}")

    sf=pick_sf(a.soundfonts)
    if sf:
        wav_path=out_dir/f"{base}.wav"
        if render_wav(midi_path,sf,wav_path):
            print(f"WAV: {wav_path} (sf={sf.name})")
            try:
                subprocess.run(["ffmpeg","-y","-i",str(wav_path),"-codec:a","libmp3lame","-qscale:a","2",str(wav_path.with_suffix(".mp3"))],check=True,capture_output=True)
                print(f"MP3: {wav_path.with_suffix('.mp3')}")
            except Exception:pass
    else:
        print("Sem soundfont, so MIDI. Coloque .sf2 em soundfonts/")

if __name__=="__main__":main()
