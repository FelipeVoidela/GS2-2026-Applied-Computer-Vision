"""
Experimento da quarta rodada que NAO entrou no sistema entregue (ver resultados/experimentos/quarta_rodada.json).
Script de bancada, guardado para conferencia. Uso: python yolo_unificado_treinar.py <pasta do repositorio> exp-loc2c
"""
import sys, shutil
from pathlib import Path
from ultralytics import YOLO
raiz=Path(sys.argv[1]); DS=raiz/"dados"/"dataset"; L=raiz/"runs"/"_loc2c"
if L.exists(): shutil.rmtree(L)
for parte in ("train","val"):
    (L/"images"/parte).mkdir(parents=True); (L/"labels"/parte).mkdir(parents=True)
    for im in (DS/"images"/parte).glob("*.jpg"):
        shutil.copy2(im,L/"images"/parte/im.name)
        ls=[l.split() for l in (DS/"labels"/parte/(im.stem+".txt")).read_text().splitlines()]
        (L/"labels"/parte/(im.stem+".txt")).write_text("".join(f"{0 if int(l[0])<3 else 1} {' '.join(l[1:])}\n" for l in ls))
(L/"data.yaml").write_text(f"path: {L.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: boca_de_lobo\n  1: lixo_sarjeta\n")
YOLO(str(raiz/"yolov8n.pt")).train(data=str(L/"data.yaml"),epochs=60,imgsz=640,batch=16,device="cpu",workers=4,patience=25,seed=2026,deterministic=True,
    project=str(raiz/"runs"),name=sys.argv[2],exist_ok=True,degrees=15.0,fliplr=0.5,flipud=0.2,hsv_h=0.015,hsv_s=0.6,hsv_v=0.45,scale=0.5,translate=0.1,mosaic=1.0,close_mosaic=10,plots=False)
