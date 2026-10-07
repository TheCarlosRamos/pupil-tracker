# Pupil Tracker

Rastreador local de direcao do olhar usando a webcam, OpenCV e MediaPipe Face Landmarker. O video e processado localmente; o script nao envia imagens da camera para um servidor.

## Requisitos

- Python 3.14
- Webcam
- Windows PowerShell (os comandos abaixo usam PowerShell)

As versoes em `requirements.txt` foram usadas para validar o projeto.

## Instalacao e execucao

```powershell
git clone https://github.com/TheCarlosRamos/pupil-tracker.git
cd pupil-tracker
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile '.\face_landmarker.task'
python .\pupil_tracker.py
```

Olhe para o centro da tela e pressione `c` para calibrar. Olhe em diferentes direcoes para conferir a classificacao. Pressione `q` para sair.

O arquivo `face_landmarker.task` e baixado separadamente e ignorado pelo Git. O script atual e uma aplicacao Python local; para hospedar no Vercel, seria necessario portar o processamento para o navegador.

## Ajustes

Os limiares de filtro, ganho e zona morta podem ser ajustados no inicio de `pupil_tracker.py`. Eles dependem da camera, iluminacao e posicionamento do rosto.