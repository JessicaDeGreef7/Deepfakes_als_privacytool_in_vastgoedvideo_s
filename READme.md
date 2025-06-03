# Deepfakes als privacytool in vastgoedvideo's
Werking:
1. Detecteren en extraheren gezichten met RetinaFace
2. Afbeeldingen van gezichten omzetten prompts met de gezichtskenmerken (dankzij LAVIS)
3. Met prompts een gezicht genereren (Stable Diffussion)
4. Simswap (deepfake)

Instaleren:
1. git clone https://github.com/JessicaDeGreef7/Deepfakes_als_privacytool_in_vastgoedvideo_s.git
2. cd open masterproef
3. pip install -r requirements.txt

Runnen:
1. cd pipeline
2. python verwissel.py --video_path path_to_video --save_path path_to_save_resultaat
   
vb: python verwissel.py --video_path ../images/persoon_1.mp4 --save_path ./resultaat.mp4
