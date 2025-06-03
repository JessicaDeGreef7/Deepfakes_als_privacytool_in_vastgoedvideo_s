"""
Voor het deel van SimSwap heb ik gebruik gemaakt
van ChatGPT. Dit om foutmeldingen beter te 
begrijpen en op te lossen.
OpenAI (2025). ChatGPT (versie 24 mei 2025) [large language model]. https://chatopenai.com
"""

#################################################
#                   Imports                     #
#################################################

import sys
import argparse 
from types import SimpleNamespace
############## imports RetinaFace ###############
import tensorflow as tf

from retinaface import RetinaFace
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
sys.path.insert(0, '../RetinaFace/deepface')
from deepface import DeepFace
import pandas as pd
import math

############## imports Image2text  ##############
import torch
from lavis.models import load_model_and_preprocess
from collections import Counter

########### imports Stable diffusion ############
from diffusers import StableDiffusionPipeline

################ imports SimSwap ################
import os
import sys
# # caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, '../Deepfake/SimSwap')

# #import test_multi_video_swapsingle
from insightface_func.face_detect_crop_multi import Face_detect_crop
# # adding Folder_SimSwap to the system path
sys.path.insert(0, '../Deepfake/SimSwap')
from models.models import create_model
from options.test_options import TestOptions
import glob
from torchvision import transforms
import torch.nn.functional as F
from util.reverse2original import reverse2wholeimage
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from parsing_model.model import BiSeNet
from util.norm import SpecificNorm
from util.videoswap import video_swap


import nvidia_smi
import gc

#################################################
#              Globale variabele                #
#################################################

aantal_gezichten_gedetecteerd = 0
herkende_gezichten_embedding = pd.DataFrame()
encrypt_herkende_gezichten_embedding = []
embedding_herkende_gezichten = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multisepcific_dir = "../images/wissel/"
temp_results_dir='./temp_results'  
save_path = "/media/student/USB DISK/masterproef/resultaten/resultaat video" 
torch.cuda.memory_summary(device=None, abbreviated=False)
aantal_keer_bepaald_gezicht = []
crop_size = 224
threshold = 0.45

#################################################
#                   Code                        #
#################################################

################ code RetinaFace ################

def retinaface(video):
    print("start RetinaFace")
    # pass argument 0 for device index (hopefully laptop cam)
    vcap = cv2.VideoCapture(video)
    if not vcap.isOpened():
        print("Can't open camera!")
        exit()

    gedetecteerde_gezichten_frame = {}
    detecteer_gezichten_frames = {}
    frames_met_gezichten = []
    # loop until
    # - error reading frame from camera
    ret = True
    frame_index = 0

    frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vcap.get(cv2.CAP_PROP_FPS)
    stap = 1
    if fps > 10:
        stap = int(fps/10)

    for frame_index in range(0,frame_count, stap): 
        # read a frame
        ret, frame = vcap.read()

        # ret == True if frame read correctly
        if not ret:
            print("Can't read frame, exiting...")
            break

        gedetecteerde_gezichten_frame, detecteer_gezichten_frames = retinaface_frame(frame, frame_index, detecteer_gezichten_frames, gedetecteerde_gezichten_frame)

        if gedetecteerde_gezichten_frame != {}:
            frames_met_gezichten.append(frame_index)

    # everything done
    # - release capture
    # - release video writer
    # - destroy all windows
    print("Retinaface done")
    vcap.release()
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()
    return detecteer_gezichten_frames, frames_met_gezichten

def retinaface_frame(frame, frame_index, detecteer_gezichten_frames, vorig_frame_gedetecteerde_gezichten = {}):
  
  huidig_gedetecteerde_gezichten = RetinaFace.detect_faces(img_path = frame)
  lengte_huidig_gedetecteerde_gezichten = len(huidig_gedetecteerde_gezichten)
  global aantal_gezichten_gedetecteerd
  global aantal_keer_bepaald_gezicht
  global herkende_gezichten_embedding
  nieuwe_gezichten = []
  
  lengte_vorig_gedetecteerde_gezichten = len(vorig_frame_gedetecteerde_gezichten)
  
  if huidig_gedetecteerde_gezichten != {}:
    embedding_gezichten = []
    if aantal_gezichten_gedetecteerd == 0:
        
        aantal_gezichten_gedetecteerd += lengte_huidig_gedetecteerde_gezichten
        for plaats in range(aantal_gezichten_gedetecteerd):
            aantal_keer_bepaald_gezicht.append(1)
            nieuwe_gezichten.append(plaats)
            embedding_gezichten.append(plaats)
        
        detecteer_gezichten_frames[frame_index] = []
        i = 0
        for key in huidig_gedetecteerde_gezichten.keys():
            five_landmarks_list = huidig_gedetecteerde_gezichten[key]['landmarks']
            five_landmarks = [five_landmarks_list['right_eye'],five_landmarks_list['left_eye'], five_landmarks_list['nose'], five_landmarks_list['mouth_right'], five_landmarks_list['mouth_left']]
            detecteer_gezichten_frames[frame_index].append([i,huidig_gedetecteerde_gezichten[key]['facial_area'],five_landmarks])
            i += 1

        gezicht_extraheren(frame, nieuwe_gezichten, embedding_gezichten)  
    
        for plaats in range(aantal_gezichten_gedetecteerd):
            alpha = DeepFace.represent(img_path = multisepcific_dir + "SRC_" + str(plaats) + "_1.jpg", enforce_detection=False, detector_backend='skip')[0]["embedding"]
            embedding_herkende_gezichten.append(alpha)
    else:
        plaatsen, gezicht_om_toe_te_voegen, embedding_gezichten, bestond = zoekNieuwGezicht(huidig_gedetecteerde_gezichten, frame)
        keys = huidig_gedetecteerde_gezichten.keys()
        detecteer_gezichten_frames[frame_index] = []
        for i in range(lengte_huidig_gedetecteerde_gezichten):
             #zorg voor dat gezicht 

            if not bestond[i]:
                embedding_herkende_gezichten.append(gezicht_om_toe_te_voegen[i])
                aantal_keer_bepaald_gezicht.append(1)
            else:
                aantal_keer_bepaald_gezicht[embedding_gezichten[i]] = aantal_keer_bepaald_gezicht[embedding_gezichten[i]] + 1

            plaats = plaatsen[i]       

            key = list(keys)[plaats]
            five_landmarks_list = huidig_gedetecteerde_gezichten[key]['landmarks']
            five_landmarks = [five_landmarks_list['right_eye'],five_landmarks_list['left_eye'], five_landmarks_list['nose'], five_landmarks_list['mouth_right'], five_landmarks_list['mouth_left']]

            detecteer_gezichten_frames[frame_index].append([embedding_gezichten[i],huidig_gedetecteerde_gezichten[key]['facial_area'],five_landmarks])

        gezicht_extraheren(frame, plaatsen, embedding_gezichten) 

  return huidig_gedetecteerde_gezichten, detecteer_gezichten_frames

def zoekNieuwGezicht(huidig_gedetecteerde_gezichten, frame):
    nummer_gezicht = 0
  
    gevonden_gezichten = []
    bestonden = []
    plaatsen = []
    plaatsen_in_embedding = []  
    gezichten = RetinaFace.extract_faces(img_path = frame, align = True)
    while nummer_gezicht < len(huidig_gedetecteerde_gezichten):
        plaats_in_embedding = 0
        gevonden = False 
        te_herkennen_gezicht = DeepFace.represent(img_path = gezichten[nummer_gezicht], enforce_detection=False, detector_backend='skip')[0]["embedding"]

        lengte_embedding_herkende_gezichten = len(embedding_herkende_gezichten)

        for i in range(lengte_embedding_herkende_gezichten):
            calculated_similarity = cosine_similarity(embedding_herkende_gezichten[i], te_herkennen_gezicht)

            if calculated_similarity >= (1 - threshold):
                gevonden = True
                gevonden_gezichten.append(te_herkennen_gezicht)
                bestonden.append(True)
                break
            
            plaats_in_embedding += 1
            
        plaatsen_in_embedding.append(plaats_in_embedding)    
        plaatsen.append(nummer_gezicht)
        
        if gevonden == False:
            bestonden.append(False)
            gevonden_gezichten.append(te_herkennen_gezicht)
        nummer_gezicht += 1
    return plaatsen, gevonden_gezichten, plaatsen_in_embedding, bestonden
    
def cosine_similarity(v1, v2):
    noemer = 0
    kwadraard_som_x = 0
    kwadraard_som_y = 0
    for x, y in zip(v1, v2):
        noemer += x * y 
        kwadraard_som_x += x **2
        kwadraard_som_y += y **2
    teller = math.sqrt(kwadraard_som_x)*math.sqrt(kwadraard_som_y)
    return noemer / teller


    
def gezicht_extraheren(path, gezichten, embedding_gezichten):

    faces = RetinaFace.extract_faces(img_path = path, align = True)
    global aantal_gezichten_gedetecteerd
    global aantal_keer_bepaald_gezicht
    
    for g in range(len(gezichten)):
        gezicht = gezichten[g]
        data = Image.fromarray(faces[gezicht]) 
        
        data.save("../images/wissel/SRC_"+str(embedding_gezichten[g])+"_"+str(aantal_keer_bepaald_gezicht[embedding_gezichten[g]])+'.jpg') 
        g = RetinaFace.detect_faces(img_path="../images/wissel/SRC_"+str(embedding_gezichten[g])+"_"+str(aantal_keer_bepaald_gezicht[embedding_gezichten[g]])+'.jpg')
        
    return

################ code Image2Text ################

def image2text():
    antwoorden = []
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

    for persoon in range(aantal_gezichten_gedetecteerd):
        
        source_path = os.path.join(multisepcific_dir,'SRC_'+str(persoon)+'_*')
        gezichten_path = sorted(glob.glob(source_path))

        antwoorden_1_persoon = {
            "glasses": [], 
            "gender": [], 
            "age": [], 
            "eye color": [], 
            "hair color": [], 
            "hair style": [], 
            "ethnicity": [],
            "facial hair": [],
            "facial expression": []}
        for gezicht_path in gezichten_path:
            vragenImage2text(Image.open(gezicht_path).convert("RGB"), antwoorden_1_persoon, model, vis_processors, txt_processors)
        antwoord_1_persoon = {}
        for key_antwoorden in antwoorden_1_persoon.keys():
            antwoord_1_persoon[key_antwoorden] = Counter(antwoorden_1_persoon[key_antwoorden]).most_common(1)[0][0]
        antwoorden.append(toPrompt(antwoord_1_persoon))
    print(antwoorden)

    model.cpu()
    return antwoorden

def vragenImage2text(raw_image, antwoorden, model, vis_processors, txt_processors):
    
    # ask a glasses question.
    question = "Does this person has glasses?"
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)
    glasses = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")[0]
    antwoorden["glasses"].append(glasses)


    # ask the questions.
    gender = wichQuestion("gender", image, model, vis_processors, txt_processors)[0]
    antwoorden["gender"].append(gender)
    age = wichQuestion("age", image, model, vis_processors, txt_processors)[0]
    antwoorden["age"].append(age)
    eye_color = wichQuestion("color eyes", image, model, vis_processors, txt_processors)[0]
    antwoorden["eye color"].append(eye_color)
    hair_color = wichQuestion("color hair", image, model, vis_processors, txt_processors)[0]
    antwoorden["hair color"].append(hair_color)
    hair_style = wichQuestion("hair style", image, model, vis_processors, txt_processors)[0]
    antwoorden["hair style"].append(hair_style)
    ethnicity = whatQuestion("ethnicity", image, model, vis_processors, txt_processors)[0]
    antwoorden["ethnicity"].append(ethnicity)

    if gender != "female":
        question = "Does this person has facial hair?"
        question = txt_processors["eval"](question)
        facial_hair = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")[0]
    else:
        facial_hair = "nothing"
    antwoorden["facial hair"].append(facial_hair)

    facial_expression = whatQuestion("facial epression", image, model, vis_processors, txt_processors)[0]
    antwoorden["facial expression"].append(facial_expression)

    return antwoorden

def wichQuestion(what, image, model, vis_processors, txt_processors):  
    question = "Which " + what + " does this person has?"
    question = txt_processors["eval"](question)
    return model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")

def whatQuestion(what, image, model, vis_processors, txt_processors):
    question = "What " + what + " does this person has?"
    question = txt_processors["eval"](question)
    return model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")

def toPrompt(antwoorden):
    if antwoorden["glasses"] == "yes":
        prompt = "a " + antwoorden["gender"] + " of the age of " + antwoorden["age"] +" with ethnicity "+antwoorden["ethnicity"] +", with "+ antwoorden["hair style"] + ", " + antwoorden["hair color"] + " hair and " + antwoorden["eye color"] + " eyes with " + antwoorden["facial hair"] + " as facial hair and glasses and has "+antwoorden["facial expression"] + "high quality photography"
    else:
        prompt = "a " + antwoorden["gender"] + " of the age of " + antwoorden["age"] +" with ethnicity "+antwoorden["ethnicity"] +",with "+ antwoorden["hair style"] + ", " + antwoorden["hair color"] + " hair and " + antwoorden["eye color"] + " eyes with " + antwoorden["facial hair"] + " as facial hair and has  "+antwoorden["facial expression"] +" high quality photography"
    return prompt

############ code Stable diffussion #############
def stableDiffussion(antwoorden):
    model_path = "stablediffusionapi/realistic-vision-v13"

    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe.to("cuda")

    images = []
    for i in range(aantal_gezichten_gedetecteerd):
        prompt = antwoorden[i]

        image = pipe(prompt=prompt).images[0]

        #image.save(multisepcific_dir + "DST_" + str(i) + "_"+str(v)+".jpg")

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
    del pipe
    torch.cuda.empty_cache()
    
    return images

################# code SimSwap ##################
def simSwap(video, frames_met_gezichten, imgs, detecteerde_gezichten_frame):
    print("start SimSwap")
    # img = cv2 rgb 512, 512, 3 
    # omzetten naar tensor met extra dim batch # len =4 b,c,w,h
    transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    opt = TestOptions().parse()
    start_epoch, epoch_iter = 1, 0

    torch.nn.Module.dump_patches = True
    mode = 'None'
    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    vcap = cv2.VideoCapture(video)
    if not vcap.isOpened():
        #print("Can't open camera!")
        exit()

    frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vcap.get(cv2.CAP_PROP_FPS)

    stap = 1
    if fps > 10:
        stap = int(fps/10)
        fps = 10

    # while ret:
    for frame_index in range(0,frame_count,stap): 

        # read a frame
        ret, frame = vcap.read()

        # ret == True if frame read correctly
        if not ret:
            print("Can't read frame, exiting...")
            break

        # The specific person to be swapped(source)
        if frame_index in frames_met_gezichten:
            if aantal_gezichten_gedetecteerd == 1:
                single_face_video_swap(frame, frame_index, app, model, imgs, transformer_Arcface, detecteerde_gezichten_frame)
            else:
                target_id_norm_list, source_specific_id_nonorm_list = get_images_for_multi_videoswap(app, model, imgs, transformer_Arcface, detecteerde_gezichten_frame, frame_index, frame)

                multi_video_swap(frame, frame_index, target_id_norm_list,source_specific_id_nonorm_list, detecteerde_gezichten_frame, opt.id_thres, model, app)
        else:
            if not os.path.exists(temp_results_dir):
                os.mkdir(temp_results_dir)
            frame = frame.astype(np.uint8)
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

    model.to("cpu")

    # everything done
    # - release capture
    # - release video writer
    # - destroy all windows
    vcap.release()
    #output.release()
    cv2.destroyAllWindows()
    print("SimSwap Done")
    return fps

def single_face_video_swap(frame, frame_index, app, model, image, transformer_Arcface, detecteerde_gezichten_frame):
    
    transformer = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    with torch.no_grad():
        img_a_align_crop, _ = app.get(image[0],crop_size)

        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
        img_a_align_crop_pil.save("debug_identity_input.jpg")
        img_a = transformer_Arcface(img_a_align_crop_pil)

        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        # img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))      
        latend_id = model.netArc(img_id_downsample)       
        latend_id = F.normalize(latend_id, p=2, dim=1)
       
        single_video_swap(frame, frame_index, latend_id, model, app, detecteerde_gezichten_frame)
 
import fractions       
        
def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0
      
def single_video_swap(frame, frame_index, id_vetor, swap_model, detect_model, detecteerde_gezichten_frame, temp_results_dir = './temp_results'):
    
    spNorm = SpecificNorm()
    
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    # while ret:
    detect_results = detect_model.get(frame,crop_size)

    if detect_results is not None:

        if not os.path.exists(temp_results_dir):
                os.mkdir(temp_results_dir)
        frame_align_crop_list = detect_results[0]
        for i, crop in enumerate(frame_align_crop_list):
            save_path = f"crop_{i}.jpg"
            cv2.imwrite(save_path, crop)

        frame_mat_list = detect_results[1]
        for i, mat in enumerate(frame_mat_list):
            save_path = f"mat_{i}.txt"
            np.savetxt(save_path, mat, fmt="%.6f")

        for i, crop in enumerate(frame_align_crop_list):
            cv2.imwrite(f"debug_crop_{i}.jpg", crop)

        img = np.array(frame, dtype=np.float)
        for frame_align_crop in range(len(frame_align_crop_list)):

            # BGR TO RGB
            # frame_align_crop_RGB = frame_align_crop[...,::-1]
            frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop_list[frame_align_crop],cv2.COLOR_BGR2RGB))[None,...].cuda()

            swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]

            img = reverse2wholeimage(img, frame_align_crop_tenor,swap_result, frame_mat_list[frame_align_crop], crop_size, frame, pasring_model =net, norm = spNorm)

        final_img = img.astype(np.uint8)

        cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), final_img)

        torch.cuda.empty_cache()

    else:
        if not os.path.exists(temp_results_dir):
            os.mkdir(temp_results_dir)
        frame = frame.astype(np.uint8)
        cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

def get_images_for_multi_videoswap(app, model, images, transformer_Arcface, detecteerde_gezichten_frame, frame_index,frame):
    
    source_specific_id_nonorm_list = []
    target_id_norm_list = []
    with torch.no_grad():

        frame_info = detecteerde_gezichten_frame[frame_index]
        aantal_gezichten_in_frame = len(frame_info)

        for i in range(aantal_gezichten_in_frame):

            specific_person_align_crop, _ = app.get(frame,crop_size, detecteerde_gezichten_frame[frame_index][i])
            specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB)) 
            specific_person = transformer_Arcface(specific_person_align_crop_pil)
            specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
            # convert numpy to tensor
            specific_person = specific_person.cuda()
            #create latent id
            specific_person_downsample = F.interpolate(specific_person, size=(112,112))
            specific_person_id_nonorm = model.netArc(specific_person_downsample)
            source_specific_id_nonorm_list.append(specific_person_id_nonorm.clone())

            g = detecteerde_gezichten_frame[frame_index][i][0]

            img_a_align_crop, _ = app.get(images[g],crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            # convert numpy to tensor
            img_id = img_id.cuda()
            #create latent id
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            target_id_norm_list.append(latend_id.clone())

        assert len(target_id_norm_list) == len(source_specific_id_nonorm_list), "The number of images in source and target directory must be same !!!"
    return target_id_norm_list, source_specific_id_nonorm_list

def multi_video_swap(frame, frame_index, target_id_norm_list,source_specific_id_nonorm_list, detecteerde_gezichten_frame, id_thres, swap_model, detect_model):

    spNorm = SpecificNorm()

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    # while ret:
    aantal_mensen_in_frame = len(detecteerde_gezichten_frame[frame_index])
    img = np.array(frame, dtype=np.float)
    for i in range(aantal_mensen_in_frame):

        detect_results = detect_model.get(frame,crop_size,detecteerde_gezichten_frame[frame_index][i])

        if detect_results is not None:
            if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]


            #id_compare_values = [] 
            #frame_align_crop_tenor_list = []
            for f in range(len(frame_align_crop_list)):
                frame_align_crop = frame_align_crop_list[f]
                # BGR TO RGB
                # frame_align_crop_RGB = frame_align_crop[...,::-1]

                frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                swap_result = swap_model(None, frame_align_crop_tenor, target_id_norm_list[f], None, True)[0]

                img = reverse2wholeimage(img, frame_align_crop_tenor,swap_result, frame_mat_list[f], crop_size, frame, \
                pasring_model = net, norm = spNorm)

        else:
            if not os.path.exists(temp_results_dir):
                os.mkdir(temp_results_dir)
            frame = frame.astype(np.uint8)
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)      
    final_img = img.astype(np.uint8)

    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), final_img)

    torch.cuda.empty_cache()

    return

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)



if __name__ == "__main__":
    opt = TestOptions().parse()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", dest='video_path', type=str, help='Add video_path')
    parser.add_argument("--output_path", dest='output_path', type=str, help='Add output_path')
    args = parser.parse_args()
    
    output_path = args.output_path
    video_path = args.video_path

    aantal_gezichten_gedetecteerd = 0
    orginele_video = os.path.join(video_path)
    
    
    os.system('rm -rf ./temp_results/*')
    os.system('rm -rf ../images/wissel/SRC_*')

    detecteerde_gezichten_frame, retinaface_frames_met_gezichten = retinaface(orginele_video)
    
    antwoorden = image2text()
    
    img = stableDiffussion(antwoorden)
    
    fps = simSwap(orginele_video, retinaface_frames_met_gezichten, img, detecteerde_gezichten_frame)
    
    
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    #fps = 10
    clips = ImageSequenceClip(image_filenames,fps = fps)
    clips.write_videofile(save_path, codec="libx264")
    os.system('rm -rf ./temp_results/*')
    os.system('rm -rf ../images/wissel/SRC_*')
        