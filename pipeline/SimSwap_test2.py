import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from insightface_func.face_detect_crop_single import Face_detect_crop
import sys
sys.path.insert(0, '../Deepfake/SimSwap')
from models.models import create_model
from options.test_options import TestOptions
#from util.videoswap import video_swap
import os

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])


if __name__ == '__main__':
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()


    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)
    with torch.no_grad():
        pic_a = opt.pic_a_path
        # img_a = Image.open(pic_a).convert('RGB')
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # pic_b = opt.pic_b_path
        # img_b_whole = cv2.imread(pic_b)
        # img_b_align_crop, b_mat = app.get(img_b_whole,crop_size)
        # img_b_align_crop_pil = Image.fromarray(cv2.cvtColor(img_b_align_crop,cv2.COLOR_BGR2RGB)) 
        # img_b = transformer(img_b_align_crop_pil)
        # img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        # img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        video_swap(opt.video_path, latend_id, model, app, opt.output_path,temp_results_dir=opt.temp_path,\
            use_mask=opt.use_mask,crop_size=crop_size)

import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
#from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    # while ret:
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            detect_results = detect_model.get(frame,crop_size)

            if detect_results is not None:
                # print(frame_index)
                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                frame_align_crop_list = detect_results[0]
                frame_mat_list = detect_results[1]
                swap_result_list = []
                frame_align_crop_tenor_list = []
                for frame_align_crop in frame_align_crop_list:

                    # BGR TO RGB
                    # frame_align_crop_RGB = frame_align_crop[...,::-1]

                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                    swap_result_list.append(swap_result)
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                    

                reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                    os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
        else:
            break

    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac')

import cv2
import numpy as np
# import  time
import torch
from torch.nn import functional as F
import torch.nn as nn


def encode_segmentation_rgb(segmentation, no_neck=True):
    parse = segmentation

    face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    # hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    # hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse==mouth_id)
    mouth_map[valid_index] = 255
    # valid_index = np.where(parse==hair_id)
    # hair_map[valid_index] = 255
    #return np.stack([face_map, mouth_map,hair_map], axis=2)
    return np.stack([face_map, mouth_map], axis=2)


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def postprocess(swapped_face, target, target_mask,smooth_mask):
    # target_mask = cv2.resize(target_mask, (self.size,  self.size))

    mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
    face_mask_tensor = mask_tensor[0] + mask_tensor[1]
    
    soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
    soft_face_mask_tensor.squeeze_()

    soft_face_mask = soft_face_mask_tensor.cpu().numpy()
    soft_face_mask = soft_face_mask[:, :, np.newaxis]

    result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
    result = result[:,:,::-1]# .astype(np.uint8)
    return result

def reverse2wholeimage(b_align_crop_tenor_list,swaped_imgs, mats, crop_size, oriimg, logoclass, save_path = '', \
                    no_simswaplogo = False,pasring_model =None,norm = None, use_mask = False):

    target_image_list = []
    img_mask_list = []
    if use_mask:
        smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
    else:
        pass

    # print(len(swaped_imgs))
    # print(mats)
    # print(len(b_align_crop_tenor_list))
    for swaped_img, mat ,source_img in zip(swaped_imgs, mats,b_align_crop_tenor_list):
        swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
        img_white = np.full((crop_size,crop_size), 255, dtype=float)

        # inverse the Affine transformation matrix
        mat_rev = np.zeros([2,3])
        div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
        mat_rev[0][0] = mat[1][1]/div1
        mat_rev[0][1] = -mat[0][1]/div1
        mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
        div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
        mat_rev[1][0] = mat[1][0]/div2
        mat_rev[1][1] = -mat[0][0]/div2
        mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

        orisize = (oriimg.shape[1], oriimg.shape[0])
        if use_mask:
            source_img_norm = norm(source_img)
            source_img_512  = F.interpolate(source_img_norm,size=(512,512))
            out = pasring_model(source_img_512)[0]
            parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            tgt_mask = encode_segmentation_rgb(vis_parsing_anno)
            if tgt_mask.sum() >= 5000:
                # face_mask_tensor = tgt_mask[...,0] + tgt_mask[...,1]
                target_mask = cv2.resize(tgt_mask, (crop_size,  crop_size))
                # print(source_img)
                target_image_parsing = postprocess(swaped_img, source_img[0].cpu().detach().numpy().transpose((1, 2, 0)), target_mask,smooth_mask)
                

                target_image = cv2.warpAffine(target_image_parsing, mat_rev, orisize)
                # target_image_parsing = cv2.warpAffine(swaped_img, mat_rev, orisize)
            else:
                target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)[..., ::-1]
        else:
            target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
        # source_image   = cv2.warpAffine(source_img, mat_rev, orisize)

        img_white = cv2.warpAffine(img_white, mat_rev, orisize)


        img_white[img_white>20] =255

        img_mask = img_white

        # if use_mask:
        #     kernel = np.ones((40,40),np.uint8)
        #     img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        # else:
        kernel = np.ones((40,40),np.uint8)
        img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        kernel_size = (20, 20)
        blur_size = tuple(2*i+1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

        # kernel = np.ones((10,10),np.uint8)
        # img_mask = cv2.erode(img_mask,kernel,iterations = 1)



        img_mask /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])

        # pasing mask

        # target_image_parsing = postprocess(target_image, source_image, tgt_mask)

        if use_mask:
            target_image = np.array(target_image, dtype=np.float) * 255
        else:
            target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255


        img_mask_list.append(img_mask)
        target_image_list.append(target_image)
        

    # target_image /= 255
    # target_image = 0
    img = np.array(oriimg, dtype=np.float)
    for img_mask, target_image in zip(img_mask_list, target_image_list):
        img = img_mask * target_image + (1-img_mask) * img
        
    final_img = img.astype(np.uint8)
    if not no_simswaplogo:
        final_img = logoclass.apply_frames(final_img)
    cv2.imwrite(save_path, final_img)