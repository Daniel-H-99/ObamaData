from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from PIL import Image
import cv2
from collections import OrderedDict

import pickle
from tqdm import tqdm

def fit_lip_to_face(kp, normed_lip, scale_coeff, tilt, mean):
    ref_x = np.linalg.norm(kp[45] - kp[36], ord=2, axis=0)
    ref_y = np.linalg.norm(kp[30] - kp[27], ord=2, axis=0)
    scale = np.array([ref_x, ref_y]) * scale_coeff
    kp_dn = normed_lip * scale[np.newaxis]
    x, y = kp_dn[:, 0], kp_dn[:, 1]
    c, s = np.cos(tilt), np.sin(tilt)
    x_dash, y_dash = x*c + y*s, -x*s + y*c
    kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1, 1))))
    lip = kp_tilt + mean
    lip = lip.astype('int')
    new_kp = np.concatenate([kp[:48], lip], axis=0)
    
    return new_kp

def extract_scale_coeff(kp):
    ref_x = np.linalg.norm(kp[45] - kp[36], ord=2, axis=0)
    ref_y = np.linalg.norm(kp[30] - kp[27], ord=2, axis=0)
    target_x = np.linalg.norm(kp[54] -kp[48], ord=2, axis=0)
    target_y = np.linalg.norm(kp[57] - kp[51], ord=2, axis=0)
    scale = np.array([target_x / ref_x, target_y / ref_y])
    return scale

def normalize_lip(lip):
    ref_x = np.linalg.norm(lip[6] - lip[0], ord=1, axis=0)
    ref_y = np.linalg.norm(lip[9] - lip[3], ord=1, axis=0)
    return lip / np.array([ref_x, ref_y])[np.newaxis]
    
def save_all_tensors(opt, real_A, fake_B, fake_B_first, fake_B_raw, real_B, flow_ref, conf_ref, flow, weight, modelD):
    if opt.label_nc != 0:
        input_image = tensor2label(real_A, opt.label_nc)
    elif opt.dataset_mode == 'pose':
        input_image = tensor2im(real_A)
        if real_A.size()[2] == 6:
            input_image2 = tensor2im(real_A[0, -1, 3:])
            input_image[input_image2 != 0] = input_image2[input_image2 != 0]
    else:
        c = 3 if opt.input_nc >= 3 else 1
        input_image = tensor2im(real_A[0, -1, :c], normalize=False)
    if opt.use_instance:
        edges = tensor2im(real_A[0, -1, -1:], normalize=False)
        input_image += edges[:,:,np.newaxis]
    
    if opt.add_face_disc:
        ys, ye, xs, xe = modelD.module.get_face_region(real_A[0, -1:])
        if ys is not None:
            input_image[ys, xs:xe, :] = input_image[ye, xs:xe, :] = input_image[ys:ye, xs, :] = input_image[ys:ye, xe, :] = 255 

    visual_list = [('input_image', input_image),
                   ('fake_image', tensor2im(fake_B)),
                   ('fake_first_image', tensor2im(fake_B_first)),
                   ('fake_raw_image', tensor2im(fake_B_raw)),
                   ('real_image', tensor2im(real_B)),                                                          
                   ('flow_ref', tensor2flow(flow_ref)),
                   ('conf_ref', tensor2im(conf_ref, normalize=False))]
    if flow is not None:
        visual_list += [('flow', tensor2flow(flow)),
                        ('weight', tensor2im(weight, normalize=False))]
    visuals = OrderedDict(visual_list)
    return visuals

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor[0, -1]
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor[:3]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0        
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def tensor2label(output, n_label, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float()    
    if output.size()[0] > 1:
        output = output.max(0, keepdim=True)[1]
    #print(output.size())
    output = Colorize(n_label)(output)
    output = np.transpose(output.numpy(), (1, 2, 0))
    #img = Image.fromarray(output, "RGB")
    return output.astype(imtype)

def tensor2flow(output, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float().numpy()
    output = np.transpose(output, (1, 2, 0))
    #mag = np.max(np.sqrt(output[:,:,0]**2 + output[:,:,1]**2)) 
    #print(mag)
    hsv = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(output[..., 0], output[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def add_dummy_to_tensor(tensors, add_size=0):
    if add_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        dummy = torch.zeros_like(tensors)[:add_size]
        tensors = torch.cat([dummy, tensors])
    return tensors

def remove_dummy_from_tensor(tensors, remove_size=0):
    if remove_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # Cityscapes train
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    elif N == 20: # Cityscapes eval
        cmap = np.array([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30), 
                         (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), 
                         (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,  0)], 
                         dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0            
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0], cmap[i, 1], cmap[i, 2] = r, g, b             
    return cmap

def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7-j))*((i & (1 << (3*j))) >> (3*j))
            g = g + (1 << (7-j))*((i & (1 << (3*j+1))) >> (3*j+1))
            b = b + (1 << (7-j))*((i & (1 << (3*j+2))) >> (3*j+2))

        cmap[i, :] = np.array([r, g, b])

    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
        
    
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_file_list(data_dir, suffix=""):
    file_list = []

    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if suffix in filename:
                file_list.append(os.path.join(dirpath, filename))

    file_list = sorted(file_list)

    return file_list


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()

    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size))
        else:
            # raise KeyError('unexpected key "{}" in state_dict'.format(name))
            pass


def load_coef(data_dir, load_num=float('inf')):
    coef_list = []
    count = 0

    for filename in tqdm(get_file_list(data_dir)):
        coef = torch.load(filename)
        coef_list.append(coef)
        count += 1
        if count >= load_num:
            break

    return coef_list


def landmark_detection(image_list, save_path):
    fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    landmark_dict = {}

    for i in tqdm(range(len(image_list))):
        image_name = image_list[i]
        image = io.imread(image_name)
        preds = fa_2d.get_landmarks(image)

        assert preds is not None

        landmark_dict[image_name] = preds[0][:, :2]

    with open(save_path, 'wb') as f:
        pickle.dump(landmark_dict, f)


def plot_landmark(data_dir):
    create_dir(os.path.join(data_dir, 'landmark'))

    with open(os.path.join(data_dir, 'landmark.pkl'), 'rb') as f:
        landmark_dict = pickle.load(f)

    image_list = get_file_list(os.path.join(data_dir, 'crop'))

    for image_name in tqdm(image_list):
        image = cv2.imread(image_name)
        landmark = landmark_dict[image_name]

        for point in landmark:
            image = cv2.circle(image, (point[0], point[1]), radius=0, color=(255, 0, 0), thickness=-1)

        cv2.imwrite(os.path.join(data_dir, 'landmark', os.path.basename(image_name)), image)

def drawLips(keypoints, new_img, c = (255, 255, 255), th = 1, show = False):

	keypoints = np.float32(keypoints)

	for i in range(48, 59):
		cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
	for i in range(60, 67):
		cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)

	if (show == True):
		cv2.imshow('lol', new_img)
		cv2.waitKey(10000)

def getOriginalKeypoints(kp_features_mouth, N, tilt, mean):
	kp_dn = N * kp_features_mouth * 1.5
	x, y = kp_dn[:, 0], kp_dn[:, 1]
	c, s = np.cos(tilt), np.sin(tilt)
	x_dash, y_dash = x*c + y*s, -x*s + y*c
	kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1, 1))))
	kp = kp_tilt + mean
	return kp

