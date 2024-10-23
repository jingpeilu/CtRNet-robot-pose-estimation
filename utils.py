import os
import random
import cv2
import sys
import numpy as np
import h5py
import torch
from torch.autograd import Variable
import transforms3d.quaternions as quaternions


from PIL import Image, ImageOps, ImageEnhance

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def calc_dis(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def norm(x, p=2):
    return np.power(np.sum(x ** p), 1. / p)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor), requires_grad=requires_grad)


def to_np(x):
    return x.detach().cpu().numpy()


'''
data utils
'''

def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt(
        (std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


'''
image utils
'''

def resize(img, size, interpolation=Image.BILINEAR):

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def crop(img, i, j, h, w):
    return img.crop((j, i, j + w, i + h))


def adjust_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):

    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


'''
record utils
'''

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def overwrite_image(image, points_predicted, color = (0,255,0), point_size = 8):

    # TODO: Separate this to another function
    height, width = image.shape[:2]

    #Clipping points so that they don't fall outside the image size
    #points_predicted[:,0] = points_predicted[:,0].clip(0, height-1)
    #points_predicted[:,1] = points_predicted[:,1].clip(0, width-1)

    points_predicted = points_predicted.astype(int)

    # Printing as a circle
    for i in range(len(points_predicted)):
        #print(points)
        
        points = points_predicted[i]
        image = cv2.circle(image,tuple(points), point_size, color, 1)
            
            #image = cv2.putText(image, str(i) + " " + str(round(scores[i],2)), tuple(points), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0,0,255), 1, cv2.LINE_AA)  
    return image


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def to_numpy_img(img):
        img = torch.squeeze(img).permute(1,2,0)
        img_np = img.detach().cpu().numpy().copy()
        img_np = normalize_data(img_np)
        return img_np



def get_inverse_pose_vrep(pose):
    R_t = np.linalg.inv(pose)
    t = R_t[:3,3]
    quat_tmp = quaternions.mat2quat(R_t[:3,:3])
    quat = [quat_tmp[1],quat_tmp[2],quat_tmp[3],quat_tmp[0]]
    pose = np.concatenate((t.reshape(-1),quat))
    return pose






class Panda_visualization(object):
    def __init__(self, fx, fy, px, py, D):
        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
        self.dist = np.array(D)


    def get_camera_matrix(self):
        P = np.array([[self.fx,     0,          self.px],
                      [0,           self.fy,    self.py],
                      [0,           0,          1]])
        return P

    def dehomogenize_3d(self, vec):
        vec = vec.reshape((-1,1))
        vec = vec/vec[3]
        return vec[:3]

    def dehomogenize_2d(self, vec):
        vec = vec.reshape((-1,1))
        vec = vec/vec[2]
        return vec[:2]

    def pose_to_matrix(self, pose):
        t = pose[:3].reshape(3,1)
        quat_tmp = pose[3:]
        quat = [quat_tmp[3],quat_tmp[0],quat_tmp[1],quat_tmp[2]]
        R = quaternions.quat2mat(quat)
        T = np.vstack((np.hstack((R,t)),[0,0,0,1]))
        return T

    def project_to_img(self, vec,pose):
        v = np.ones(4)
        v[0] = vec[0]
        v[1] = vec[1]
        v[2] = vec[2]
        p = pose @ v
        p = self.dehomogenize_3d(p)
        P = self.get_camera_matrix()
        coord = self.dehomogenize_2d(P @ p)
        #p = p.reshape(3,1)
        #coord = get_coor_by_P(p)
        return coord.astype(int)

    def draw_axis(self, img, pose):
        ori = self.project_to_img([0,0,0],pose)
        x = self.project_to_img([0.1,0,0],pose)
        y = self.project_to_img([0.0,0.1,0],pose)
        z = self.project_to_img([0.0,0,0.1],pose)
        img = cv2.line(img,tuple(ori.ravel()), tuple(x.ravel()), (255/255.0,0,0), 6)
        img = cv2.line(img,tuple(ori.ravel()), tuple(y.ravel()), (0,255/255.0,0), 6)
        img = cv2.line(img,tuple(ori.ravel()), tuple(z.ravel()), (0,0,255/255.0), 6)
        return img


class Baxter_visualization(object):
    def __init__(self, fx, fy, px, py, D):
        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
        self.dist = np.array(D)


    def get_camera_matrix(self):
        P = np.array([[self.fx,     0,          self.px],
                      [0,           self.fy,    self.py],
                      [0,           0,          1]])
        return P

    def dehomogenize_3d(self, vec):
        vec = vec.reshape((-1,1))
        vec = vec/vec[3]
        return vec[:3]

    def dehomogenize_2d(self, vec):
        vec = vec.reshape((-1,1))
        vec = vec/vec[2]
        return vec[:2]

    def pose_to_matrix(self, pose):
        t = pose[:3].reshape(3,1)
        quat_tmp = pose[3:]
        quat = [quat_tmp[3],quat_tmp[0],quat_tmp[1],quat_tmp[2]]
        R = quaternions.quat2mat(quat)
        T = np.vstack((np.hstack((R,t)),[0,0,0,1]))
        return T



    def T_from_DH(self, alp,a,d,the):
        T = np.array([[np.cos(the), -np.sin(the), 0, a],
                    [np.sin(the)*np.cos(alp), np.cos(the)*np.cos(alp), -np.sin(alp), -d*np.sin(alp)],
                    [np.sin(the)*np.sin(alp), np.cos(the)*np.sin(alp), np.cos(alp), d*np.cos(alp)],
                    [0,0,0,1]])
        return T

    def get_bl_T_Jn(self, n, theta):
        assert n in [0,2,4,6,8]
        bl_T_0 = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0.27035],
                        [0,0,0,1]])
        T_7_ee = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0.3683],
                        [0,0,0,1]])
        
        T_0_1 = self.T_from_DH(0,0,0,theta[0])
        T_1_2 = self.T_from_DH(-np.pi/2,0.069,0,theta[1]+np.pi/2)
        T_2_3 = self.T_from_DH(np.pi/2,0,0.36435,theta[2])
        T_3_4 = self.T_from_DH(-np.pi/2,0.069,0,theta[3])
        T_4_5 = self.T_from_DH(np.pi/2,0,0.37429,theta[4])
        T_5_6 = self.T_from_DH(-np.pi/2,0.010,0,theta[5])
        T_6_7 = self.T_from_DH(np.pi/2,0,0,theta[6])
        if n == 0:
            T = T_0_1
        elif n == 2:
            T = bl_T_0  @ T_0_1 @ T_1_2
        elif n == 4:
            T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4
        elif n == 6:
            T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6
        elif n == 8:
            T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ T_6_7 @ T_7_ee
        else:
            T = None
        return T


    def get_coor_by_P(self, v_pos):
        v_pos = v_pos.reshape(3,1)
        v_pos = v_pos.astype(float)
        
        p_2d,_ = cv2.projectPoints(v_pos,rvec = (0,0,0), tvec = (0,0,0),cameraMatrix = self.get_camera_matrix(), distCoeffs = self.dist)
        p_2d = p_2d.astype(int)

        return p_2d.ravel()

    def project_to_img(self, vec,pose):
        v = np.ones(4)
        v[0] = vec[0]
        v[1] = vec[1]
        v[2] = vec[2]
        p = pose @ v
        p = self.dehomogenize_3d(p)
        P = self.get_camera_matrix()
        coord = self.dehomogenize_2d(P @ p)
        #p = p.reshape(3,1)
        #coord = get_coor_by_P(p)
        return coord.astype(int)

    def draw_axis(self, img, pose):
        ori = self.project_to_img([0,0,0],pose)
        x = self.project_to_img([0.2,0,0],pose)
        y = self.project_to_img([0.0,0.2,0],pose)
        z = self.project_to_img([0.0,0,0.2],pose)
        img = cv2.line(img,tuple(ori.ravel()), tuple(x.ravel()), (255/255.0,0,0), 8)
        img = cv2.line(img,tuple(ori.ravel()), tuple(y.ravel()), (0,255/255.0,0), 8)
        img = cv2.line(img,tuple(ori.ravel()), tuple(z.ravel()), (0,0,255/255.0), 8)
        return img


    def draw_skeleton(self, img,pose,theta, line_width=5):

        def to_image(p):
            p = self.dehomogenize_3d(p)
            #p = p.reshape(3,1)
            #coord = get_coor_by_P(p)
            P = self.get_camera_matrix()
            coord = self.dehomogenize_2d(P @ p)
            coord = coord.astype(int)
            return coord

        bl_T_0 = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0.27035],
                        [0,0,0,1]])
        T_7_ee = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0.3683],
                        [0,0,0,1]])
        T_0_1 = self.T_from_DH(0,0,0,theta[0])
        T_1_2 = self.T_from_DH(-np.pi/2,0.069,0,theta[1]+np.pi/2)
        T_2_3 = self.T_from_DH(np.pi/2,0,0.36435,theta[2])
        T_3_4 = self.T_from_DH(-np.pi/2,0.069,0,theta[3])
        T_4_5 = self.T_from_DH(np.pi/2,0,0.37429,theta[4])
        T_5_6 = self.T_from_DH(-np.pi/2,0.010,0,theta[5])
        T_6_7 = self.T_from_DH(np.pi/2,0,0,theta[6])
        p = pose @ T_0_1 @ np.array([0,0,0,1]).reshape(-1,1)
        coord_1 = to_image(p)
        p = pose @ bl_T_0  @ T_0_1 @ np.array([0,0,0,1]).reshape(-1,1)
        coord_2 = to_image(p)
        img = cv2.line(img,tuple(coord_1.ravel()), tuple(coord_2.ravel()), (250/255.0,14/255.0,250/255.0), line_width)
        #img = cv2.circle(img,tuple(coord_2.ravel()), line_width, (250,14,250), -1)
        
        p = pose @ bl_T_0  @ T_0_1 @ T_1_2 @ np.array([0,0,0,1]).reshape(-1,1)
        coord_3 = to_image(p)
        img = cv2.line(img,tuple(coord_2.ravel()), tuple(coord_3.ravel()), (140/255.0,14/255.0,250/255.0), line_width)
        #img = cv2.circle(img,tuple(coord_3.ravel()), line_width, (140,14,250), -1)
        
        p = pose @ bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ np.array([0,0,0,1]).reshape(-1,1)
        coord_4 = to_image(p)
        img = cv2.line(img,tuple(coord_3.ravel()), tuple(coord_4.ravel()),  (11/255.0,14/255.0,250/255.0), line_width)
        #img = cv2.circle(img,tuple(coord_4.ravel()), line_width, (11,14,250), -1)
        
        p = pose @ bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ np.array([0,0,0,1]).reshape(-1,1)
        coord_5 = to_image(p)
        img = cv2.line(img,tuple(coord_4.ravel()), tuple(coord_5.ravel()), (11/255.0,250/255.0,209/255.0), line_width)
        #img = cv2.circle(img,tuple(coord_5.ravel()), line_width, (11,250,209), -1)
        
        p = pose @ bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ np.array([0,0,0,1]).reshape(-1,1)
        coord_6 = to_image(p)
        img = cv2.line(img,tuple(coord_5.ravel()), tuple(coord_6.ravel()), (11/255.0,250/255.0,98/255.0), line_width) 
        #img = cv2.circle(img,tuple(coord_6.ravel()), line_width, (11,250,98), -1)
        
        p = pose @ bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ np.array([0,0,0,1]).reshape(-1,1)
        coord_7 = to_image(p)
        img = cv2.line(img,tuple(coord_6.ravel()), tuple(coord_7.ravel()), (179/255.0,249/255.0,11/255.0), line_width)   
        #img = cv2.circle(img,tuple(coord_7.ravel()), line_width, (179,249,11), -1)
        
        p = pose @ bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ T_6_7 @ np.array([0,0,0,1]).reshape(-1,1)
        coord_8 = to_image(p)
        img = cv2.line(img,tuple(coord_7.ravel()), tuple(coord_8.ravel()),  (249/255.0,209/255.0,11/255.0), line_width)  
        #img = cv2.circle(img,tuple(coord_8.ravel()), line_width, (249,209,11), -1)
        
        p = pose @ bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ T_6_7 @ T_7_ee @ np.array([0,0,0,1]).reshape(-1,1)
        coord_9 = to_image(p)
        img = cv2.line(img,tuple(coord_8.ravel()), tuple(coord_9.ravel()),  (249/255.0,106/255.0,11/255.0), line_width)  
        #img = cv2.circle(img,tuple(coord_9.ravel()), line_width, (249,106,11), -1)
        
        return img





def find_ndds_data_in_dir(
    input_dir, data_extension="json", image_extension=None, requested_image_types="all",
):

    # Input argument handling
    # Expand user shortcut if it exists
    input_dir = os.path.expanduser(input_dir)
    assert os.path.exists(
        input_dir
    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir)

    assert isinstance(
        data_extension, str
    ), 'Expected "data_extension" to be a string, but it is "{}".'.format(
        type(data_extension)
    )
    data_full_ext = "." + data_extension

    if image_extension is None:
        # Auto detect based on list of image extensions to try
        # In case there is a tie, prefer the extensions that are closer to the front
        image_exts_to_try = ["png", "jpg"]
        num_image_exts = []
        for image_ext in image_exts_to_try:
            num_image_exts.append(len([f for f in dirlist if f.endswith(image_ext)]))
        max_num_image_exts = np.max(num_image_exts)
        idx_max = np.where(num_image_exts == max_num_image_exts)[0]
        # If there are multiple indices due to ties, this uses the one closest to the front
        image_extension = image_exts_to_try[idx_max[0]]
        # Mention to user if there are multiple cases to ensure they are aware of the selection
        if len(idx_max) > 1 and max_num_image_exts > 0:
            print(
                'Multiple sets of images detected in NDDS dataset with different extensions. Using extension "{}".'.format(
                    image_extension
                )
            )
    else:
        assert isinstance(
            image_extension, str
        ), 'If specified, expected "image_extension" to be a string, but it is "{}".'.format(
            type(image_extension)
        )
    image_full_ext = "." + image_extension

    assert (
        requested_image_types is None
        or requested_image_types == "all"
        or isinstance(requested_image_types, list)
    ), "Expected \"requested_image_types\" to be None, 'all', or a list of requested_image_types."

    # Read in json files
    data_filenames = [f for f in dirlist if f.endswith(data_full_ext)]

    # Sort candidate data files by name
    data_filenames.sort()

    data_names = [os.path.splitext(f)[0] for f in data_filenames if f[0].isdigit()]

    # If there are no matching json files -- this is not an NDDS dataset -- return None
    if not data_names:
        return None, None

    data_paths = [os.path.join(input_dir, f) for f in data_filenames if f[0].isdigit()]

    if requested_image_types == "all":
        # Detect based on first entry
        first_entry_name = data_names[0]
        matching_image_names = [
            f
            for f in dirlist
            if f.startswith(first_entry_name) and f.endswith(image_full_ext)
        ]
        find_rgb = (
            True
            if first_entry_name + ".rgb" + image_full_ext in matching_image_names
            else False
        )
        find_depth = (
            True
            if first_entry_name + ".depth" + image_full_ext in matching_image_names
            else False
        )
        find_cs = (
            True
            if first_entry_name + ".cs" + image_full_ext in matching_image_names
            else False
        )
        if len(matching_image_names) > 3:
            print("Image types detected that are not yet implemented in this function.")

    elif requested_image_types:
        # Check based on known data types
        known_image_types = ["rgb", "depth", "cs"]
        for this_image_type in requested_image_types:
            assert (
                this_image_type in known_image_types
            ), 'Image type "{}" not recognized.'.format(this_image_type)

        find_rgb = True if "rgb" in requested_image_types else False
        find_depth = True if "depth" in requested_image_types else False
        find_cs = True if "cs" in requested_image_types else False

    else:
        find_rgb = False
        find_depth = False
        find_cs = False

    dict_of_lists_images = {}
    n_samples = len(data_names)

    if find_rgb:
        rgb_paths = [
            os.path.join(input_dir, f + ".rgb" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                rgb_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(rgb_paths[n])
        dict_of_lists_images["rgb"] = rgb_paths

    if find_depth:
        depth_paths = [
            os.path.join(input_dir, f + ".depth" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                depth_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(depth_paths[n])
        dict_of_lists_images["depth"] = depth_paths

    if find_cs:
        cs_paths = [
            os.path.join(input_dir, f + ".cs" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                cs_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(cs_paths[n])
        dict_of_lists_images["class_segmentation"] = cs_paths

    found_images = [
        dict(zip(dict_of_lists_images, t)) for t in zip(*dict_of_lists_images.values())
    ]

    # Create output dictionaries
    dict_of_lists = {"name": data_names, "data_path": data_paths}

    if find_rgb or find_depth or find_cs:
        dict_of_lists["image_paths"] = found_images

    found_data = [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]

    # Process config files, which are data files that don't have an associated image
    found_configs = {"camera": None, "object": None, "unsorted": []}
    data_filenames_without_images = [f for f in data_filenames if not f[0].isdigit()]

    for data_filename in data_filenames_without_images:
        if data_filename == "_camera_settings" + data_full_ext:
            found_configs["camera"] = os.path.join(input_dir, data_filename)
        elif data_filename == "_object_settings" + data_full_ext:
            found_configs["object"] = os.path.join(input_dir, data_filename)
        else:
            found_configs["unsorted"].append(os.path.join(input_dir, data_filename))

    return found_data, found_configs


def transform_DREAM_to_CPLSim_TCR(TC_R):
    TC_R = TC_R.T
    TC_R[:3,-1] = TC_R[:3,-1] * 0.01
    TC_R[0,:] = -TC_R[0,:]
    TC_R[1,:] = -TC_R[1,:]
    TC_R[:3,1] = -TC_R[:3,1]
    return TC_R



def solve_pnp(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    refinement=True,
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec = cv2.solvePnP(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            flags=method,
        )

        if refinement:
            pnp_retval, rvec, tvec = cv2.solvePnP(
                canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
                projections_proc.reshape(projections_proc.shape[0], 1, 2),
                camera_K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
        tvec = tvec[:, 0]
        rvec = rvec[:, 0]

        pose_6d = np.concatenate((rvec, tvec))

    except:
        pnp_retval = False
        pose_6d = None

    return pnp_retval, pose_6d