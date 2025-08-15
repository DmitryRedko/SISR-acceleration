import os, cv2, torch, random, numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def _to_float01(img_u8):  return img_u8.astype(np.float32) / 255.0
def _to_uint8(img_f):     return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(img, b_range=(0.8, 1.2), c_range=(0.8, 1.2)):
    b = random.uniform(*b_range)
    c = random.uniform(*c_range)
    out = img * c * b
    return np.clip(out, 0.0, 1.0)

def adjust_gamma(img, g_range=(0.8, 1.2)):
    gamma = random.uniform(*g_range)
    # избегаем 0
    img = np.clip(img, 1e-6, 1.0)
    return np.power(img, gamma)

def jitter_hsv(img, h_shift=(-8, 8), s_mult=(0.9, 1.1), v_mult=(0.9, 1.1)):
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    H = (H + random.uniform(*h_shift)) % 180.0
    S = np.clip(S * random.uniform(*s_mult), 0, 255)
    V = np.clip(V * random.uniform(*v_mult), 0, 255)
    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)

def white_balance_shift(img, gain_range=(0.9, 1.1)):
    gains = np.array([random.uniform(*gain_range),
                      random.uniform(*gain_range),
                      random.uniform(*gain_range)], dtype=np.float32)
    out = img * gains[None,None,:]
    out = out / (np.mean(gains) + 1e-6)
    return np.clip(out, 0.0, 1.0)

def vignette(img, strength=(0.0, 0.25)):
    s = random.uniform(*strength)
    if s <= 1e-6: return img
    h, w, _ = img.shape
    Y, X = np.ogrid[:h, :w]
    cy, cx = h/2, w/2
    r = np.sqrt((Y-cy)**2 + (X-cx)**2)
    mask = 1.0 - s * (r / r.max())**2
    mask = mask[...,None]
    out = img * mask
    return np.clip(out, 0.0, 1.0)

def jpeg_compress(img, q=(60, 95)):
    qv = random.randint(*q)
    enc = cv2.imencode('.jpg', _to_uint8(img), [int(cv2.IMWRITE_JPEG_QUALITY), qv])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(dec, 0.0, 1.0)

def slight_blur(img, k=(0, 1)):
    if random.randint(*k) == 0: return img
    return cv2.GaussianBlur(img, (3,3), 0)

def add_rain(img, drop_prob=0.15, length=(8, 14), thickness=(1, 2), alpha=(0.05, 0.12), angle=(-25, 25)):
    if random.random() > drop_prob:
        return img
    h, w, _ = img.shape
    rain = np.zeros((h, w), dtype=np.float32)
    count = int(0.002 * h * w)  
    L = random.randint(*length)
    T = random.randint(*thickness)
    ang = np.deg2rad(random.uniform(*angle))
    dx = int(np.cos(ang) * L)
    dy = int(np.sin(ang) * L)

    for _ in range(count):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        x2 = np.clip(x+dx, 0, w-1)
        y2 = np.clip(y+dy, 0, h-1)
        cv2.line(rain, (x, y), (x2, y2), color=1.0, thickness=T)

    rain = cv2.GaussianBlur(rain, (3,3), 0)
    a = random.uniform(*alpha)
    rain_rgb = np.stack([rain, rain, rain], axis=-1)
    out = np.clip(img*(1.0 - a) + rain_rgb*a, 0.0, 1.0)
    return out

def add_noise(img, kind=("gauss","poisson","speckle"), gauss_sigma=(0.005,0.03), speckle_sigma=(0.01,0.05), poisson_scale=(20,80)):
    t = random.choice(kind)
    if t == "gauss":
        sigma = random.uniform(*gauss_sigma)
        noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
        out = img + noise
    elif t == "speckle":
        sigma = random.uniform(*speckle_sigma)
        noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
        out = img + img * noise
    else:  # poisson
        scale = random.uniform(*poisson_scale)
        out = np.random.poisson(np.clip(img,0,1) * scale) / float(scale)
    return np.clip(out, 0.0, 1.0)

def photometric_augment(img):
    """Комбинируем мягкие фото-аугментации без геометрии."""
    if random.random() < 0.7: img = adjust_brightness_contrast(img)
    if random.random() < 0.5: img = adjust_gamma(img)
    if random.random() < 0.5: img = jitter_hsv(img)
    if random.random() < 0.5: img = white_balance_shift(img)
    if random.random() < 0.4: img = vignette(img)
    if random.random() < 0.4: img = jpeg_compress(img)
    if random.random() < 0.3: img = slight_blur(img)
    if random.random() < 0.3: img = add_rain(img)
    return img

class MultiImagePatchesDataset(Dataset):
    def __init__(self, noisy_images, gt_images, num_noisy=6, crop_size=256, seed=None):
        """
        num_noisy: сколько версий noisy склеиваем по каналам (первая — исходная).
        crop_size: размер патча (резка на 4 квадранта).
        """
        self.num_noisy = num_noisy
        self.crop_size = crop_size
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        self.noisy_patches, self.gt_patches = [], []

        print("Генерация патчей...")
        for ni, gi in tqdm(zip(noisy_images, gt_images), total=len(noisy_images)):
            noisy_img = cv2.cvtColor(cv2.imread(ni), cv2.COLOR_BGR2RGB)
            gt_img    = cv2.cvtColor(cv2.imread(gi), cv2.COLOR_BGR2RGB)

            h, w, _ = noisy_img.shape
            mid_h, mid_w = h // 2, w // 2
            coords = [(0,0,mid_h,mid_w), (0,mid_w,mid_h,w), (mid_h,0,h,mid_w), (mid_h,mid_w,h,w)]

            for y1,x1,y2,x2 in coords:
                n_patch = cv2.resize(noisy_img[y1:y2, x1:x2], (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                g_patch = cv2.resize(gt_img[y1:y2, x1:x2],    (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                self.noisy_patches.append(n_patch)
                self.gt_patches.append(g_patch)

    def __len__(self): return len(self.noisy_patches)

    def __getitem__(self, idx):
        base_noisy_u8 = self.noisy_patches[idx]
        gt_u8         = self.gt_patches[idx]

        base_noisy = _to_float01(base_noisy_u8)
        gt         = _to_float01(gt_u8)

        noisy_versions = []
        for i in range(self.num_noisy):
            if i == 0:
                cur = base_noisy                    
            else:
                cur = photometric_augment(base_noisy)  
                cur = add_noise(cur)               
            noisy_versions.append(torch.from_numpy(cur.transpose(2,0,1)).float())

        noisy_stacked = torch.cat(noisy_versions, dim=0)                 # [num_noisy*3, H, W]
        gt_tensor     = torch.from_numpy(gt.transpose(2,0,1)).float()    # [3, H, W]
        return noisy_stacked, gt_tensor
