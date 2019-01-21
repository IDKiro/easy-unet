from __future__ import division
import numpy as np


class ImageGenerator(object):

    def __init__(self, nx, ny, a_min=None, a_max=None, **kwargs):
        self.channels = 3
        self.n_class = 2
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3

    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y

    def _load_data_and_label(self):
        data, label = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels
        
        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)

        if np.amax(data) != 0:
            data /= np.amax(data)

        return data
        
    def _next_data(self):
        data, label = self._create_image_and_label()
        return self._to_rgb(data), label

    def _create_image_and_label(self, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 20, rectangles=False):
        image = np.ones((self.nx, self.ny, 1))
        label = np.zeros((self.nx, self.ny, 3), dtype=np.bool)
        mask = np.zeros((self.nx, self.ny), dtype=np.bool)
        for _ in range(cnt):
            a = np.random.randint(border, self.nx-border)
            b = np.random.randint(border, self.ny-border)
            r = np.random.randint(r_min, r_max)
            h = np.random.randint(1,255)

            y,x = np.ogrid[-a:self.nx-a, -b:self.ny-b]
            m = x*x + y*y <= r*r
            mask = np.logical_or(mask, m)

            image[m] = h

        label[mask, 1] = 1
        
        if rectangles:
            mask = np.zeros((self.nx, self.ny), dtype=np.bool)
            for _ in range(cnt//2):
                a = np.random.randint(self.nx)
                b = np.random.randint(self.ny)
                r =  np.random.randint(r_min, r_max)
                h = np.random.randint(1,255)
        
                m = np.zeros((self.nx, self.ny), dtype=np.bool)
                m[a:a+r, b:b+r] = True
                mask = np.logical_or(mask, m)
                image[m] = h
                
            label[mask, 2] = 1
            
            label[..., 0] = ~(np.logical_or(label[...,1], label[...,2]))
        
        image += np.random.normal(scale=sigma, size=image.shape)
        image -= np.amin(image)
        image /= np.amax(image)
        
        if rectangles:
            return image, label
        else:
            return image, label[..., 1]

    def _to_rgb(self, img):
        img = img.reshape(img.shape[0], img.shape[1])
        img[np.isnan(img)] = 0
        img -= np.amin(img)
        img /= np.amax(img)
        blue = np.clip(4*(0.75-img), 0, 1)
        red  = np.clip(4*(img-0.25), 0, 1)
        green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
        rgb = np.stack((red, green, blue), axis=2)
        return rgb


class AverageMeter(object):
	def __init__(self):
		self.reset()

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