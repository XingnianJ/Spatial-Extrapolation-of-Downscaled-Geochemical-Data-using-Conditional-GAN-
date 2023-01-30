from os import listdir, mkdir
from os.path import exists
import numpy as np
from imageio import imread, imwrite
from PIL import Image as image
def clip(self):
    self.data_path='/training area/images exported from ArcGIS/'  or '/verification area/images exported from ArcGIS/'
    self.save_path = '/training area/images after segementation/'  or '/verification area/images after segementation/'
    self.stride = 1
    self.size = 16

    path = self.data_path
    save_path = self.save_path
    x_stride = y_stride = self.stride
    size = self.size

    tif_list = listdir(path)
    for tif in tif_list:
        temp = save_path + '\\' + tif.replace(path, '')
        if not exists(temp):
            mkdir(temp)
        Image = imread(path + '\\' + tif)
        Image = (Image - np.min(Image)) / (np.max(Image) - np.min(Image))
        Image = np.array(Image)
        x = 0
        y = 0
        symbol1 = 0
        symbol2 = 0
        index = 1
        while True:
            while True:
                # judge = 0
                if x + size > Image.shape[1]:
                    back_stride = x + size - Image.shape[1]
                    x = x - back_stride
                    symbol1 = 1
                if y + size > Image.shape[0]:
                    back_stride = y + size - Image.shape[0]
                    y = y - back_stride
                    symbol2 = 1
                clip_data = image.fromarray(Image[y:y + size, x:x + size])
                judge = 1
                x = x + x_stride
                if judge == 1:
                    temp_path = temp + '\\' + '2sample_' + str(index) + '_' + tif.replace('.tif',
                                                                                          '') + '_' + str(
                        x - x_stride) + '_' + str(y) + '.tif'
                    index += 1
                    imwrite(temp_path, clip_data)
                if symbol1:
                    x = 0
                    symbol1 = 0
                    break
            y += y_stride
            if symbol2:
                break