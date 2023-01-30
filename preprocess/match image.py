from os import listdir, mkdir
from shutil import copyfile
from os.path import exists
def makeDataset(self):
        self.data_path = '/training area/images after segementation/'  or '/verification area/images after segementation/'
        self.save_path = '/training area/images after matching/'  or '/verification area/images after matching/'
        orignal_path = self.data_path + '\\'
        des_path = self.save_path + '\\'
        element = listdir(orignal_path)
        for ele in element:
            imagepath = orignal_path + ele
            all_sample_image = listdir(imagepath)
            all_sample_index = [0] * len(all_sample_image)
            i = 0
            for sample in all_sample_image:
                all_sample_index[i] = int(sample.split('_')[1])
                i += 1
            # i = 0
            for i in range(len(all_sample_index) - 1):
                for j in range(0, len(all_sample_index) - i - 1):
                    if all_sample_index[j] > all_sample_index[j + 1]:
                        all_sample_index[j], all_sample_index[j + 1] = all_sample_index[j + 1], all_sample_index[j]
                        all_sample_image[j], all_sample_image[j + 1] = all_sample_image[j + 1], all_sample_image[j]
            # i = 0
            for path in all_sample_image:
                temp_path = path
                if not exists((des_path + temp_path.split('_')[0] + '_' + temp_path.split('_')[1] + '_' +
                               temp_path.split('_')[4] + '_' + temp_path.split('_')[5]).replace('.tif',
                                                                                                '')):
                    mkdir((des_path + temp_path.split('_')[0] + '_' + temp_path.split('_')[1] + '_' +
                           temp_path.split('_')[4] + '_' + temp_path.split('_')[5]).replace('.tif', ''))
                copyfile(orignal_path + ele + '\\' + path, (
                        des_path + temp_path.split('_')[0] + '_' + temp_path.split('_')[1] + '_' +
                        temp_path.split('_')[4] + '_' + temp_path.split('_')[5]).replace('.tif',
                                                                                         '') + '\\' + temp_path)