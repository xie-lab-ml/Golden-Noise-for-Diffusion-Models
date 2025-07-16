import os
import numpy as np

from torch.utils.data import DataLoader, Dataset

from utils import load_prompt, load_pick_prompt, load_pick_discard_prompt


class NoiseDataset(Dataset):
    def __init__(self,
                 prompt_version='pick',
                 pick=False,
                 all_file=False,
                 discard=False,
                 data_dir=None,
                 prompt_path=None,
                 ):

        # self.prompt_file = load_prompt(prompt_path, prompt_version)
        
        if pick:
            if discard:
                self.data_dict, self.bad_index_list = load_pick_discard_prompt(prompt_path)
            else:
                self.data_dict = load_pick_prompt(prompt_path)
        else:
            self.prompt_file, self.seed_file = load_prompt(prompt_path, prompt_version)

        self.pick = pick

        if all_file:
            self.dir_path = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]

            self.file_paths = []
            for dir in self.dir_path:
                self.file_paths.extend([os.path.join(dir, path) for path in os.listdir(dir)])
            # self.file_index = [int(file.split('/')[-1].split('_')[0]) for file in self.file_paths]
        else:
            self.file_paths = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]
            # self.file_index = [int(file.split('/')[-1].split('_')[0]) for file in self.file_paths]

        self.original_noise_list = []
        self.optimized_noise_list = []
        self.prompt_list = []

        # when you need to discard bad samples, you should comment the line below
        if not discard and not pick:
            self.set_npz()
        
        if not discard and pick:
            self.set_pick_npz()

        if discard and pick:
            self.set_pick_discard_npz()
      

    def set_pick_npz(self):

        from tqdm import tqdm
        for file in tqdm(self.file_paths):
            file_content = np.load(file)
           
            self.original_noise_list.append(file_content['arr_0'].squeeze())
            self.optimized_noise_list.append(file_content['arr_1'].squeeze())
            self.prompt_list.append(self.data_dict[file_content['arr_2'].item()])
        
        print("Finsh Loading Data !!!")

    def set_pick_discard_npz(self):

        from tqdm import tqdm
        for file in tqdm(self.file_paths):
            file_content = np.load(file)

            if file_content['arr_2'] in self.bad_index_list:
                continue
            else:
                self.original_noise_list.append(file_content['arr_0'].squeeze())
                self.optimized_noise_list.append(file_content['arr_1'].squeeze())


                try:
                    self.prompt_list.append(self.data_dict[file_content['arr_2'].item()])
                except:
                    self.optimized_noise_list = self.optimized_noise_list[:-1]
                    self.original_noise_list = self.original_noise_list[:-1]
                    continue

        print("Finsh Loading Data !!!")

    def set_npz(self):
        for file in self.file_paths:
            file_content = np.load(file)
            self.original_noise_list.append(file_content['arr_0'].squeeze())
            self.optimized_noise_list.append(file_content['arr_1'].squeeze())
            self.prompt_list.append(self.prompt_file[file_content['arr_2']])

    def __getitem__(self, idx):
        return self.original_noise_list[idx], self.optimized_noise_list[idx], self.prompt_list[idx]

    def __len__(self):
        return len(self.prompt_list)