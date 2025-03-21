import tensorflow as tf
from stardist.models import StarDist2D, Config2D
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os
import json
from matplotlib import pyplot as plt
from csbdeep.utils import normalize
import cv2
from math import ceil
import sys

class StarDistAPI:
    def __init__(self,
                 image_dir,
                 model_dir,
                 epochs=1,
                 batch_size=4,
                 val_per=20,
                 image_format='XYC',
                 mask_format='XYC',
                 imagej=True,
                 overwrite=False,
                 config_kwargs={},
                ):
        
        image_dir = Path(image_dir).absolute()
        self.image_dir = Path(image_dir) / Path('images')
        self.mask_dir = Path(image_dir) / Path('masks')
        self.csv_dir = Path(image_dir) / Path('csv')
        
        self.model_dir = Path(model_dir).absolute()
        model_name = self.model_dir.name
        model_dir = self.model_dir.parent
        
        config = Config2D(grid=(2, 2), **config_kwargs) if overwrite else None
        
        self.model = StarDist2D(config=config,
                                name=model_name,
                                basedir=model_dir)
        
        if config is not None:
            self.thresholds_optimized = False
        else:
            self.thresholds_optimized = True
         
        self.epochs = epochs
        self.val_size = max(int(ceil((val_per / 100) * batch_size)), 
                            1)
        self.image_format = image_format
        self.mask_format = mask_format
        self.imagej = imagej
        self.batch_size = batch_size
        
        self.train_columns = ['dist_loss', 'prob_class_loss']
        self.val_columns = ['val_dist_loss', 'val_prob_class_loss']
        
        self.history_columns = self.train_columns + self.val_columns
        self.history = pd.DataFrame([], columns=self.history_columns)
        
        
    def train(self):
        if self.imagej:
            loader = ImageJLoader(self.image_dir, self.mask_dir)
        if not self.imagej:
            loader = VGGLoader(self.image_dir, self.csv_dir)
            
        #TODO: DELETE
        #sys.exit()    
        
        length = len(loader)

        for epoch in range(self.epochs):    
            X_train, y_train, classes_train = [], [], []
            X_val, y_val, classes_val = [], [], []
            dataset = loader.data()
            for X, y, classes in dataset:
                if len(X_val) < self.val_size:
                    X_val.append(X)
                    y_val.append(y)
                    classes_val.append(classes)
                else:
                    X_train.append(X)
                    y_train.append(y)
                    classes_train.append(classes)
                
                if len(X_train) == self.batch_size:
                    X_arr = np.array(X_train)
                    X_val_arr = np.array(X_val)
                    
                    if self.model.config.axes != self.image_format:
                        X_arr = X_arr.transpose(axes=(1, 0, 2))
                        X_val_arr = X_val_arr.transpose(axes=(1, 0, 2))

                    y_arr = np.array(y_train)
                    y_val_arr = np.array(y_val)
                  
                    if self.model.config.axes != self.mask_format:
                        y_arr = y_arr.transpose(axes=(1, 0, 2))
                        y_val_arr = y_val_arr.transpose(axes=(1, 0, 2))
                    
                    classes_arr = classes_train.copy()
                    classes_val_arr = classes_val.copy()
                    
                    
                    
                    X_train = []; y_train = []; classes_train = []
                    X_val = []; y_val = []; classes_val = []
                                        
                    history = self.model.train(X_arr, y_arr, validation_data=(X_val_arr, y_val_arr, classes_val_arr), epochs=1, classes=classes_arr)
                    self._add_history(history)
                    
                    yield (length * epoch + (length - len(loader))) / (length * self.epochs) # Percentage of training done
                    
            if not self.thresholds_optimized:
                self.model.optimize_thresholds(X_arr, y_arr)
                self.thresholds_optimized = True
        
        yield 1
        
    def _add_history(self, history):
        history = pd.DataFrame(history.history)[self.history_columns]
        self.history = pd.concat([self.history, history], axis=0, ignore_index=True)
        
    def history_chart(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        
        sns.lineplot(
            data=self.history[[self.train_columns[0], self.val_columns[0]]],
            ax=axes[0],
            legend=True
        )
        
        sns.lineplot(
            data=self.history[[self.train_columns[1], self.val_columns[1]]],
            ax=axes[1],
            legend=True
        )
        
        handles = [
            Line2D([0], [0], color='blue', linewidth=2, linestyle='solid'),
            Line2D([0], [0], color='orange', linewidth=2, linestyle='dashed')
        ]
        
        axes[0].set_title('Distance loss')
        axes[1].set_title('Class probability loss')
        
        axes[0].set_xlabel('Steps')
        axes[1].set_xlabel('Steps')
        
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel('Loss')
        
        axes[0].legend(handles=handles ,labels=['Training Distance Loss', 'Validation Distance Loss'])
        axes[1].legend(handles=handles ,labels=['Training Class Loss', 'Validation Class Loss'])
        
        plt.savefig(self.model_dir / Path('training_stats.png'))
        
        
class Loader:
    def __init__(self,
                 image_dir: os.PathLike,
                 label_dir: os.PathLike,
                 memory_limit:int=32):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.memory_limit = memory_limit
        
        self.image_paths = list(map(lambda x: Path(str(self.image_dir / Path(x))),
                                    os.listdir(self.image_dir)))
        
        self.label_paths =  list(map(lambda x: Path(str(self.label_dir / Path(x))),
                                    os.listdir(self.label_dir)))
        
        self.image_len = (0, min(self.memory_limit, len(self.image_paths)))
        self.length = len(self.image_paths)
        
        self._load_images()
    
    def _load_images(self):
        if self.image_len[0] >= self.length:
            self.image_len = (0, min(self.memory_limit, len(self.image_paths)))
            
        self.images = {}
        self.labels = {}
        self.class_mappings = {}
        
        for index in range(*self.image_len):
            name = self.image_paths[index].stem
            path = self.image_paths[index]
            
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                img_rgb = np.array(img_rgb)
                img_rgb = normalize(img_rgb, 1, 99.8)
                self.images[name] = img_rgb
                self._load_label(name, img.size)
            
    def _load_label(self, name, size):
       pass
    
    def data(self):
        for index in range(len(self.image_paths)):
            name = self.image_paths[index].stem
            
            if len(self.images) == 0 and index != len(self.image_paths) - 1:
                self.image_len = (self.image_len[1]+1, min(self.image_len[1] + self.memory_limit, len(self.image_paths) - self.image_len[1]))
                self._load_images()
            
            img, lbl, mapping = self.images.pop(name), self.labels.pop(name), self.class_mappings.pop(name)
            self.length -= 1
            
           
            yield img, lbl, mapping
        self.length = len(self.image_paths)

        
    def __len__(self):
        return self.length
    
#TODO: Rewrite to include object differentiation to class mapping
class ImageJLoader(Loader):
    def __init__(self,
                 image_dir: os.PathLike,
                 mask_dir: os.PathLike,
                 memory_limit: int=32):
        
        super().__init__(image_dir=image_dir, label_dir=mask_dir, memory_limit=memory_limit)
        
    #Overwrite
    def _load_label(self, name, *args):
        path = list(self.label_dir.glob(f"{name}.*"))[0]
        with Image.open(path) as label:
            
            mask = self._binary_mapper(label)
            self.labels[name] = mask
            
            found_classes = self._get_countour_pixels(label)
            object_ids = [i+1 for i in range(len(found_classes))]
            class_mapping = {k:v for k, v in zip(object_ids, found_classes)}
            self.class_mappings[name] = class_mapping
            
    def _binary_mapper(self, img: Image.Image):
        """Turns a single channel image into a binary mask where any color results in a 1, and 0s lead to 0

        Args:
            img (PIL.Image.Image): The input image in mode 'L'

        Returns:
            mask (nd.array): The boolean array
        """
        data = np.array(img)
        mask = np.ma.make_mask(data)
        mask = mask.astype(np.uint8)
        return mask
    
    def _get_countour_pixels(self, img: Image.Image):
        # L/greyscale image
        data = np.array(img)
        contours = cv2.findContours(data, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pixel_lst = []
        for contour in contours:
            moment = cv2.moments(contour)
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00']) 
            pixel_lst.append(data[cx, cy])
        return pixel_lst
            
    
class VGGLoader(Loader):
    def __init__(self,
                 image_dir: os.PathLike,
                 csv_dir: os.PathLike,
                 memory_limit:int=32):
        super().__init__(image_dir=image_dir, label_dir=csv_dir, memory_limit=memory_limit)
        

    def _load_label(self, name, size):
        path = self.label_dir / Path(name + ".csv")
        self.labels[name], self.class_mappings[name] = self._csv_to_label_mask(path, size)
        
    
    def _csv_to_label_mask(self, path, img_size):
        class_mapper_iter = {}
        
        def add_label(row):
            class_mapper_iter
            
            shape = json.loads(row['region_shape_attributes'])
            classes = json.loads(row['region_attributes'])
            object_id = row['region_id'] + 1
            
            class_int = int(classes['class_name']) + 1
            
            class_mapper_iter[object_id] = class_int
            
            if shape['name'] == 'circle':
                canvas.circle(xy=(shape['cx'], shape['cy']), radius=shape['r'], fill=class_int, outline=class_int)
                #canvas.circle(xy=(shape['cx'], shape['cy']), radius=shape['r'], fill=object_id, outline=object_id)
            else:
                raise ValueError('Shape not a circle, please implement non circular shapes in this code')
        
        mask = Image.new(mode='L', size=img_size, color=0)
        canvas = ImageDraw.Draw(mask, mode='L')
        
        
        data = pd.read_csv(path)[['region_shape_attributes', 'region_attributes', 'region_id']]
        data.apply(add_label, axis=1)

        # im = plt.imshow(mask,
        #            interpolation=None)
        
        # from matplotlib import patches as mpatches
        # values = list(range(1, 5))
        # labels = ["No split", "1-split", "2-split", "3-split"]
        # colors = [im.cmap(im.norm(value)) for value in values]
        # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values)) ]
        
        
        # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        # plt.show()
        
        return np.array(mask).astype(np.uint8), class_mapper_iter
    
if __name__ == '__main__':
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
           
    #images_dir needs a subdirectory named images and one named csv
    image_dir = "data"
    
    #model dir where model is stored or being written to
    model_dir = "test"
    
    epochs = 1
    batch_size = 4
    validation_percentage = 20
    image_format = 'YXC'
    mask_format = 'YXC'
    imagej = False
    
    # Makes a new model, turn off once one is created
    # This consumes a lot of memory, be careful
    overwrite = False
    
    
    classes = 4
    
    config = {
        'axes': 'YXC',
        'n_rays': 32,
        'n_channel_in': 3,
        'n_classes': classes
    }

    
    model = StarDistAPI(image_dir,
                        model_dir,
                        epochs,
                        batch_size,
                        validation_percentage,
                        image_format,
                        mask_format,
                        imagej,
                        overwrite,
                        config_kwargs=config
                        )
    
    for progress in model.train():
        print(f"Progress: {progress*100}% done")
    model.history_chart()

# if __name__ == '__main__':
    
#     train = False
    
#     if train:
#         model = StarDist2D(config=Config2D(
#                 n_channel_in=3
#             ), name='empty-model', basedir='test/model')
#         model.prepare_for_training()

#         X_train = []
#         y_train = []
#         for _ in range(10):
#             X, y = next(dataset)
#             X_train.append(X)
#             y_train.append(y)
            
#         X_val = []
#         y_val = []
#         for _ in range(5):
#             X, y = next(dataset)
#             X_val.append(X)
#             y_val.append(y)
        
#     else:
#         model = StarDist2D(config=None, name='empty-model', basedir='test/model')
   
#     loader.give_counts = True
#     X_test = []
#     y_test = []
#     for _ in range(5):
#         X, y = next(dataset)
#         X_test.append(X)
#         y_test.append(y)
    
#     if train:
#         loader.give_counts = False
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
        
#         X_val = np.array(X_val)
#         y_val = np.array(y_val)    
        
#         print('Model is training...')
#         history = model.train(X_train, y_train, validation_data=(X_val, y_val), epochs=1)
#         #model.
#         print('Model done training!')
    
       
#     X_test = np.array(X_test)
#     y_test = np.array(y_test)
    
    
#     print('Model is predicting...')
#     for img in X_test:
#         mask, output = model.predict_instances(img, n_tiles=model._guess_n_tiles(img))
#         break
   
#     print('Model done predicting!')
    
#     print(len(output['points']), y_test[0])
#     print(output)
#     print(mask.shape)