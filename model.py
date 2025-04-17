import os
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
import pandas as pd
import json
from stardist.models import StarDist2D, Config2D
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from csbdeep.utils import normalize
import cv2
from math import ceil
import random
from stardist.matching import matching_dataset
from tqdm import tqdm


class VGGDataloader:
    def __init__(self, 
                 image_dir:os.PathLike, 
                 csv_dir:os.PathLike,
                 channels=3,
                 transpose_images=False,
                 transpose_labels=False):
        # Image and label masks
        self.images = []
        self.labels = []
        self.class_maps = []
        
        self._transpose_labels = transpose_labels
        
        image_dir, csv_dir = Path(image_dir), Path(csv_dir)
        
        image_paths = sorted(os.listdir(image_dir))
        csv_paths = sorted(os.listdir(csv_dir))
        
        for img_path, csv_path in zip(image_paths, csv_paths):
            img_path, csv_path = Path(img_path), Path(csv_path)
            img = Image.open(image_dir / img_path)
            match channels:
                case 3:
                    img = img.convert('RGB')
                case 1:
                    img = img.convert('L')
                case _:
                    raise ValueError("Unsupported img channel type")
            try:
                label, class_dict = self.csv_to_label(csv_dir / csv_path, img.size)
                self.images.append(normalize(np.array(img), 1, 99.8))
                self.labels.append(label)
                self.class_maps.append(class_dict)
            except ValueError:
                print(f"Error loading csv file {csv_path}, skipping...")
        
        
        shapes = pd.DataFrame([img.shape for img in self.images])
        common_shapes = shapes.groupby([0, 1, 2]).value_counts().sort_values(ascending=False)
        default_size = common_shapes.index[0]
                    
        zipped = list(zip(self.images, self.labels, self.class_maps))
        zipped = list(filter(lambda x: x[0].shape == default_size, zipped))
        
        random.shuffle(zipped)
        self.images, self.labels, self.class_maps = list(zip(*zipped))

        self.images = np.stack(self.images, axis=0)

        if transpose_images:
            self.images = self.images.transpose((1, 0, 2))
        
        self.labels = np.stack(self.labels, axis=0)
        
    def csv_to_label(self, path: os.PathLike, img_size):
        class_mapper_iter = {}
        if self._transpose_labels:
            img_size = tuple(reversed(img_size))
        
        def add_label(row):            
            shape = json.loads(row['region_shape_attributes'])
            classes = json.loads(row['region_attributes'])
            object_id = row['region_id'] + 1
            
            class_int = int(classes['class_name']) + 1
            
            class_mapper_iter[object_id] = class_int
            
            xcoord = 'cy' if self._transpose_labels else 'cx'
            ycoord = 'cx' if self._transpose_labels else 'cy'
            if shape['name'] == 'circle':
                canvas.circle(xy=(shape[xcoord], shape[ycoord]), radius=shape['r'], fill=class_int, outline=class_int)
            else:
                raise ValueError('Shape not a circle, please implement non circular shapes in this code')
        
        mask = Image.new(mode='L', size=img_size, color=0)
        canvas = ImageDraw.Draw(mask, mode='L')
        
        data = pd.read_csv(path)[['region_shape_attributes', 'region_attributes', 'region_id']]
        data.apply(add_label, axis=1)
        
        return np.array(mask).astype(np.uint8), class_mapper_iter
     
    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx], self.class_maps[idx])
    
    def __len__(self):
        return len(self.images)
    
    def train_test_split(self, train_proportion=.9):
        zipped = list(zip(self.images, self.labels, self.class_maps))
        random.shuffle(zipped)
        
        self.images, self.labels, self.class_maps = list(zip(*zipped))
        self.images = np.stack(self.images, axis=0)
        self.labels = np.stack(self.labels, axis=0)
        
        train_size = int(train_proportion * len(self.images))
        
        train_images = self.images[:train_size, ...]
        test_images = self.images[train_size:, ...]
        
        train_labels = self.labels[:train_size, ...]
        test_labels = self.labels[train_size:, ...]
        
        train_maps = self.class_maps[:train_size]
        test_maps = self.class_maps[train_size:]
        
        return (train_images, train_labels, train_maps), (test_images, test_labels, test_maps)
        

class StarDistAPI:
    def __init__(self,
                 data_dir,
                 model_dir,
                 epochs=1,
                 val_per=10,
                 image_format='XYC',
                 mask_format='XYC',
                 overwrite=False,
                 optimize_threshold=False,
                 config_kwargs={},
                 **kwargs):
        
        data_dir = Path(data_dir).absolute()
        self.image_dir = Path(data_dir) / Path('images')
        self.mask_dir = Path(data_dir) / Path('masks')
        self.csv_dir = Path(data_dir) / Path('csv')
        
        self.model_dir = Path(model_dir).absolute()
        model_name = self.model_dir.name
        model_dir = self.model_dir.parent
        
        config = Config2D(grid=(2, 2), **config_kwargs) if overwrite else None
        
        self.model = StarDist2D(config=config,
                                name=model_name,
                                basedir=model_dir)
        
        channels = self.model.config.n_channel_in
        
        self.thresholds_optimized = not optimize_threshold
         
        self.epochs = epochs
        
        self.val_size = val_per / 100
        self.image_format = image_format
        self.mask_format = mask_format
        
        self.train_columns = ['dist_loss', 'prob_class_loss']
        self.val_columns = ['val_dist_loss', 'val_prob_class_loss']
        
        self.history_columns = self.train_columns + self.val_columns
        
        tranpose_images = image_format != self.model.config.axes
        tranpose_labels = mask_format != self.model.config.axes

        self.dataloader = VGGDataloader(self.image_dir, 
                                        self.csv_dir,
                                        channels=channels,
                                        transpose_images=tranpose_images,
                                        transpose_labels=tranpose_labels)
        
        train, val = self.dataloader.train_test_split(1 - self.val_size)
        self.train_data = train
        self.val_data = val
        
    def train(self):
        train_x, train_y, train_maps = self.train_data
        val_x, val_y, val_maps = self.val_data
        
        if not self.thresholds_optimized:
            self.model.optimize_thresholds(train_x, train_y)
            self.thresholds_optimized = True
        
        history_obj = self.model.train(train_x, train_y, validation_data=(val_x, val_y, val_maps), classes=train_maps, epochs=self.epochs)
        self.history = pd.DataFrame(history_obj.history)[self.history_columns]
        self.history_chart()
                
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

        val_x, val_y, _ = self.val_data
        predictions = [self.model.predict_instances(x, n_tiles=self.model._guess_n_tiles(x), show_tile_progress=False)[0] for x in tqdm(val_x)]

        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = [matching_dataset(val_y, predictions, thresh=t, show_progress=False) for t in tqdm(taus)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        metrics = ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality')
        counts = ('fp', 'tp', 'fn')

        for m in metrics:
            ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()

        for m in counts:
            ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend()
        
        fig_path = self.model_dir / Path('validation_plot.png')
        fig.savefig(fig_path, dpi=300)
        
        csv_path = self.model_dir / Path('validation_stats.csv')
        pd.DataFrame(stats).to_csv(csv_path)
        
        
if __name__ == '__main__':
    classes=4
    
    image_format = 'YXC'
    mask_format = 'YXC'
    
    config = {
        'axes': 'YXC',
        'n_rays': 32,
        'n_channel_in': 3,
        'n_classes': classes
    }
    
    api = StarDistAPI(
        data_dir='Frog_Embryo_v2/second-train/second-train',
        model_dir='xenopus-4-class',
        epochs=2,
        image_format='YXC',
        mask_format='YXC',
        overwrite=True,
        optimize_threshold=False,
        config_kwargs=config
    )
    
    api.train()