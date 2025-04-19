import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import albumentations as A
from stardist.models import StarDist2D, Config2D
from csbdeep.utils import normalize
from tensorflow.keras.callbacks import EarlyStopping
from stardist.matching import matching_dataset


# === Dynamic Albumentations Augmentation with Resize ===
def build_augmentation_pipeline(target_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.ElasticTransform(alpha=120, sigma=6, alpha_affine=10, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Resize(height=target_size[0], width=target_size[1])  # Ensures consistent shape
    ])

def apply_augmentations(img, mask):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    h, w = img.shape[:2]
    augmenter = build_augmentation_pipeline(target_size=(h, w))
    img = np.clip(img, 0, 1).astype(np.float32)
    mask = mask.astype(np.uint8)
    augmented = augmenter(image=img, mask=mask)
    return augmented['image'], augmented['mask']


# === Data Loader for VIA CSV Annotations ===
class VGGDataloader:
    def __init__(self, image_dir: os.PathLike, csv_dir: os.PathLike,
                 channels=3, transpose_images=False, transpose_labels=False):
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
            if channels == 3:
                img = img.convert('RGB')
            elif channels == 1:
                img = img.convert('L')
            else:
                raise ValueError("Unsupported image channel type")

            try:
                label, class_dict = self.csv_to_label(csv_dir / csv_path, img.size)
                self.images.append(normalize(np.array(img), 1, 99.8))
                self.labels.append(label)
                self.class_maps.append(class_dict)
            except ValueError as e:
                print(f"Error in {csv_path}: {e}, skipping...")

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
                canvas.ellipse([
                    (shape[xcoord] - shape['r'], shape[ycoord] - shape['r']),
                    (shape[xcoord] + shape['r'], shape[ycoord] + shape['r'])
                ], fill=class_int, outline=class_int)
            else:
                raise ValueError('Only circular shapes supported.')

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

        return (
            (self.images[:train_size], self.labels[:train_size], self.class_maps[:train_size]),
            (self.images[train_size:], self.labels[train_size:], self.class_maps[train_size:])
        )


# === StarDist Trainer ===
class StarDistAPI:
    def __init__(self, data_dir, model_dir, epochs=1, val_per=10,
                 image_format='XYC', mask_format='XYC',
                 overwrite=False, optimize_threshold=False, config_kwargs={}, **kwargs):
        
        data_dir = Path(data_dir).absolute()
        self.image_dir = data_dir / 'images'
        self.csv_dir = data_dir / 'csv'
        self.model_dir = Path(model_dir).absolute()
        model_name = self.model_dir.name
        model_dir = self.model_dir.parent

        config = Config2D(grid=(2, 2), **config_kwargs) if overwrite else None
        self.model = StarDist2D(config=config, name=model_name, basedir=model_dir)

        channels = self.model.config.n_channel_in
        self.thresholds_optimized = not optimize_threshold
        self.epochs = epochs
        self.val_size = val_per / 100
        self.train_columns = ['dist_loss', 'prob_class_loss']
        self.val_columns = ['val_dist_loss', 'val_prob_class_loss']
        self.history_columns = self.train_columns + self.val_columns

        self.transpose_images = image_format != self.model.config.axes
        self.transpose_labels = mask_format != self.model.config.axes

        self.dataloader = VGGDataloader(
            self.image_dir, self.csv_dir,
            channels=channels,
            transpose_images=self.transpose_images,
            transpose_labels=self.transpose_labels
        )

        self.train_data, self.val_data = self.dataloader.train_test_split(1 - self.val_size)
        self.taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def train(self):
        train_x, train_y, train_maps = self.train_data
        val_x, val_y, val_maps = self.val_data       

        if not self.thresholds_optimized:
            self.model.optimize_thresholds(train_x, train_y)
            self.thresholds_optimized = True

        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        history_obj = self.model.train(
            train_x, train_y,
            validation_data=(val_x, val_y, val_maps),
            classes=train_maps,
            epochs=self.epochs,
            augmenter=apply_augmentations
        )

        self.history = pd.DataFrame(history_obj.history)[self.history_columns]
        self.history.to_csv(self.model_dir / 'training_stats.csv', index=False)
        self.history_chart()

    def history_chart(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))

        sns.lineplot(data=self.history[[self.train_columns[0], self.val_columns[0]]], ax=axes[0])
        sns.lineplot(data=self.history[[self.train_columns[1], self.val_columns[1]]], ax=axes[1])

        handles = [
            Line2D([0], [0], color='blue', linewidth=2, linestyle='solid'),
            Line2D([0], [0], color='orange', linewidth=2, linestyle='dashed')
        ]
        
        axes[0].set_title('Distance loss')
        axes[1].set_title('Class probability loss')

        for ax in axes:
            ax.set_xlabel('Steps')
            ax.set_ylabel('Loss')

        axes[0].legend(['Train', 'Val'], handles=handles)
        axes[1].legend(['Train', 'Val'], handles=handles)

        plt.savefig(self.model_dir / 'training_stats.png')

        val_x, val_y, val_maps = self.val_data
        
        predictions = [self.model.predict_instances(x, n_tiles=self.model._guess_n_tiles(x), show_tile_progress=False) for x in tqdm(val_x)]
        labels, details = list(zip(*predictions))
                
        if self.transpose_labels:
            class_labels = [self.instances_to_class_map(lbl, detail, transpose=True) for lbl, detail in predictions]
            labels = [lbl.transpose((1, 0, 2)) for lbl in labels]
        else:
            class_labels = [self.instances_to_class_map(lbl, detail['class_id']) for lbl, detail in predictions]
        
        val_maps = [(maps[k] for k in maps) for maps in val_maps]
        val_class_y = [self.instances_to_class_map(lbl, maps) for lbl, maps in zip(val_y, val_maps)]   
        
        dist_stats = [matching_dataset(val_y, labels, thresh=t, show_progress=False) for t in tqdm(self.taus)]
        class_stats = [matching_dataset(val_class_y, class_labels, thresh=t, show_progress=False) for t in tqdm(self.taus)]

        self.make_iou_plot('dist_val', dist_stats)
        self.make_iou_plot('class_val', class_stats)
        
    def make_iou_plot(self, name, stats):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        metrics = ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality')
        counts = ('fp', 'tp', 'fn')

        for m in metrics:
            ax1.plot(self.taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()

        for m in counts:
            ax2.plot(self.taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend()
        
        fig_path = self.model_dir / Path(f'{name}.png')
        fig.savefig(fig_path, dpi=300)
        
        csv_path = self.model_dir / Path(f'{name}_stats.csv')
        pd.DataFrame(stats).to_csv(csv_path)
        
    def instances_to_class_map(self, labels, class_dct, transpose=False):
        """
        Converts instance-labeled pixels to class-labeled pixels.

        Parameters:
        - labels (ndarray): 2D array from model.predict_instances, each object has a unique ID.
        - details (dict): Dictionary from model.predict_instances, may contain object metadata.
        - class_labels (list or ndarray): List of class IDs for each instance, indexed by label ID - 1.

        Returns:
        - class_map (ndarray): 2D array where each pixel is labeled with the class ID instead of instance ID.
        """
        class_map = np.zeros_like(labels, dtype=np.uint8)
        class_dct = {k+1:v+1 for k, v in enumerate(class_dct)}
        
        for instance_id in np.unique(labels):
            if instance_id == 0:
                continue  # background
            class_id = class_dct[instance_id]
            class_map[labels == instance_id] = class_id

        if transpose: class_map = class_map.transpose((1, 0))

        return class_map


# === MAIN CALL ===
if __name__ == '__main__':
    config = {
        'axes': 'YXC',
        'n_rays': 32,
        'n_channel_in': 3,
        'n_classes': 4
    }

    api = StarDistAPI(
        data_dir='Frog_Embryo_v2/second-train/second-train',
        model_dir='xenopus-4-class',
        epochs=10,
        image_format='YXC',
        mask_format='YXC',
        overwrite=True,
        optimize_threshold=False,
        config_kwargs=config
    )

    api.train()
