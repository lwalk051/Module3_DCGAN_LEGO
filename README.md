Module 3 — DCGAN on LEGO Bricks (Foster, Chapter 4)
ODU | GAN Coding Exercise | Option A
Dataset: LEGO bricks, approximately 40,000 grayscale 64 × 64 images.

Reproducible setup
Ran in ODU Wahab Open OnDemand JupyterLab with a Python environment containing TensorFlow, NumPy, Matplotlib and Pillow. This should download the LEGO bricks dataset shipped with Foster's Chapter 4 DCGAN notebook, using the repository's Kaggle download script (Kaggle credentials required) or download the same dataset manually. Set DATA_DIR below to the directory containing image files. This is not Fashion-MNIST.

Reference: David Foster, Generative Deep Learning (2nd ed.), O'Reilly, 2023, Chapter 4; https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan/01_dcgan

This notebook implements the Chapter 4 DCGAN design pattern (convolutional discriminator, transposed-convolution generator, adversarial BCE with non-saturating generator objective), while adding reproducible plots, samples and diagnostics required by the assignment. Compare architecture settings with Foster's notebook if exact replication is required.

Dataset setup: automatic Kaggle download or uploaded ZIP extraction is included in Section 0.

import os, sys, random, pathlib, time, json, platform, csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print('Python:',sys.version.split()[0], '| TensorFlow:',tf.__version__)
print('OS:',platform.platform())
print('GPU:',tf.config.list_physical_devices('GPU'))
2026-11-22 18:38:31.797476: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-11-22 18:38:31.833347: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-11-22 18:38:31.833389: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-11-22 18:38:31.855766: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Python: 3.10.14 | TensorFlow: 2.16.2
OS: Linux-4.18.0-553.182.1.el8_10.x86_64-x86_64-with-glibc2.34
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
0. Obtain Foster’s original LEGO bricks dataset (no Docker)
This notebook uses the exact Kaggle dataset referenced by Foster: joosthazelzet/lego-brick-images. The next cell checks for existing images, then downloads them with kagglehub if needed. It requires internet access on the Jupyter server; Kaggle may ask you to authenticate.

Run the cells in order. The first download can take time.

# Dataset acquisition — original Foster LEGO bricks images only.
from pathlib import Path
import os, sys, subprocess, zipfile

DATA_DIR = Path(os.environ.get('LEGO_DATA_DIR', './data/bricks')).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

def count_images(folder):
    return sum(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in folder.rglob('*'))

def safe_extract(zip_file, destination):
    # Refuse archive members that escape the destination folder.
    root = destination.resolve()
    with zipfile.ZipFile(zip_file) as z:
        for item in z.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Unsafe ZIP member: {item.filename}')
        z.extractall(root)

if count_images(DATA_DIR) == 0:
    # Prefer an uploaded archive; no Kaggle credentials needed for this route.
    candidates = sorted(Path('.').glob('*.zip')) + sorted(Path('data').glob('*.zip'))
    candidates = [p for p in candidates if 'lego' in p.name.lower() or 'brick' in p.name.lower() or p.name.lower() == 'archive.zip']
    if candidates:
        print('Extracting uploaded ZIP:', candidates[0])
        safe_extract(candidates[0], DATA_DIR)

if count_images(DATA_DIR) == 0:
    print('Downloading the original dataset: joosthazelzet/lego-brick-images')
    try:
        try:
            import kagglehub
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])
            import kagglehub
        downloaded = Path(kagglehub.dataset_download('joosthazelzet/lego-brick-images'))
        print('Kaggle download/cache:', downloaded)
        if count_images(downloaded):
            # Use the cached images in place instead of duplicating gigabytes of files.
            DATA_DIR = downloaded
        else:
            archives = list(downloaded.rglob('*.zip'))
            if archives:
                safe_extract(archives[0], DATA_DIR)
    except Exception as exc:
        print('Automatic download unavailable:', type(exc).__name__, str(exc)[:500])
        raise RuntimeError(
            'Download https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images '
            'on your computer, upload the ZIP beside this notebook in JupyterLab, '
            'and rerun this cell. If Kaggle requires authentication, sign in using '
            'its supported login method; do not paste API tokens into notebook cells.'
        ) from exc

image_count = count_images(DATA_DIR)
print('Image directory:', DATA_DIR.resolve())
print('Images detected:', image_count)
if image_count == 0:
    raise FileNotFoundError('No LEGO images detected. Check your uploaded ZIP and dataset folders.')
os.environ['LEGO_DATA_DIR'] = str(DATA_DIR.resolve())
Downloading the original dataset: joosthazelzet/lego-brick-images
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aaf179d0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglehub/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aaf17ca0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglehub/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aaf17f70>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglehub/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aaf40070>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglehub/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aaf40490>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglehub/
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aad74e50>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglesdk/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aad752d0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglesdk/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aad75480>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglesdk/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aad75630>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglesdk/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aad757e0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /kagglesdk/
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aada4850>: Failed to establish a new connection: [Errno -2] Name or service not known')': /tqdm/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aada4cd0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /tqdm/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aada4e80>: Failed to establish a new connection: [Errno -2] Name or service not known')': /tqdm/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aada5030>: Failed to establish a new connection: [Errno -2] Name or service not known')': /tqdm/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7ff3aada51e0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /tqdm/

[notice] A new release of pip is available: 23.0.1 -> 26.3
[notice] To update, run: pip install --upgrade pip
Downloading to /home/lwalk051/.cache/kagglehub/datasets/joosthazelzet/lego-brick-images/4.archive...
100%|██████████| 1.00G/1.00G [00:19<00:00, 53.8MB/s]
Extracting files...
Kaggle download/cache: /home/lwalk051/.cache/kagglehub/datasets/joosthazelzet/lego-brick-images/versions/4
Image directory: /home/lwalk051/.cache/kagglehub/datasets/joosthazelzet/lego-brick-images/versions/4
Images detected: 46384
1. Dataset configuration and verification
Point DATA_DIR to the extracted LEGO bricks images. Nested directories are supported. The loader accepts PNG/JPG/JPEG/BMP, converts to grayscale, resizes to 64 × 64, and normalizes to [-1, 1].

# Upload/extract the ORIGINAL Foster LEGO bricks images into data/bricks, or set LEGO_DATA_DIR.
# The notebook does not silently substitute another dataset.
DATA_DIR = pathlib.Path(os.environ.get('LEGO_DATA_DIR', './data/bricks')).expanduser()
IMG_SIZE=64; BATCH_SIZE=64; LATENT_DIM=128; EPOCHS=20
LEARNING_RATE=0.0002; BETA_1=0.5; SAMPLE_COUNT=25
OUTPUT_DIR=pathlib.Path('module3_dcgan_outputs'); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
EXTENSIONS={'.png','.jpg','.jpeg','.bmp'}
paths=sorted(p for p in DATA_DIR.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS) if DATA_DIR.is_dir() else []
print('Dataset:',DATA_DIR.resolve(),'Images:',len(paths))
if not paths:
    raise FileNotFoundError('LEGO bricks images not found. Extract the original bricks dataset to ./data/bricks or set LEGO_DATA_DIR to its image directory. No training was started.')
if len(paths)<1000: print('WARNING: fewer than 1,000 images. Confirm the full bricks dataset was extracted.')
def load_image(path):
    raw=tf.io.read_file(path)
    img=tf.io.decode_image(raw,channels=1,expand_animations=False)
    img.set_shape((None,None,1))
    img=tf.image.resize(img,(IMG_SIZE,IMG_SIZE))
    return tf.cast(img,tf.float32)/127.5-1.0
dataset=(tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
         .shuffle(min(len(paths),10000),seed=SEED,reshuffle_each_iteration=True)
         .map(load_image,num_parallel_calls=tf.data.AUTOTUNE)
         .batch(BATCH_SIZE,drop_remainder=False).prefetch(tf.data.AUTOTUNE))
preview=next(iter(dataset))
fig,axes=plt.subplots(2,5,figsize=(10,4))
for ax,img in zip(axes.flat,preview[:10]):
    ax.imshow((img.numpy().squeeze()+1)/2,cmap='gray',vmin=0,vmax=1); ax.axis('off')
fig.suptitle('Real LEGO bricks — dataset sanity check');fig.tight_layout();plt.show();plt.close(fig)
Dataset: /home/lwalk051/.cache/kagglehub/datasets/joosthazelzet/lego-brick-images/versions/4 Images: 46384
2026-11-22 18:43:11.335949: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1928] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 487 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:1a:00.0, compute capability: 7.0
2. DCGAN architectures
Discriminator produces a single logit; generator uses tanh to match [-1,1] normalization. BCE from logits is used for the original GAN-style objective. Generator objective is non-saturating: encourage discriminator to classify generated images as real.

from tensorflow.keras import layers, Model

def build_generator():
    z=layers.Input(shape=(LATENT_DIM,))
    x=layers.Dense(4*4*256,use_bias=False)(z)
    x=layers.BatchNormalization()(x);x=layers.ReLU()(x)
    x=layers.Reshape((4,4,256))(x)
    for filters in [128,64,32]:
        x=layers.Conv2DTranspose(filters,4,strides=2,padding='same',use_bias=False)(x)
        x=layers.BatchNormalization()(x);x=layers.ReLU()(x)
    out=layers.Conv2DTranspose(1,4,strides=2,padding='same',activation='tanh')(x)
    return Model(z,out,name='generator')

def build_discriminator():
    inp=layers.Input(shape=(64,64,1));x=inp
    for filters in [32,64,128,256]:
        x=layers.Conv2D(filters,4,strides=2,padding='same')(x)
        x=layers.LeakyReLU(negative_slope=0.2)(x)
        x=layers.Dropout(0.3)(x)
    x=layers.Flatten()(x)
    return Model(inp,layers.Dense(1)(x),name='discriminator')

generator=build_generator();discriminator=build_discriminator()
assert generator.output_shape==(None,64,64,1)
assert discriminator.output_shape==(None,1)
generator.summary();discriminator.summary()
Model: "generator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 4096)           │       524,288 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 4096)           │        16,384 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu (ReLU)                    │ (None, 4096)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ reshape (Reshape)               │ (None, 4, 4, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose                │ (None, 8, 8, 128)      │       524,288 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 8, 8, 128)      │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_1 (ReLU)                  │ (None, 8, 8, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_1              │ (None, 16, 16, 64)     │       131,072 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 16, 16, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_2 (ReLU)                  │ (None, 16, 16, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_2              │ (None, 32, 32, 32)     │        32,768 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_3           │ (None, 32, 32, 32)     │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_3 (ReLU)                  │ (None, 32, 32, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_3              │ (None, 64, 64, 1)      │           513 │
│ (Conv2DTranspose)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,230,209 (4.69 MB)
 Trainable params: 1,221,569 (4.66 MB)
 Non-trainable params: 8,640 (33.75 KB)
Model: "discriminator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 64, 64, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 32, 32, 32)     │           544 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu (LeakyReLU)         │ (None, 32, 32, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 32, 32, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 16, 16, 64)     │        32,832 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_1 (LeakyReLU)       │ (None, 16, 16, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 16, 16, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 8, 8, 128)      │       131,200 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_2 (LeakyReLU)       │ (None, 8, 8, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 8, 8, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 4, 4, 256)      │       524,544 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_3 (LeakyReLU)       │ (None, 4, 4, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (Dropout)             │ (None, 4, 4, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 4096)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │         4,097 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 693,217 (2.64 MB)
 Trainable params: 693,217 (2.64 MB)
 Non-trainable params: 0 (0.00 B)
3. Training, per-epoch losses, and fixed-seed sample grids
The plotted losses are epoch means of minibatch BCE values; they are not accuracy metrics. Both generator and discriminator receive independent Adam optimizers. Fixed latent vectors allow visual comparisons across epochs.

bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt=tf.keras.optimizers.Adam(LEARNING_RATE,beta_1=BETA_1)
d_opt=tf.keras.optimizers.Adam(LEARNING_RATE,beta_1=BETA_1)
fixed_z=tf.random.stateless_normal([SAMPLE_COUNT,LATENT_DIM],seed=[SEED,0])

@tf.function
def train_step(real):
    batch=tf.shape(real)[0]
    z=tf.random.normal([batch,LATENT_DIM])
    # Update discriminator using detached generated samples.
    with tf.GradientTape() as dt:
        fake=generator(z,training=True)
        real_logits=discriminator(real,training=True)
        fake_logits=discriminator(tf.stop_gradient(fake),training=True)
        d_real=bce(tf.ones_like(real_logits),real_logits)
        d_fake=bce(tf.zeros_like(fake_logits),fake_logits)
        d_loss=d_real+d_fake
    d_grads=dt.gradient(d_loss,discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_grads,discriminator.trainable_variables))
    # Update generator against the current discriminator.
    with tf.GradientTape() as gt:
        new_fake=generator(tf.random.normal([batch,LATENT_DIM]),training=True)
        new_logits=discriminator(new_fake,training=False)
        g_loss=bce(tf.ones_like(new_logits),new_logits)
    g_grads=gt.gradient(g_loss,generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads,generator.trainable_variables))
    return g_loss,d_loss,d_real,d_fake

def save_grid(epoch):
    generated=generator(fixed_z,training=False).numpy()
    fig,axes=plt.subplots(5,5,figsize=(7,7))
    for ax,img in zip(axes.flat,generated):
        ax.imshow(np.clip((img.squeeze()+1)/2,0,1),cmap='gray',vmin=0,vmax=1)
        ax.axis('off')
    fig.suptitle(f'Generated LEGO bricks — epoch {epoch} (fixed latent vectors)')
    fig.tight_layout();fig.savefig(OUTPUT_DIR/f'generated_epoch_{epoch:03d}.png',dpi=160)
    plt.show();plt.close(fig)
    return generated

config={'epochs_requested':EPOCHS,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE,
        'beta_1':BETA_1,'latent_dim':LATENT_DIM,'dataset_images':len(paths),'dataset_path':str(DATA_DIR.resolve()),
        'seed':SEED,'tensorflow':tf.__version__,'python':sys.version.split()[0],
        'platform':platform.platform(),'gpus':[str(g) for g in tf.config.list_physical_devices('GPU')]}
(OUTPUT_DIR/'config.json').write_text(json.dumps(config,indent=2))
history={'epoch':[],'g_loss':[],'d_loss':[],'d_real':[],'d_fake':[],'seconds':[],'steps':[]}
save_grid(0)
steps_total=0
for epoch in range(1,EPOCHS+1):
    start=time.perf_counter(); totals=np.zeros(4,dtype=float);steps=0
    for real in dataset:
        values=train_step(real)
        totals+=np.array([float(v.numpy()) for v in values]);steps+=1;steps_total+=1
    means=totals/steps
    history['epoch'].append(epoch)
    for key,val in zip(['g_loss','d_loss','d_real','d_fake'],means): history[key].append(float(val))
    history['seconds'].append(round(time.perf_counter()-start,2))
    history['steps'].append(steps_total)
    print(f'Epoch {epoch:03d}/{EPOCHS} | steps={steps_total} | G={means[0]:.4f} D={means[1]:.4f} | {history["seconds"][-1]:.1f}s',flush=True)
    with (OUTPUT_DIR/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history));writer.writeheader()
        writer.writerows([dict(zip(history,[history[k][i] for k in history])) for i in range(len(history['epoch']))])
    (OUTPUT_DIR/'history.json').write_text(json.dumps({'config':config,'history':history},indent=2))
    if epoch in {1,5,10,15,EPOCHS}:
        save_grid(epoch)
        generator.save_weights(str(OUTPUT_DIR/f'generator_epoch_{epoch:03d}.weights.h5'))
        discriminator.save_weights(str(OUTPUT_DIR/f'discriminator_epoch_{epoch:03d}.weights.h5'))
print('TRAINING FINISHED. Total steps:',steps_total)
2026-11-22 18:43:14.749547: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:465] Loaded cuDNN version 8907
2026-11-22 18:43:15.323791: W external/local_tsl/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 645.36MiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2026-11-22 18:43:21.178364: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indiscriminator_1/dropout_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2026-11-22 18:43:23.378365: W external/local_tsl/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 771.50MiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2026-11-22 18:46:13.535337: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indiscriminator_1/dropout_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
Epoch 001/20 | steps=725 | G=1.0098 D=1.1867 | 177.8s
2026-11-22 18:46:13.963335: W external/local_tsl/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 719.75MiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2026-11-22 18:46:14.094211: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 002/20 | steps=1450 | G=1.4254 D=1.3036 | 131.9s
2026-11-22 18:48:27.379550: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 003/20 | steps=2175 | G=1.4676 D=1.2890 | 130.9s
2026-11-22 18:50:38.264660: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 004/20 | steps=2900 | G=1.3029 D=1.3190 | 131.4s
2026-11-22 18:52:49.709072: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 005/20 | steps=3625 | G=1.1232 D=1.3252 | 131.5s
2026-11-22 18:55:01.227566: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 006/20 | steps=4350 | G=1.0597 D=1.3265 | 131.3s
2026-11-22 18:57:13.588965: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 007/20 | steps=5075 | G=0.9670 D=1.3281 | 131.6s
2026-11-22 18:59:25.168087: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 008/20 | steps=5800 | G=0.9935 D=1.3089 | 133.4s
2026-11-22 19:01:38.613852: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 009/20 | steps=6525 | G=0.9629 D=1.3068 | 132.9s
2026-11-22 19:03:51.489797: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 010/20 | steps=7250 | G=0.9963 D=1.2935 | 132.2s
2026-11-22 19:06:03.655145: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 011/20 | steps=7975 | G=0.9624 D=1.2974 | 129.3s
2026-11-22 19:08:13.945637: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 012/20 | steps=8700 | G=0.7614 D=1.3886 | 128.9s
2026-11-22 19:10:22.906680: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 013/20 | steps=9425 | G=0.9047 D=1.3097 | 129.5s
2026-11-22 19:12:32.444516: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 014/20 | steps=10150 | G=0.9283 D=1.3014 | 129.6s
2026-11-22 19:14:42.063641: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 015/20 | steps=10875 | G=0.9334 D=1.2953 | 128.5s
2026-11-22 19:16:50.584560: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 016/20 | steps=11600 | G=0.9442 D=1.2911 | 128.8s
2026-11-22 19:19:00.786497: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 017/20 | steps=12325 | G=0.9543 D=1.2789 | 128.5s
2026-11-22 19:21:09.310916: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 018/20 | steps=13050 | G=0.9681 D=1.2704 | 128.7s
2026-11-22 19:23:17.999656: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 019/20 | steps=13775 | G=0.9630 D=1.2579 | 129.9s
2026-11-22 19:25:27.959446: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 020/20 | steps=14500 | G=0.9819 D=1.2444 | 129.4s
2026-11-22 19:27:37.346139: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
TRAINING FINISHED. Total steps: 14500
4. Generator/discriminator loss curves
if not history['epoch']: raise RuntimeError('Training must complete before plotting.')
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(history['epoch'],history['g_loss'],label='Generator BCE',marker='o',markersize=3)
ax.plot(history['epoch'],history['d_loss'],label='Discriminator total BCE',marker='s',markersize=3)
ax.plot(history['epoch'],history['d_real'],label='D(real) BCE',alpha=.6,linestyle='--')
ax.plot(history['epoch'],history['d_fake'],label='D(fake) BCE',alpha=.6,linestyle='--')
ax.set(xlabel='Epoch',ylabel='Mean minibatch BCE',title='DCGAN LEGO training losses')
ax.grid(alpha=.2);ax.legend();fig.tight_layout()
fig.savefig(OUTPUT_DIR/'loss_curves.png',dpi=180);plt.show();plt.close(fig)
print('Final recorded losses:',{k:round(history[k][-1],4) for k in ['g_loss','d_loss','d_real','d_fake']})
Final recorded losses: {'g_loss': 0.9819, 'd_loss': 1.2444, 'd_real': 0.6331, 'd_fake': 0.6113}
5. Quantitative evidence and cautious failure-mode screening
These diagnostics screen for behavior; they do not establish mode collapse or oscillation automatically. Compare the plotted losses with the fixed-noise sample grids. Loss values alone do not measure image fidelity.

g=np.asarray(history['g_loss']); d=np.asarray(history['d_loss'])
changes_g=np.diff(g); changes_d=np.diff(d)
def reversals(delta):
    signs=np.sign(delta); signs=signs[signs!=0]
    return int(np.sum(signs[1:]!=signs[:-1])) if len(signs)>1 else 0
samples=generator(fixed_z,training=False).numpy()
flat=samples.reshape(SAMPLE_COUNT,-1)
# Mean pairwise image distance: heuristic only; not a validated mode-collapse score.
pairwise=np.sqrt(np.mean((flat[:,None,:]-flat[None,:,:])**2,axis=-1))
upper=pairwise[np.triu_indices(SAMPLE_COUNT,k=1)]
metrics={'epochs_completed':len(g),'training_steps':int(history['steps'][-1]),
         'total_training_seconds':round(sum(history['seconds']),2),
         'generator_first_last':[float(g[0]),float(g[-1])],
         'discriminator_first_last':[float(d[0]),float(d[-1])],
         'generator_direction_reversals':reversals(changes_g),
         'discriminator_direction_reversals':reversals(changes_d),
         'final_sample_mean_pairwise_pixel_rmse':float(upper.mean()),
         'final_sample_min_pairwise_pixel_rmse':float(upper.min())}
(OUTPUT_DIR/'diagnostics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
print('Interpretation: inspect loss_curves.png and epoch grids before assigning a failure-mode label.')
{
  "epochs_completed": 20,
  "training_steps": 14500,
  "total_training_seconds": 2656.06,
  "generator_first_last": [
    1.0097644504647831,
    0.9819109346126688
  ],
  "discriminator_first_last": [
    1.186692327754251,
    1.2443833111894542
  ],
  "generator_direction_reversals": 8,
  "discriminator_direction_reversals": 5,
  "final_sample_mean_pairwise_pixel_rmse": 0.2644461393356323,
  "final_sample_min_pairwise_pixel_rmse": 0.11181779205799103
}
Interpretation: inspect loss_curves.png and epoch grids before assigning a failure-mode label.
# Package supporting evidence for download after successful execution.
import zipfile
archive=pathlib.Path('Module3_DCGAN_LEGO_Evidence.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUTPUT_DIR.rglob('*'):
        if p.is_file(): z.write(p,arcname=str(p))
print('Evidence ZIP:',archive.resolve())
print('Save this executed .ipynb notebook too. Do not submit an unexecuted copy.')
Evidence ZIP: /home/lwalk051/Module 3/Module3_DCGAN_LEGO_Evidence.zip
Save this executed .ipynb notebook too. Do not submit an unexecuted copy.
6. Evidence-based training diagnosis (complete after execution)
Examine loss_curves.png alongside generated_epoch_001.png, intermediate grids and the final grid. Record specific epochs, visible features, and loss values. A fluctuating adversarial loss by itself is not proof of failure. Potential modes:

Oscillating loss: alternating sharp changes in G/D losses with unstable sample quality.
Mode collapse: many generated samples become near-identical despite varied input noise.
Uninformative loss: loss numbers change little while images remain noisy; the loss alone cannot establish sample quality.
Hyperparameter sensitivity: change learning rate or batch size in a separate controlled run and compare curves/images.
Fill in after training:

Training configuration and hardware: [record actual details].
Loss curve evidence (epochs and values): [record actual details].
Image grid evidence (which epochs, what changes): [record actual details].
Diagnosed failure mode or reason none is convincingly established: [record actual details].
Whether 20 epochs were sufficient or more were needed: [record actual details].
7. Predicted behavior of the other two options
WGAN-GP: A Wasserstein critic with gradient penalty should provide a more informative optimization signal when real and generated distributions barely overlap and may reduce some instability. Its critic loss is not numerically comparable to BCE; multiple critic updates and gradient penalty increase computation. It does not guarantee immunity from mode collapse.

Conditional GAN: Providing Blond_Hair labels on CelebA enables class-directed face generation, but adds label conditioning and potential class-imbalance effects. Because it changes both data domain and conditioning, direct visual comparison to LEGO DCGAN is not controlled. For LEGO bricks, equivalent conditioning would require meaningful brick labels.

Citation: Foster, D. (2023). Generative deep learning (2nd ed.). O'Reilly Media. Chapter 4. Code: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition

Disclosure example: "I used ChatGPT to help adapt and document the Jupyter notebook and organize the report. I executed the code, inspected the generated images and thimings, and verified the reported results.
