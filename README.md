DCGAN LEGO Bricks Image Generation

Old Dominion University (ODU)
Generative AI – Module 3 GAN Coding Exercise
Option A: Deep Convolutional Generative Adversarial Network (DCGAN)

Author: Larkin Walker
Date: November 2026
Reference: David Foster, Generative Deep Learning, 2nd Edition, Chapter 4

1. Project Overview

This project implements and trains a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic grayscale images of LEGO bricks.

The implementation follows the DCGAN approach presented in David Foster's Generative Deep Learning (2nd Edition), with additional functionality for monitoring training performance, visualizing generated images, and evaluating potential GAN training instability.

The experiment was conducted using Old Dominion University's Wahab High-Performance Computing (HPC) environment through Open OnDemand JupyterLab.

The primary objectives were to:

Train a DCGAN using the original LEGO bricks dataset.

Monitor generator and discriminator training losses.

Generate synthetic LEGO brick images using random latent vectors.

Evaluate training behavior and potential instability.

Compare DCGAN conceptually with WGAN-GP and Conditional GAN architectures.

2. Repository Structure

Module3_DCGAN_LEGO/
│
├── README.md
│
├── notebook/
│   └── DCGAN LEGO Jupyter notebook
│
├── report/
│   └── Module 3 experiment report
│
└── results/
    ├── config.json
    ├── history.csv
    ├── history.json
    ├── diagnostics.json
    ├── loss_curves.png
    ├── generated_epoch_000.png
    ├── generated_epoch_001.png
    ├── generated_epoch_005.png
    ├── generated_epoch_010.png
    ├── generated_epoch_015.png
    ├── generated_epoch_020.png
    └── model checkpoint files

The filenames shown above describe the expected experiment outputs. The notebook and report filenames may differ from those displayed.

3. Dataset

Dataset: LEGO Brick Images
Source: Kaggle
Dataset Author: joosthazelzet
Images detected: 46,384
Image resolution: 64 × 64 pixels
Color format: Grayscale

Dataset link: https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images

The dataset contains images of LEGO bricks that are used to train the generator to produce similar synthetic images.

Preprocessing

The images are:

Loaded from the dataset directory.

Converted to grayscale.

Resized to 64 × 64 pixels.

Normalized to the range [-1, 1].

Shuffled and organized into batches of 64 images.

The notebook supports downloading the original dataset through Kaggle or manually uploading the dataset as a ZIP archive.

No alternative dataset, such as Fashion-MNIST, was substituted.

4. Computing Environment

The experiment was executed on ODU's Wahab HPC environment using JupyterLab.

Component

Configuration

Operating System

Linux

Python

3.10.14

TensorFlow

2.16.2

GPU

NVIDIA Tesla V100-SXM2-16GB

Execution Environment

ODU Wahab Open OnDemand

Random Seed

42

Docker

Not required

Python Dependencies

pip install tensorflow==2.16.2 numpy matplotlib pillow kagglehub

The installation command assumes a compatible Python environment with access to the required packages.

5. DCGAN Architecture

The model consists of two neural networks trained adversarially.

Generator

The generator converts a randomly sampled latent vector into a synthetic LEGO brick image.

Latent dimension: 128

Dense projection and reshaping layers

Transposed convolutional layers

Batch normalization

ReLU activation

Tanh output activation

Output: 64 × 64 × 1 grayscale image

Total generator parameters: 1,230,209

Discriminator

The discriminator attempts to distinguish real LEGO brick images from generated images.

Convolutional layers

LeakyReLU activation

Dropout regularization

Flatten and dense output layers

Single output logit

Total discriminator parameters: 693,217

The generator and discriminator are trained using binary cross-entropy losses with independent Adam optimizers.

6. Training Configuration

Parameter

Value

Training epochs

20

Batch size

64

Learning rate

0.0002

Adam beta_1

0.5

Latent dimension

128

Training images

46,384

Total training steps

14,500

Random seed

42

Sample grid

25 images (5 × 5)

The notebook records generator and discriminator losses after every epoch.

Generated image grids are saved at initialization and selected training epochs to allow visual comparison of model progress.

7. Experimental Results

The DCGAN successfully completed all 20 training epochs.

Training Performance

Metric

Result

Completed epochs

20

Total training steps

14,500

Total training time

2,656.06 seconds

Initial generator loss

1.0098

Final generator loss

0.9819

Initial discriminator loss

1.1867

Final discriminator loss

1.2444

Final discriminator real loss

0.6331

Final discriminator fake loss

0.6113

Total training time was approximately 44.3 minutes.

Training Loss Interpretation

The generator and discriminator losses fluctuated throughout training, which is common in adversarial learning.

The generator loss initially increased before decreasing during subsequent epochs. The discriminator loss also changed throughout the experiment.

The final generator loss was approximately 0.9819, while the discriminator loss was approximately 1.2444.

These values indicate that both networks continued participating in adversarial optimization. However, GAN loss values alone cannot establish the realism or diversity of generated images.

Training Diagnostics

Additional diagnostic measurements included:

Diagnostic

Result

Generator loss direction reversals

8

Discriminator loss direction reversals

5

Mean pairwise generated-image pixel RMSE

0.2644

Minimum pairwise generated-image pixel RMSE

0.1118

The direction reversals indicate that training losses did not follow a strictly monotonic trend.

Pairwise image-distance measurements provide a preliminary indication of variation among generated samples. However, they are not validated measures of image quality or mode collapse.

8. Generated Images

The notebook saves generated image grids at the following training stages:

Epoch 0: Untrained generator

Epoch 1: Initial training

Epoch 5: Early training

Epoch 10: Intermediate training

Epoch 15: Later training

Epoch 20: Final training

Each grid contains 25 generated grayscale images.

The same fixed latent vectors are used across the saved grids, allowing visual comparisons of the generator's output as training progresses.

The image grids should be reviewed alongside the loss curves when evaluating whether the model learned meaningful LEGO brick features.

9. Training Stability and Limitations

GAN training is challenging because the generator and discriminator are optimized against one another.

Potential failure modes include:

Training oscillation: Generator and discriminator performance may fluctuate as each network adapts to the other.

Mode collapse: The generator may produce a limited range of similar images rather than representing the diversity of the training dataset.

Uninformative losses: Changes in adversarial loss may not correspond directly to improvements in image quality.

The completed experiment exhibited fluctuations in generator and discriminator losses.

However, the numerical diagnostics alone do not establish that mode collapse or severe training instability occurred. A stronger conclusion requires inspection of the generated image grids and comparison with real LEGO brick images.

The experiment also did not calculate formal image-generation metrics such as Fréchet Inception Distance (FID).

10. Comparison With Other GAN Approaches

WGAN-GP

Wasserstein GAN with Gradient Penalty uses a critic and a Wasserstein-based training objective rather than the standard discriminator binary cross-entropy objective.

Potential advantages include improved optimization behavior and more informative training signals in some settings.

However, WGAN-GP introduces additional computational requirements, including gradient penalty calculations and potentially multiple critic updates.

Conditional GAN

A Conditional GAN incorporates additional information, such as class labels, to control generated outputs.

For example, conditioning could allow a generator to produce specific categories of images.

Unlike an unconditional DCGAN, a Conditional GAN requires suitable conditioning information during training.

Experimental Scope

Only Option A, DCGAN with LEGO bricks, was implemented and trained for this assignment.

WGAN-GP and Conditional GAN were discussed conceptually and were not experimentally evaluated.

11. Reproducing the Experiment

Step 1: Download the Repository

git clone https://github.com/lwalk051/Module3_DCGAN_LEGO.git
cd Module3_DCGAN_LEGO

Step 2: Open JupyterLab

Launch JupyterLab using ODU Wahab Open OnDemand or another compatible Python environment.

Step 3: Install Dependencies

Ensure TensorFlow, NumPy, Matplotlib, Pillow, and KaggleHub are available.

Step 4: Obtain the Dataset

Download the original LEGO Brick Images dataset from Kaggle.

Alternatively, use the notebook's automatic dataset acquisition functionality if network access and authentication are available.

Step 5: Run the Notebook

Open the notebook located in the notebook/ directory and execute the cells in order.

The notebook will:

Configure the environment.

Locate or download the LEGO dataset.

Preprocess the images.

Construct the generator and discriminator.

Train the DCGAN.

Save training losses and image grids.

Calculate diagnostic metrics.

Package experimental evidence.

Step 6: Review Results

Examine the generated output directory for training logs, loss curves, generated images, model checkpoints, and diagnostic results.

Training duration may vary depending on GPU allocation and system resources.

12. Troubleshooting

Dataset not found

Verify that the original LEGO dataset has been downloaded or uploaded and that the LEGO_DATA_DIR environment variable points to the correct image directory.

Kaggle download fails

Download the dataset manually from Kaggle and upload the ZIP archive into JupyterLab.

GPU memory warnings

Reduce the batch size if memory allocation failures prevent execution. Record the changed configuration because it differs from the baseline experiment.

TensorFlow warnings

Some CUDA initialization and dataset-end warnings appeared during execution. Despite these messages, the recorded training run completed all 20 epochs.

13. Conclusion

This project demonstrated the implementation and training of a DCGAN using the original LEGO brick image dataset.

The model completed 20 epochs and 14,500 training steps on ODU's Wahab HPC environment.

Training loss curves and diagnostic measurements were recorded, and generated image grids were saved at selected epochs.

The experiment illustrates both the practical workflow of adversarial image generation and the importance of evaluating GAN outputs using more than training loss alone.

Future improvements could include additional training epochs, controlled hyperparameter experiments, and quantitative image-quality evaluation.

14. References

Foster, D. (2023). Generative deep learning: Teaching machines to paint, write, compose, and play (2nd ed.). O'Reilly Media.

Foster's companion GitHub repository:
https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition

Original DCGAN notebook:
https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan/01_dcgan

LEGO Brick Images dataset:
https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images

15. AI Use Disclosure

ChatGPT was used to assist with adapting the Jupyter notebook, organizing documentation, troubleshooting implementation issues, and preparing explanations of the experimental results.

The reported numerical results are based on the recorded notebook execution. Interpretations of image quality and training stability should be verified against the saved experimental evidence.

Old Dominion University — Generative AI, Module 3
Project: DCGAN LEGO Bricks Image Generation
Repository: https://github.com/lwalk051/Module3_DCGAN_LEGO
