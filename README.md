# PCA-CNN Hybrid Approach for Hyperspectral Pansharpening

[PCA-CNN Hybrid Approach for Hyperspectral Pansharpening](https://ieeexplore.ieee.org/document/10288481) introduces PCA-Z-PNN, a simple yet effective method for Hyperspectral Pansharpening that combines the strengths of classical PCA decomposition and the potentials of Deep Learning. The strengths of this method are many:
- It is completely unsupervised, so it works at full resolution without any degradation procedure;
- It works from scratch, so it does not use any type of training;
- It works with images with an arbitrary number of bands.
The proposed method has been tested on three real hyperspectral images (PRISMA Dataset) and one synthetic dataset (Pavia University from ROSIS) and compared with several Pansharpening methods, both model-based and Deep-Learning.

## Cite PCA-Z-PNN

If you use PCA-Z-PNN in your research, please use the following BibTeX entry.

    @article{Guarino2023pca,
      title={PCA-CNN Hybrid Approach for Hyperspectral Pansharpening},
      author={Guarino, Giuseppe and Ciotola, Matteo and Vivone, Gemine and Poggi, Giovanni and Scarpa, Giuseppe},
      journal={IEEE Geoscience and Remote Sensing Letters},
      year={2023},
      publisher={IEEE}
    }

## Team members

*   Giuseppe Guarino (giuseppe.guarino2@unina.it);

*   Matteo Ciotola (matteo.ciotola@unina.it);

*   Gemine Vivone;

*   Giovanni Poggi;

*   Giuseppe Scarpa  (giuseppe.scarpa@uniparthenope.it).

## License

Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/giu-guarino/PCA-Z-PNN/LICENSE.txt)
(included in this package)

## Prerequisites

All the functions and scripts were tested on Windows and Ubuntu O.S., with these constrains:

*   Python 3.10.10
*   PyTorch 2.0.0
*   Cuda 11.7 or 11.8 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

*   Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads)
*   Create a folder in which save the algorithm
*   Download the algorithm and unzip it into the folder or, alternatively, from CLI:

<!---->

    git clone https://github.com/giu-guarino/PCA-Z-PNN

*   Create the virtual environment with the `pca_z_pnn_env.yaml`

<!---->

    conda env create -n pca_z_pnn_env -f pca_z_pnn_env.yaml

*   Activate the Conda Environment

<!---->

    conda activate pca_z_pnn_env

*   Test it

<!---->

    python test.py -i example/PRISMA_example.mat -o ./Output_folder/ 

## Usage

### Before to start

The easiest way for testing this algorithm is to create a `.mat` file. It must contain:

*   `I_MS_LR`: Original Hyperspectral Stack in channel-last configuration (Dimensions: H x W x B);
*   `I_MS`: Upsampled version of original Hyperspectral Stack in channel-last configuration (Dimensions: HR x WR x B);
*   `I_PAN`: Original Panchromatic band, without the third dimension (Dimensions: HR x WR).
*   `Wavelengths`: Array of wavelengths (Dimensions: B x 1)

where R is the ratio of the sensor.

Please refer to `--help` for more details.

### Testing

This network works completely from scratch. Therefore, you can test it right away providing as input your dataset:

    python test.py -i path/to/file.mat

Several options are possible. Please refer to the parser help for more details:

    python test.py -h

## Dataset

You can find the dataset used in this work at these links:

*   [PRISMA](https://openremotesensing.net/knowledgebase/panchromatic-and-hyperspectral-image-fusion-outcome-of-the-2022-whispers-hyperspectral-pansharpening-challenge/)

*   [ROSIS](https://paperswithcode.com/dataset/pavia-university)

In the experiments reported in the paper, Pavia University Dataset has been downgraded with a ratio 6 (same ratio of PRISMA sensor).
# PCA-Z-PNN
