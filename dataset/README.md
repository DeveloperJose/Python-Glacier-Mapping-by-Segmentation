# Dataset

## Sources of Data
The labels come from "Clean Ice and Debris covered glaciers of HKH Region" https://rds.icimod.org/metadata/c6a59a04-e6f7-4bf6-a6a8-f1cd534a6b62 and the report which describes the creation of the dataset is (The Status of Glaciers in Hindu Kush Himalayan (HKH) Region)[https://lib.icimod.org/records/wt6cp-2bt35].

Specifically, in the report there are tables listing which images were used for which basins. The 41 scenes that cover the HKH region are

We change the Landsat 5 scenes with Landsat 7 scenes since they are so few in number and to keep all input data with the same number of bands and exact representations.

This is the exact process used but not fully described by ... et al and described in Baraka S, Akera B, Aryal B, Sherpa T, Shresta F, Ortiz A, Sankaran K, Lavista Ferres J, Matin M, Bengio Y. 2020. Machine Learning for Glacier Monitoring in the Hindu Kush Himalaya. NeurIPS 2020 Climate Change AI Workshop (2020). lila.science/datasets/hkh-glacier-mapping/ , and one of the dataset authors continued this work in Aryal, B. ., Miles, K. E., Vargas Zesati, S. A., & Fuentes, O. (2023). Boundary Aware U-Net for Glacier Segmentation. Proceedings of the Northern Lights Deep Learning Workshop, 4. https://doi.org/10.7557/18.6789 https://septentrio.uit.no/index.php/nldl/article/view/6789

In those datasets there were two issues that we found through visual inspection, one in the gapfill implementation, and another in the selection of scenes from Landsat 7 used to replace the Landsat 5 scenes in the report. Details to follow.

Gapfill and NSPI information to be added here.

Velocity data information to be added here.

Preprocessing pipeline information to be added here.
