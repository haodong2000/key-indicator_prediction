## Prediction of Key Performance Indicators of Industrial Processes Based on Deep Learning

&copy; Haodong Li

- [Data Analysis & Pre-process](#part_1) 
- [LSTM-based Model Design](#part_2) 
- [Transformer-based Model Design](#part_3) 
- [CL-based Model Design](#part_4) 
- [Experiments](#part_5)

<a name="part_1"/>

#### Data Analysis & Pre-process

- 115 dimensions & 29070 time steps
- 6 key indicators & 109 auxiliary indicators

| Hot metal Si (01)| Hot metal S (53)|Hot metal Mn (54)|
|:---:|:---:|:---:|
|![001_╚▄πèSi](https://user-images.githubusercontent.com/67775090/236666136-cea58945-164c-4d6e-b719-cfa2d785783a.png) | ![053_╚▄πèS](https://user-images.githubusercontent.com/67775090/236666139-efd36878-35a7-4696-94b1-cda60b6e5359.png) | ![054_╚▄πèMn](https://user-images.githubusercontent.com/67775090/236666142-8cf6fd88-2f55-490e-900a-b4e52b932c6c.png) |
| ![fft_01](https://user-images.githubusercontent.com/67775090/236666220-4b702032-71dd-4fc9-ba6f-994ae297bcb3.png) | ![fft_53](https://user-images.githubusercontent.com/67775090/236666225-20906cd1-ee29-49ae-8426-9465f802c4d8.png) | ![fft_54](https://user-images.githubusercontent.com/67775090/236666227-7aa62db7-6db2-4d10-9cb3-5b8d0724f059.png) |

|Hot metal P (55)|Hot metal C (56)|Hot metal Ti (57)|
|:---:|:---:|:---:|
| ![055_╚▄πèP](https://user-images.githubusercontent.com/67775090/236666146-b8b0e3d0-52d5-49f9-b45b-3171fbd984c8.png) | ![056_╚▄πèC╖╓╬÷éÄ](https://user-images.githubusercontent.com/67775090/236666149-0469cf40-e8fe-4fd3-944d-b8a73ec106c6.png) | ![057_╚▄πèTi](https://user-images.githubusercontent.com/67775090/236666151-a32cf8ec-d77b-45cd-8e8a-cf83b5755a65.png) |
| ![fft_55](https://user-images.githubusercontent.com/67775090/236666228-9cda92d2-773e-42c5-87e9-cee944fb4fdf.png) | ![fft_56](https://user-images.githubusercontent.com/67775090/236666230-b807239d-0984-4b46-9a29-1014c2996041.png) | ![fft_57](https://user-images.githubusercontent.com/67775090/236666233-c3125121-e9b6-4f39-beee-7c88d83abb9d.png) |

- First row: data characteristics & correlation distribution among variables 
  - The upper part of the graph is the distribution line of the data, the yellow is the original data, and the red is the data after mean smoothing;
  - The lower half of the graph represents the distribution of correlation coefficients between this indicator and all 115 indicators.
- Second row: frequency domain distribution of the data
- Data pre-process: Max-Min Normalization & `numpy.nan_to_num`

<a name="part_2"/>

#### LSTM-based Model Design

|`Simple_LSTM`|`ResNet_LSTM`|
|:---:|:---:|
| ![simple_lstm_page-0001](https://user-images.githubusercontent.com/67775090/236666697-f60d837d-6c7d-4947-b506-e99f38a60151.jpg) | ![resnet_rnn_page-0001](https://user-images.githubusercontent.com/67775090/236666722-beb9ad23-637f-4a78-bbbe-4abc3dc5fd27.jpg)|

|`CNN_LSTM`|`EfficientNetV2_LSTM`: `tf.keras.applications.EfficientNetV2S` + LSTM|
|:---:|:---:|
| ![cnn_rnn_page-0001](https://user-images.githubusercontent.com/67775090/236666711-45f47700-4de7-4768-90a3-55b3b0bab9de.jpg)  | ![efficientnetv2_rnn_new_page-0001](https://user-images.githubusercontent.com/67775090/236666728-0cf44c38-385c-4754-a47b-798445a15c14.jpg) |

<a name="part_3"/>

#### Transformer-based Model Design

- Overview

<img src="https://user-images.githubusercontent.com/67775090/236666860-d896c799-1882-4ae5-8b7b-26b06758b0a1.jpg" width="400">

- Detailed architecture

|Encoder|Decoder|
|:---:|:---:|
| ![transformer_encoder_NEW_page-0002](https://user-images.githubusercontent.com/67775090/236666894-95b5f0ba-b382-4be1-a33e-c9c58c480b86.jpg) | ![transformer_decoder_NEW_page-0002](https://user-images.githubusercontent.com/67775090/236666889-5b7d862a-e36e-4ec1-8d0f-e7a83aa3c7cb.jpg) |

<a name="part_4"/>

#### CL-based Model Design

- Overview (CL means Continual Learning)

<img src="https://user-images.githubusercontent.com/67775090/236666950-413e37c1-060b-40dd-a211-b54416b876b4.jpg" width="400">

- Detailed architecture

|`FastNet_1`| `FastNet_2` & `SlowNet_2`|
|:---:|:---:|
| ![CL_fastnet_1_page-0001](https://user-images.githubusercontent.com/67775090/236666965-91c9c2bd-db2e-42de-a34a-de1dc7db63b5.jpg) | ![CL_2_page-0001](https://user-images.githubusercontent.com/67775090/236667017-51919fa3-b053-4b48-9f3a-3e49aa995cf7.jpg)  |

|`SlowNet_1`|`MLP_End`|
|:---:|:---:|
| ![CL_slownet_1_page-0002](https://user-images.githubusercontent.com/67775090/236667660-813a65e0-c267-4533-83f4-7a012a33d934.jpg) | ![CL_3_page-0002](https://user-images.githubusercontent.com/67775090/236667672-5f519c97-ce24-49ea-ba39-504eafc13be2.jpg)|

<a name="part_5"/>

#### Experiments

- Results on 6 key indicators prediction

| Model               | RMSE Loss                    | R^2 Score Accuracy          |
|:-------------------:|:----------------------------:|:--------------------------:|
| `CNN_LSTM`            | 0.047456759959459305±3.58e-3 | 0.9457983374595642±6.59e-3 |
| CL-based Model           | 0.053124434375849953±8.71e-4 | 0.9473923005326821±2.04e-3 |
| Transformer-based Model  | 0.05200807997651065±1.50e-3  | 0.9524179648107557±2.88e-3 |
| `EfficientNetV2_LSTM` | 0.043179091066122055±2.08e-3 | 0.9531577825546265±3.95e-3 |
| `ResNet_LSTM`         | 0.04068516939878464±2.65e-4  | 0.9558192491531372±3.10e-4 |
| `Simple_LSTM`         | 0.0395905040204525±1.43e-4   | 0.9569856524467468±1.75e-4 |

- Results on 1 key indicators prediction (only Hot metal Si (01))
- `EfficientNetV2_LSTM` requires the number of selected key variables must be divisible by 3

| Model             | RMSE Loss            | R2 Score Accuracy  |
|-------------------|----------------------|--------------------|
| `CNN_LSTM`          | 0.0405864343047142   | 0.8959161043167114 |
| `Simple_LSTM`       | 0.039217736572027206 | 0.901961088180542  |
| `ResNet_LSTM`       | 0.03927604481577873  | 0.9023586511611938 |
| [Baseline](https://ieeexplore.ieee.org/document/9882520/)          | 0.03596              | 0.9334             |
| CL-based Model          | 0.036562133335719144 | 0.9352364961074216 |
| Transformer-based Model | 0.009228735077959387 | 0.9901837524193436 |

- Training log & prediction result visualization (take `EfficientNetV2_LSTM` with 6 key indicators scenario for example)

<img src="https://user-images.githubusercontent.com/67775090/236668233-bd5e0be4-438b-4c86-8fc2-bcf62a2aa78e.png" width="600">

| Hot metal Si (01)| Hot metal S (53)|Hot metal Mn (54)|
|:---:|:---:|:---:|
| ![simple_lstm_pred_1](https://user-images.githubusercontent.com/67775090/236668238-abeaab26-0b7c-4ddf-b941-5d1fd83a220d.png) | ![simple_lstm_pred_2](https://user-images.githubusercontent.com/67775090/236668247-4a6f541a-3d55-4c54-9bdb-f568bdb60e66.png)  | ![simple_lstm_pred_3](https://user-images.githubusercontent.com/67775090/236668258-fa150840-2d48-4210-8562-848728d532f0.png)  |

|Hot metal P (55)|Hot metal C (56)|Hot metal Ti (57)|
|:---:|:---:|:---:|
|![simple_lstm_pred_4](https://user-images.githubusercontent.com/67775090/236668264-03852b4c-4d9d-46ee-86da-364bf26f68c0.png)  | ![simple_lstm_pred_5](https://user-images.githubusercontent.com/67775090/236668268-f2b6f057-134d-4067-9e4a-940932da2408.png)  |![simple_lstm_pred_6](https://user-images.githubusercontent.com/67775090/236668272-d573d292-bd1c-4acc-80ae-f385e9d6fc46.png)  |

