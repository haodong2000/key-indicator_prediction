## Prediction of Key Performance Indicators of Industrial Processes Based on Deep Learning

&copy; Haodong Li

- [Configuration & Usage](#part_0) 
- [Data Analysis & Pre-process](#part_1) 
- [LSTM-based Model Design](#part_2) 
- [Transformer-based Model Design](#part_3) 
- [CL-based Model Design](#part_4) 
- [Experiments](#part_5)

<a name="part_0"/>

### Configuration & Usage

- Core packages

```
tensorflow 2.9.1
torch 1.12.0
```

- Usage

```
$ python run.py --delay [number of time steps]
```

<a name="part_1"/>

#### Data Analysis & Pre-process

- 115 dimensions & 29070 time steps
- 6 key indicators & 109 auxiliary indicators
- Due to relevant agreement, the data is kept confidential. Please contact Prof. Zhang via xinminzhang@zju.edu.cn if needed.

| Hot metal Si (01)| Hot metal S (53)|Hot metal Mn (54)|
|:---:|:---:|:---:|
|![001_╚▄πèSi](https://user-images.githubusercontent.com/67775090/236666136-cea58945-164c-4d6e-b719-cfa2d785783a.png) | ![053_╚▄πèS](https://user-images.githubusercontent.com/67775090/236666139-efd36878-35a7-4696-94b1-cda60b6e5359.png) | ![054_╚▄πèMn](https://user-images.githubusercontent.com/67775090/236666142-8cf6fd88-2f55-490e-900a-b4e52b932c6c.png) |
| ![fft_01](https://user-images.githubusercontent.com/67775090/236666220-4b702032-71dd-4fc9-ba6f-994ae297bcb3.png) | ![fft_53](https://user-images.githubusercontent.com/67775090/236666225-20906cd1-ee29-49ae-8426-9465f802c4d8.png) | ![fft_54](https://user-images.githubusercontent.com/67775090/236666227-7aa62db7-6db2-4d10-9cb3-5b8d0724f059.png) |

|Hot metal P (55)|Hot metal C (56)|Hot metal Ti (57)|
|:---:|:---:|:---:|
| ![055_╚▄πèP](https://user-images.githubusercontent.com/67775090/236666146-b8b0e3d0-52d5-49f9-b45b-3171fbd984c8.png) | ![056_╚▄πèC╖╓╬÷éÄ](https://user-images.githubusercontent.com/67775090/236666149-0469cf40-e8fe-4fd3-944d-b8a73ec106c6.png) | ![057_╚▄πèTi](https://user-images.githubusercontent.com/67775090/236666151-a32cf8ec-d77b-45cd-8e8a-cf83b5755a65.png) |
| ![fft_55](https://user-images.githubusercontent.com/67775090/236666228-9cda92d2-773e-42c5-87e9-cee944fb4fdf.png) | ![fft_56](https://user-images.githubusercontent.com/67775090/236666230-b807239d-0984-4b46-9a29-1014c2996041.png) | ![fft_57](https://user-images.githubusercontent.com/67775090/236666233-c3125121-e9b6-4f39-beee-7c88d83abb9d.png) |

- First row: data characteristics & correlation distribution among variables 
  - The upper part of the graph is the distribution of the data, the yellow is the original data, and the red is the data after mean smoothing;
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

- Results on 6 key indicators prediction (`time_step` = 0, only values on the next time step is predicted)

| Model               | `RMSE` Loss                    | `R2 Score` Accuracy          |
|:-------------------:|:----------------------------:|:--------------------------:|
| `CNN_LSTM`            | 0.047456759959459305±3.58e-3 | 0.9457983374595642±6.59e-3 |
| CL-based Model           | 0.053124434375849953±8.71e-4 | 0.9473923005326821±2.04e-3 |
| Transformer-based Model  | 0.05200807997651065±1.50e-3  | 0.9524179648107557±2.88e-3 |
| `EfficientNetV2_LSTM` | 0.043179091066122055±2.08e-3 | 0.9531577825546265±3.95e-3 |
| `ResNet_LSTM`         | 0.04068516939878464±2.65e-4  | 0.9558192491531372±3.10e-4 |
| **`Simple_LSTM`**        | **0.0395905040204525±1.43e-4**   | **0.9569856524467468±1.75e-4** |

- Accuracy results on 6 key indicators prediction in multi time steps (1\~20)

| `R2 Score` Accuracy     | 1        | 2        | 3        | 4        | 5        | 6        | 7        | 8        | 9        | 10       | 11       | 12       | 13       | 14       | 15       | 16       | 17       | 18       | 19       | 20       |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| `CNN_LSTM`            | 0.908867 | 0.886703 | 0.899269 | 0.892492 | 0.887989 | 0.891364 | 0.887657 | 0.890357 | 0.852183 | 0.878856 | 0.876410 | 0.882148 | 0.876686 | 0.867974 | 0.873978 | 0.860266 | 0.865002 | 0.864405 | 0.873603 | 0.872325 |
| `ResNet_LSTM`         | 0.929079 | 0.913988 | 0.907212 | 0.900088 | 0.896830 | 0.889898 | 0.888573 | 0.887636 | 0.883562 | 0.879436 | 0.875042 | 0.879181 | 0.879674 | 0.873856 | 0.875850 | 0.876290 | 0.867004 | 0.870304 | 0.869956 | 0.871067 |
| Transformer-based Model  | 0.924479 | 0.905764 | 0.890927 | 0.887882 | 0.879788 | 0.877154 | 0.867402 | 0.868933 | 0.862485 | 0.859569 | 0.853113 | 0.847314 | 0.847392 | 0.846858 | 0.846743 | 0.838788 | 0.840361 | 0.835966 | 0.836579 | 0.830244 |
| `EfficientNetV2_LSTM` | 0.926505 | 0.909907 | 0.865817 | 0.846057 | 0.893718 | 0.879946 | 0.888579 | 0.885395 | 0.888055 | 0.877716 | 0.880843 | 0.880393 | 0.868341 | 0.878926 | 0.862367 | 0.871897 | 0.871534 | 0.878088 | 0.874382 | 0.864462 |
| **`Simple_LSTM`**         | **0.932606** | **0.917350**| 0.908768 | 0.904000 | 0.899892 | 0.896070 | 0.893260 | 0.891410 | 0.887690 | 0.883684 | 0.883256 | 0.881757 | 0.882073 | 0.878427 | 0.876781 | 0.877600 | 0.874004 | 0.872749 | 0.874959 | 0.871901 |
| **CL-based Model**           | 0.926022 | 0.915719 |**0.914630** | **0.911664** | **0.907598** | **0.904705** | **0.905521**  |  **0.909347** |**0.908416** | **0.903857** | **0.905245** | **0.903743** | **0.903044** | **0.899642** | **0.898448** | **0.902504** | **0.903457** | **0.900310** | **0.897716** | **0.890362**  |

- Loss results on 6 key indicators prediction in multi time steps (1\~20)

| `RMSE` Loss     | 1        | 2        | 3        | 4        | 5        | 6        | 7        | 8        | 9        | 10       | 11       | 12       | 13       | 14       | 15       | 16       | 17       | 18       | 19       | 20       |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| `CNN_LSTM`            | 0.062719 | 0.069669 | 0.065025 | 0.068296 | 0.070231 | 0.068054 | 0.069016 | 0.067706 | 0.081297 | 0.073171 | 0.073568 | 0.071251 | 0.073282 | 0.075785 | 0.072892 | 0.079644 | 0.078324 | 0.077303 | 0.074049 | 0.074703 |
| `ResNet_LSTM`         | 0.052612 | 0.058794 | 0.061268 | 0.063988 | 0.065415 | 0.068184 | 0.068628 | 0.069234 | 0.070170 | 0.072334 | 0.073452 | 0.071944 | 0.072439 | 0.074497 | 0.073929 | 0.073369 | 0.076450 | 0.075966 | 0.075809 | 0.075544 |
| Transformer-based Model  | 0.065352 | 0.073093 | 0.078722 | 0.079686 | 0.082578 | 0.083615 | 0.086860 | 0.086253 | 0.088425 | 0.089364 | 0.091480 | 0.093046 | 0.092977 | 0.093226 | 0.093441 | 0.095786 | 0.095165 | 0.096494 | 0.096241 | 0.098178 |
| `EfficientNetV2_LSTM` | 0.054545 | 0.059823 | 0.077574 | 0.084122 | 0.065652 | 0.072862 | 0.068345 | 0.068981 | 0.068923 | 0.071301 | 0.071452 | 0.072011 | 0.075173 | 0.071815 | 0.077215 | 0.074327 | 0.075223 | 0.071996 | 0.073958 | 0.077426 |
| **`Simple_LSTM`**        | **0.050440** | **0.056559** | **0.060024** | **0.061961** | **0.063950** | **0.065591** | **0.066960** | **0.067253**| 0.068736 | 0.070218 | 0.070494 | 0.071119 | 0.070932 | 0.072555 | 0.073133 | 0.073189 | 0.074096 | 0.074467 | 0.073924 | 0.075296 |
| **CL-based Model**           | 0.062023 | 0.064925 | 0.065745 | 0.066262 | 0.067535 | 0.068989 | 0.068911 | 0.067742 |  **0.067886** | **0.069549** | **0.068565** | **0.069679** | **0.069927** | **0.069841** | **0.071422** | **0.070228** | **0.069699** | **0.070448** | **0.071425** | **0.073082** |

|Accuracy trend |Loss trend|
|:---:|:---:|
|![mts_acc](https://github.com/pandahehua/readme_images/assets/130850554/d0dbe757-74b9-49e6-9e45-c8ba8d90d089) |![mts_loss](https://github.com/pandahehua/readme_images/assets/130850554/196b073d-6da8-4a1a-97bc-2896e6c9d66f)| 

- Results on 1 key indicators prediction (only Hot metal Si (01), `time_step` = 0)
- `EfficientNetV2_LSTM` requires the number of selected key variables must be divisible by 3

| Model             | `RMSE` Loss            | `R2 Score` Accuracy  |
|:-----------------:|:--------------------:|:------------------:|
| `CNN_LSTM`          | 0.0405864343047142   | 0.8959161043167114 |
| `Simple_LSTM`       | 0.039217736572027206 | 0.901961088180542  |
| `ResNet_LSTM`       | 0.03927604481577873  | 0.9023586511611938 |
| **[Baseline](https://ieeexplore.ieee.org/document/9882520/)**          | **0.03596**              | 0.9334             |
| **CL-based Model**          | 0.036562133335719144 | **0.9352364961074216** |
| **Transformer-based Model** | **0.009228735077959387** | **0.9901837524193436** |

- Training log & prediction result visualization (take `EfficientNetV2_LSTM` with 6 key indicators scenario with `time_step` = 0 for example)

<img src="https://user-images.githubusercontent.com/67775090/236668233-bd5e0be4-438b-4c86-8fc2-bcf62a2aa78e.png" width="400">

| Hot metal Si (01)| Hot metal S (53)|Hot metal Mn (54)|
|:---:|:---:|:---:|
| ![simple_lstm_pred_1](https://user-images.githubusercontent.com/67775090/236668238-abeaab26-0b7c-4ddf-b941-5d1fd83a220d.png) | ![simple_lstm_pred_2](https://user-images.githubusercontent.com/67775090/236668247-4a6f541a-3d55-4c54-9bdb-f568bdb60e66.png)  | ![simple_lstm_pred_3](https://user-images.githubusercontent.com/67775090/236668258-fa150840-2d48-4210-8562-848728d532f0.png)  |

|Hot metal P (55)|Hot metal C (56)|Hot metal Ti (57)|
|:---:|:---:|:---:|
|![simple_lstm_pred_4](https://user-images.githubusercontent.com/67775090/236668264-03852b4c-4d9d-46ee-86da-364bf26f68c0.png)  | ![simple_lstm_pred_5](https://user-images.githubusercontent.com/67775090/236668268-f2b6f057-134d-4067-9e4a-940932da2408.png)  |![simple_lstm_pred_6](https://user-images.githubusercontent.com/67775090/236668272-d573d292-bd1c-4acc-80ae-f385e9d6fc46.png)  |

