# Body Transformer: Leveraging Robot Embodiment for Policy Learning

## Paper Review
[[BoT 논문 리뷰]Body Transformer: Leveraging Robot Embodiment for Policy Learning
](https://velog.io/@tm011899/Body-Transformer-Leveraging-Robot-Embodiment-for-Policy-Learning)

## Toy Example Implementation

![](https://velog.velcdn.com/images/tm011899/post/d97b60b9-5da7-4601-ba82-31f16a4969a6/image.png)
## Implementation

### door-expert-v2
현재 구현은 Door open 구현에 초첨이 맞추어져 있습니다.

Third Implementation commit과 다르게 논문과 동일한 *Dataset* 사용합니다.

model.py를 직접 구현하며 직접적으로 파라미터들이 삽입되어 있어 이해하기가 편합니다.

직접 구현으로 논문과 다르게 Attention map 또한 보실 수 있습니다.

코드를 보다 직관적으로 이해할 수 있게 작성하였습니다.

예시 [Video](https://wandb.ai/taemin6697/adroit-bc/runs/boi8j4ht?nw=nwusertm011899)에서 직접 학습 과정과 Video를 보실 수 있습니다.

## Requirements

```
absl-py==2.1.0
gymnasium-robotics==1.2.4
imageio==2.34.0
imageio-ffmpeg==0.4.9
minari==0.4.3
-e git+https://github.com/microsoft/MoCapAct.git@8f01663a0da1c5f1fa941417ba4f3ce71c987b18#egg=mocapact
moviepy==1.0.3
numpy==1.26.4
pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
tqdm==4.66.2
wandb==0.16.3
```

윈도우의 경우 os.environ['MUJOCO_GL'] = 'egl'를 'glfw'로 코드 혹은 라이브러리 단에서 수정하여야합니다.


<br>
<br>

#### Toy robot(이 부분은 fix implementations commit에서 지워진 파일을 보시면됩니다.)
![](https://velog.velcdn.com/images/tm011899/post/98fe346c-9b64-4041-a91a-a4875521ab9b/image.jpg)
팔과 다리 : 4개의 센서 데이터를 입력으로 받습니다.

몸통 : 1개의 센서 데이터를 입력으로 받습니다.

#### BoT Model

#### BoT Tokenizer
센서 데이터를 입력으로 받아 continuous 토큰으로 만들어 Encoder에 입력된다.
#### BoT Encoder
마스킹 처리와 Attention map, 전체 Transformer와 마스킹 구현되어있습니다.

선택적으로 BoT-Hard와 BoT-Mix가 구현되어있습니다.
#### BoT DeTokenizer
각각의 토큰 마다의 Linear Detokenizer가 구현되어있습니다.

