# Body Transformer: Leveraging Robot Embodiment for Policy Learning

## Paper Review
[[BoT 논문 리뷰]Body Transformer: Leveraging Robot Embodiment for Policy Learning
](https://velog.io/@tm011899/Body-Transformer-Leveraging-Robot-Embodiment-for-Policy-Learning)

## Toy Example Implementation

![](https://velog.velcdn.com/images/tm011899/post/d97b60b9-5da7-4601-ba82-31f16a4969a6/image.png)
## Implementation

### Toy robot
![](https://velog.velcdn.com/images/tm011899/post/98fe346c-9b64-4041-a91a-a4875521ab9b/image.jpg)
팔과 다리 : 4개의 센서 데이터를 입력으로 받습니다.

몸통 : 1개의 센서 데이터를 입력으로 받습니다.

### BoT Model

#### BoT Tokenizer
센서 데이터를 입력으로 받아 continuous 토큰으로 만들어 Encoder에 입력된다.
#### BoT Encoder
마스킹 처리와 Attention map, 전체 Transformer와 마스킹 구현되어있습니다.

선택적으로 BoT-Hard와 BoT-Mix가 구현되어있습니다.
#### BoT DeTokenizer
각각의 토큰 마다의 Linear Detokenizer가 구현되어있습니다.


<br>
현재 모방 학습만 구현되어있습니다.
