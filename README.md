# Face recognition model

This is a face recognition model, which extracts a facial feature vector from an aligned facial image.

## Model Details

### Model Description

- **Model type:** Convolutional Neural Network
- **License:** 
Original Work:

MIT License

Copyright (c) 2022 Zhong Yaoyao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Changes in Code, Finetuning etc. are also under MIT License:

MIT License

Copyright (c) 2023 Martin Knoche

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

- **Finetuned from model:** [FaceTransformer](https://github.com/zhongyy/Face-Transformer) by [zhongyy](https://github.com/zhongyy)

### Model Sources

- **Repository:** [GitHub](github.com/martlgap/octuplet-loss)
- **Paper:** [IEEExplore](https://ieeexplore.ieee.org/document/10042669)

## Uses

Use the model to extract a facial feature vector from an arbitrary aligned facial image. You can then compare that vector to other facial feature vectors to decide for same or not same person. 

`input_image`-Variable

- Dimensions: 112x112x3
- Channels: Should be in RGB format
- Type: float
- Values: Between 0 and 255

`embedding`-Variable

- Dimension: 512
- Type: float

## Bias, Risks, and Limitations

The model was originally trained and also finetuned on the [MS1M](https://exposing.ai/msceleb/) dataset. Thus please be check the MS1M dataset for bias and risks.
