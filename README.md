# Image Generation from caption using Stack-GAN
Stack GAN was introduced by Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas in their publication [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242v1.pdf). 
([Github:](https://github.com/hanzhanggit/StackGAN))

* It takes sentence as input and generates an image conditioned on the text description.
* The model consists of two-stage GAN: Stage-I, Stage-II
* Stage-I produces low-resolution image and Stage-II corrects the defects of stage-I resultant image yielding a high-resolution image.
* The input text is converted into text embedding using [Skipthought vector](https://arxiv.org/abs/1506.06726)

## Implementation details

### Dependencies:
*	Python 3.6.0
*	Tensorflow 1.7.0
*	Cuda v9.0 (For GPU computations)


### Extra packages installed:
*	prettytensor
*	progressbar
*	easydict
*	pandas
*	torchfile

#### Note: Running the code on CPU takes considerably long time as mentioned in the paper to train stage-I and Stage-II for 600 epochs would take less time running on GPU.
