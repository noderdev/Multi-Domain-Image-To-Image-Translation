Multi-Domain Image-to-Image Translation
=
Image to Image translation is the task of changing a particular aspect of an image from one domain to another. Recent studies have shown tremendous success in these areas, however there are scalability and robustness issues when dealing with more than 2 domains. For every pair of domains, there needs to be a separate independent model. StarGAN is a scalable and novel approach that addresses this issue by allowing us to develop only a single unified model which can perform the task of multi domain image to image translation.

Architecture
 =
The architecture of StarGAN comprises of 2 components, a Discriminator and a Generator. Discriminator learns to discriminate between real and fake images while the generator tries to generate fake images which are as real as possible. The 2 components work together to reach a state where the discriminator is no longer able to discriminate between real and fake images.

  <img width="800" alt="Screen Shot 2024-03-16 at 2 36 15 PM" src="https://github.com/noderdev/Multi-Domain-Image-To-Image-Translation/assets/29915581/eb533a40-0a78-476a-ab90-7a2f12dd0062">

**Defining Losses** <br />
The StarGAN network comprises of 3 types of losses which it tries to minimize to achieve the task of multi domain image to image translations.
1. Adversarial Loss: The adversarial loss is the core component of any GAN network. It involves a competitive balance between the discriminator and generator to make the generated images indistinguishable from real images. The generator tries to minimize the objective whereas the discriminator tried to maximize it.

      <img width="407" alt="Screen Shot 2024-03-16 at 2 41 04 PM" src="https://github.com/noderdev/Multi-Domain-Image-To-Image-Translation/assets/29915581/9e845438-a421-436c-b25a-a25420013f39">

2. Domain Classification Loss: The domain classification loss deals with the task of correctly translating the input image to the target domain. The domain classification loss for discriminator is given by: -
  
      <img width="287" alt="Screen Shot 2024-03-16 at 2 44 10 PM" src="https://github.com/noderdev/Multi-Domain-Image-To-Image-Translation/assets/29915581/09a79bf8-8083-49cd-8501-e9e56ee85118">

      The terms Dcls(C’/x) refers to the probability distribution over domain label computed by D. By minimizing this loss, D learns to correctly classify real images to their original Domain. The domain classification loss for Generator is given by: -

      <img width="287" alt="Screen Shot 2024-03-16 at 2 45 22 PM" src="https://github.com/noderdev/Multi-Domain-Image-To-Image-Translation/assets/29915581/b920af6b-e9fc-40f4-972c-2b63623f3ec4">

      Generator learns to generate images that can be classified as target domain c by minimizing this loss.
   
4. Reconstruction Loss: To make sure that generator only focuses on the domain related part while generating fake images, a reconstruction loss is introduced which the generator tries to minimize.

      <img width="287" alt="Screen Shot 2024-03-16 at 2 46 21 PM" src="https://github.com/noderdev/Multi-Domain-Image-To-Image-Translation/assets/29915581/0ea61b25-ede7-4044-97e5-16256c163cac">

Results on Weather Image Translation
=
We then performed image-to-image translation on the weather images dataset. The following are some of the translations: -

<img width="704" alt="Screen Shot 2024-03-16 at 2 51 44 PM" src="https://github.com/noderdev/Multi-Domain-Image-To-Image-Translation/assets/29915581/259a3e4c-cc20-4688-b668-98ea5fa33186">

Requirements
==
 Make sure to have tensorflow >= 2.0 installed and pytorch >= 2.0 installed as well.


Train
==

```sh
python main.py --mode train --dataset RaFD --image_size 128 --c_dim 4 --rafd_image_dir <path to train folder>
--sample_dir stargan_custom/samples --log_dir stargan_custom/logs --model_save_dir stargan_custom/models
--result_dir stargan_custom/results --num_iters 200000
 ```

To Generate 
==
```sh
python main.py --mode test --dataset RaFD --image_size 128 --c_dim 4 --rafd_image_dir <path to test folder>
--sample_dir stargan_custom/samples --log_dir stargan_custom/logs - -model_save_dir stargan_custom/models
--result_dir stargan_custom/results --test_iters 200000
```

Acknowledgments
==
The code is build upon [StarGAN](https://github.com/yunjey/stargan) implementation.
