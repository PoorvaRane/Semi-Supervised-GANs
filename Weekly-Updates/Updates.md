# Semi- Supervised GANs for Data Efficient Classification - Weekly Updates

**Team:** Poorva Rane, Goutam Nair, Anushree Prasanna Kumar

**Project Description:** We are working on using GANs to generate images of tissues and cells. We then leverage the learned representations and extend the discriminator to classify the tissue as cancerous and non-cancerous.

## Weekly Updates

### Sep 12, 2018 - Sep 19, 2018 (09/12 - 09/19)
Member | Tasks 
------ | ---------------
Goutam | Explore how condiational GANs can be leveraged for problem statement. Implement a simple CycleGAN in PyTroch on the dataset being used in the CycleGAN paper (maps dataset). Evaluate if using the least squares GAN improves performance.
Poorva | Perform literature survey on the various applications of semi-supervised GANs. Implement a minial Semi-supervised GAN and get it running on the MNIST dataset. Evaluate the performance of the GAN and implement different distance metrics on the GAN.
Anushree | Go through the approaches mentioned in the Camelyon17 Challenge and their basic implementations. Implement progressive growing og GANs for improved image quality generation and explore different multiclass classification techniques using GANs.


### Sep 5, 2018 - Sep 12, 2018 (09/05 - 09/12)

Member | Tasks 
------ | ---------------
Goutam | Read papers on the variants of GANs in order to understand their implementation details. Further performed literature survey on state of the art on semi-supervised learning with GANs and CycleGANs
Poorva | Did a literature survey on exisitng techniques used for tissue classification. Analyzed and identified various distance metrics that can be used between the generated and trained probability distributions for the task. Performed literature survey on GAN architectures and improved techniques for training GANs
Anushree | Explored publicly available annotated datasets for the project (<https://cancergenome.nih.gov/>). Performed literature survey on DCGANS, state of the art on semi-supervised learning with GANs and work on Progressive growing of GANs to generate high resolution images.
