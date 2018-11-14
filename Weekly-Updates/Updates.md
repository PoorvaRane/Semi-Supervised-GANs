# Semi- Supervised GANs for Data Efficient Classification - Weekly Updates

**Team:** Poorva Rane, Goutam Nair, Anushree Prasanna Kumar

**Project Description:** We are working on using GANs to generate images of tissues and cells. We then leverage the learned representations and extend the discriminator to classify the tissue as cancerous and non-cancerous.

## Weekly Updates

### Nov 14, 2018 - Nov 21, 2018(11/14 - 11/21)
Member | Tasks
------ | ---------------
Goutam |  Write scripts to generate patches within a given annotated region. Implement module to identify all possible patches conatined within a given bounding polygon. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Poorva |  Implement SGAN with weight normalization and Guassian Noise inclusion on the CIFAR dataset. Use a similar architecure to perform classification on the TCGA dataset and fine-tune hyperparameters to improve classification accuracy. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Anushree | Implement a complement generator and techniques specified in the paper "Virtual Adversarial Training" and evaluate whether these techniques are able to improve the accuracy further. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>

### Nov 07, 2018 - Nov 14, 2018(11/07 - 11/14)
Member | Tasks
------ | ---------------
Goutam |  Tfrecords obtained from PathAI for the annotated regions. Implemented module to convert tfrecords into a readable format for pytorch and store dataset as hdf5 files<ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Poorva |  Implement WGAN-GP on a publicly available dataset. Identify and tune hyperparameters to improve accuracy. Implement techniques from the paper "Improved Semisupervised learning requires a bad generator"  <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Anushree | Implement WGAN-GP on a publicly available dataset. Identify and tune hyperparameters to improve accuracy. Implement techniques from "Improved Semisupervised learning requires a bad generator"  <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>

### Oct 31, 2018 - Nov 07, 2018(10/31 - 11/07)
Member | Tasks
------ | ---------------
Goutam |  Rengineering the annotation for the dataset for supervised and unsupervised samples<ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Poorva |  Implementing the optimal loss and SGAN architecture obtain from our experiments, in Pytorch from data obtain with tfrecords  <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Anushree | Training SGAN from the lung cancer dataset received from PathAI as tfrecords on the present architecture with hyperparameter tuning <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>


### Oct 24, 2018 - Oct 31, 2018(10/24 - 10/31)
Member | Tasks
------ | ---------------
Goutam | Tuning hyperparameters of the SGAN and working on different loss functions / optimizations techniques to achieve good performance on the cancer dataset. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Poorva |  Working on discriminator training in the SGAN architecture that improves classification accuracy using fewer annotated samples <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Anushree | Creating the annotated dataset using the bitmaps (given from PathAI) that denote the tissue region to remove empty patches from the tissue slide <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>

### Oct 10, 2018 - Oct 17, 2018(10/10 - 10/17)
Member | Tasks
------ | ---------------
Goutam | Training and tuning hyperparameters of the SGAN to obtain optimial performance over the dataset. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Poorva | Running SGAN with ConvTranspose operation over Upsampling to improve performance. Working on identifying factors that will affect the current SGAN architecture when employed on the actual cancer dataset  <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Anushree | We obtained the real dataset from PathAI last week. Working to create patches and their annotation to train the SGAN on the  curated dataset of cancer images <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>


### Oct 3, 2018 - Oct 10, 2018(10/03 - 10/10)
Member | Tasks
------ | ---------------
Goutam | Tune the percentage of supervised and unsupervised samples in the training phase and try to decrease this with optimal performance. Also, look at complement generator to improve the accuracy of semi-supervised classification on MNIST  <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Poorva | Experiment with different generator and discriminator architectures to improve the training of the Semi Supervised GANs. Also, examine the usage of Deconvolution layers over Upsampling in the SGAN architecture to improve performance. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Anushree | Incorporate new slide images from the LUSC and SKCM cancerous groups into the image-patching framework. Also, train the SGAN on the newly curated dataset of high resolution cancer images <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>

### Sep 26, 2018 - Oct 3, 2018 (09/26 - 10/03)
Member | Tasks
------ | ---------------
Goutam | From our discussion with our mentor at PathAI, we need to work to obtain the accuracy of real and fake generated images and the accuracy of the classification of images on MNIST data with the Semi-supervised GAN. Also obtain metric to get the percentage of supervised samples being used to train and try to decrease this with optimal performance   <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Poorva | Explore different architectures that improve the training of the Semi Supervised GANs. Implement improved GAN training techniques such as minibatch discrimination on the existing SGAN. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Anushree | Obtain the dataset from the CAMELYON 2017 dataset. Implement the framework to get patch level label annotation from the lesion level label annotation. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>


### Sep 19, 2018 - Sep 26, 2018 (09/19 - 09/26)
Member | Tasks
------ | ---------------
 Goutam | Implement a semi-supervised GAN on the MNIST Dataset and explore Least squares GAN for as a way of stabilizing the GAN training. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
 Poorva | Explore different methods to improve training of GANs on the standard DCGAN such as Mini-batch discrimination and virtual batch normalization. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
 Anushree | Improve the performance of the DCGAN by implementing techniques such as historical averaging and one-sided label smoothing to improve GAN training. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>


### Sep 12, 2018 - Sep 19, 2018 (09/12 - 09/19)
Member | Tasks 
------ | ---------------
Goutam | Explore how condiational GANs can be leveraged for problem statement. Implement a simple CycleGAN in PyTorch on the dataset being used in the CycleGAN paper (maps dataset). <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Poorva | Perform literature survey on the various applications of semi-supervised GANs. Implement a Vanilla GAN and get it running on the MNIST dataset. Evaluate the performance of the GAN and implement different distance metrics on the GAN. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>
Anushree | Go through the approaches mentioned in the Camelyon17 Challenge and their basic implementations. Implement DCGAN and evalute performance on MNIST Dataset. <ul><li>Status of Previous Week's Tasks - Completed.</li></ul>


### Sep 5, 2018 - Sep 12, 2018 (09/05 - 09/12)

Member | Tasks 
------ | ---------------
Goutam | Read papers on the variants of GANs in order to understand their implementation details. Further performed literature survey on state of the art on semi-supervised learning with GANs and CycleGANs
Poorva | Did a literature survey on exisitng techniques used for tissue classification. Analyzed and identified various distance metrics that can be used between the generated and trained probability distributions for the task. Performed literature survey on GAN architectures and improved techniques for training GANs
Anushree | Explored publicly available annotated datasets for the project (<https://cancergenome.nih.gov/>). Performed literature survey on DCGANS, state of the art on semi-supervised learning with GANs and work on Progressive growing of GANs to generate high resolution images.
