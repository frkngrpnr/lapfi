This folder contains the code for our entry in the Apparent Personality Analysis and First Impressions Challenge in ChaLearn Looking at People Challenge and Workshop @ICPR 2016.

%% Notes
1) This code runs in MATLAB, and written in a Windows system, if you want to run it on a Linux-based system instead, please change line 17 of main.m accordingly, i.e. bss = '/';

2) Currently there are no validation labels given, so the code produces the test set estimations using only the training set. If you want to include the validation labels, you just need to read and store them as follows (after running main.m once):
%gt_val = LAPFI_read_gt_csv('validation_gt.csv');
%save([opts.base_path bss 'data' bss 'gt_val.mat'], 'gt_val');
Then the code will try to load('gt_val.mat'), and if succeeds, will use the whole development set to generate the test predictions.

3) To avoid conflict of interest, we didn't include third party tools and pre-trained models used for face detection and (partly) in feature extraction. We, however, indicated the necessary pointers and links for these external resources
For face alignment, you need the IntraFace library. For feature extraction, you need VLFeat (available at http://www.vlfeat.org/download.html), MatConvNet (available at http://www.vlfeat.org/matconvnet/) and OpenSmile (available at http://audeering.com/research/opensmile/) libraries installed. For audio feature extraction, we use the IS13_ComParE.conf file.

The test set predictions will be created in ./data/output/test


