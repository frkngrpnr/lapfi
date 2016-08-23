% This file reads the extracted features, optimizes the model and produces
% the test set estimations.
% For the full pipeline, see main.m

% In order to include the validation set labels, you need to have
% gt_val.mat in data folder, which can be obtained like this:
% gt_val = LAPFI_read_gt_csv('validation_gt.csv');
% and you need to save gt_val in a mat file with the same name, i.e.:
% save([opts.base_path bss 'data' bss 'gt_val.mat'], 'gt_val');


%% 1. Initialization
opts = struct;
opts.base_path = fileparts(mfilename('fullpath')); % determine full path of this directory
addpath(genpath(opts.base_path)); % add all subdirectories to MATLAB path
bss = '\'; % backslash symbol. '\' for Windows, '/' for Linux.
opts.bss = bss;
opts.align_faces = 0; % note that you need the IntraFace library to use the face alignment code.
opts.extract_features = 0; % if 1, features will be extracted (for this, you need VLFeat>=0.9.20 and MatConvNet>=1.0.beta18 installed). if 0, saved features will be loaded.
opts.video_path = [opts.base_path bss 'data' bss 'videos' bss 'raw'];
opts.alignment_path = [opts.base_path bss 'data' bss 'videos' bss 'aligned'];
opts.frame_features_path = [opts.base_path bss 'data' bss 'features' bss 'frame'];
load('gt_train.mat');
use_val = 0;
try
    load('gt_val.mat'); 
    use_val = 1;
    %disp('Validation ground truth exists.');
catch
    %disp('Validation ground truth does not exist.');
end

%% 2. Data preparation
%% 2.1. Face alignment
if opts.align_faces
    fprintf('%s.m: aligning faces..\n',mfilename);
    [alignment] = LAPFI_align_w_IntraFace(opts.video_path, opts.alignment_path);
end

if opts.extract_features
    fprintf('%s.m: extracting features..\n',mfilename);
    % Extract LGBP-TOP
    lgbptop = LAPFI_extract_video_TOP_features(opts.alignment_path, 'LGBPTOP');
    % Extract deep face features from VGG-Face fine tuned on FER-2013:
    load('vggfer.mat')
    LAPFI_extract_frame_features(opts.alignment_path, [opts.frame_features_path bss 'vggfer33'], 'CNN_VGGFER', 33, net);
    vggfer33fun = LAPFI_encode_frame_features([opts.frame_features_path bss 'vggfer33'], 'FUN');
    vggfer33fun = LAPFI_attach_labels(vggfer33fun);
    % Extract deep scene features from VGG-VeryDeep-19 network (available at http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat):
    net = load('imagenet-vgg-verydeep-19.mat');
    vd19 = LAPFI_extract_scene_features(opts.video_path, [opts.frame_features_path bss 'vd19'], 'CNN_VGGVD19', 39, net);
    vd19 = LAPFI_attach_labels(vd19);
else % load features
    if ~(exist('vggfer33fun','var') && exist('lgbptop', 'var') && exist('vd19', 'var'))
        disp('Loading features..');
        load('vggfer33fun'); % face feature
        load('lgbptop'); % face feature
        load('vd19'); % scene feature
        load('is13'); % audio feature
        
    end
end
%% 3. Optimize the models and estimate test labels
outpath_test = [opts.base_path bss 'data' bss 'output' bss 'test' bss];
example_file_test = [opts.base_path bss 'data' bss 'ex_pred_test.csv'];

fprintf('%s.m: validating LGBP-TOP model..\n',mfilename);
[output_test_lgbptop] = LAPFI_train_KFSI_then_estimateTest(lgbptop, outpath_test, example_file_test, use_val);
copyfile([outpath_test bss 'predictions.csv'], [outpath_test bss 'predictions_lgbptop.csv']);

fprintf('%s.m: validating CNN model..\n',mfilename);
[output_test_cnn] = LAPFI_train_KFSI_then_estimateTest(vggfer33fun, outpath_test, example_file_test, use_val);
copyfile([outpath_test bss 'predictions.csv'], [outpath_test bss 'predictions_cnn.csv']);

fset_face = fuse_features({vggfer33fun, lgbptop});
fset_scene = vd19;
fset_audio = is13;
clearvars vggfer33fun lgbptop vd19 is13

fprintf('%s.m: validating face model..\n',mfilename);
[output_test_face] = LAPFI_train_KFSI_then_estimateTest(fset_face, outpath_test, example_file_test, use_val);
copyfile([outpath_test bss 'predictions.csv'], [outpath_test bss 'predictions_face.csv']);
fprintf('%s.m: validating scene model..\n',mfilename);
[output_test_scene] = LAPFI_train_KFSI_then_estimateTest(fset_scene, outpath_test, example_file_test, use_val);
copyfile([outpath_test bss 'predictions.csv'], [outpath_test bss 'predictions_scene.csv']);
fprintf('%s.m: validating audio model..\n',mfilename);
[output_test_audio] = LAPFI_train_KFSI_then_estimateTest(fset_audio, outpath_test, example_file_test, use_val);
copyfile([outpath_test bss 'predictions.csv'], [outpath_test bss 'predictions_audio.csv']);
% score-level fusion:
pred = struct;
pred.labelnames = output_test_face.train_KFSI_output.labelnames;
%pred.pred_filename = fset_face.filename(find(fset_face.set==3));
pred.pred_filename = output_test_face.ordered.filename;
for li=1:5    
    pred.predset{li} = 0.75 * output_test_face.ordered.pred(:,li) + 0.25 * output_test_scene.ordered.pred(:,li);    
end
pred2 = pred; % sil
li = find(strcmp(pred.labelnames, 'ValueAgreeableness'));
pred.predset{li} = 0.504*output_test_cnn.ordered.pred(:,li) + 0.393*output_test_lgbptop.ordered.pred(:,li) + 0.095*output_test_scene.ordered.pred(:,li) + 0.008*output_test_audio.ordered.pred(:,li);
LAPFI_write_predictions_test(pred, outpath_test, example_file_test);

