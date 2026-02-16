clear all;clc;close all
return
%imageSize = [224 224 3];
imageSize = [224 224 1];
anchorBoxes=[24,29;
            16,18;
            35,48;
            51,84];
numClasses =1;
modelfile = 'trip_Ost_29.h5';
%layers = importKerasLayers(modelfile)
baseNetwork = importKerasNetwork(modelfile,'OutputLayerType','regression')
%baseNetwork = vgg16;
%featureLayer = 'relu5_3'; % 'activation_4' ReLU of triplet
featureLayer = 'activation_4';
lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,baseNetwork,featureLayer);
figure
plot(lgraph);
title('Network Architecture')

% onnxf = 'mobilenetv2_pytorch.onnx';
% lgraph = importONNXLayers(onnxf,'OutputLayerType','classification', ...
%     'ImportWeights',true);
% placeholderLayers = findPlaceholderLayers(lgraph);
% revLayers = string(vertcat(placeholderLayers.Name));
% lgraph = removeLayers(lgraph,revLayers)
% deepNetworkDesigner()% ¤â??±µ¦Z?¥X

