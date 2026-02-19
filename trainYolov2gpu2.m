clear all; clc; close all;warning off;
disp('Step 1: Started script');
% load trainingDataGpu2
disp('Step 2: Reading CSV 2');
trainingData = read_yolo_csv('tr2_fix.csv');
disp('Step 3: Reading CSV valid');
validData = read_yolo_csv('valid1_fix.csv');
disp('Step 4: CSVs read');
% trainingData1=trainingData;
% trainingData = trainingData(1:10:10000,:); % speed up
numClasses = width(trainingData)-1;
%gpuD=gpuDevice(1); 
disp('Step 5: Initializing GPU');
gpuD=gpuDevice(3); % Use OS GPU 2 (1050 Ti)
disp('Step 6: GPU Initialized');
imageSize = [224 224 3];
anchorBoxes=[24,29;
            16,18;
            35,48;
            51,84];
rng('shuffle');
iterCnt=0;
tic
format long
% folder='.\xFolder\';
% while 1        
%     flag=0;
%     listing = dir(folder);
%     L=length(listing);
%     if L>=3
%         for i=3:L
%             filename=listing(i).name;  
%             if strcmp(filename,'x2.mat')
%                 pause(5);
%                 ff=strcat(folder,filename);
%                 load(ff);            
%                 delete(ff);
%                 flag=1;
%                 break
%             end
%         end  
%         if flag==1
%             break
%         end
%     end
%     pause(5);
% end
% for gen=1:7:701 % mobilenetv2
%for gen=2:7:702 % resnet18
%for gen=3:7:703 % resnet50
%for gen=4:7:704 % resnet101
%for gen=5:7:705 % vgg16
%for gen=6:7:706 % vgg19
for gen=7:7:707 % googlenet
    clc
    iterCnt=iterCnt+1
    trainTime=toc/3600
    if iterCnt>=2
        minCost=min(cost)
    end
    folder='.\xFolder\';
    targetFile = fullfile(pwd, 'xFolder', 'x2.mat');
    disp(['[GPU2] Waiting for ' targetFile]);
    
    while ~exist(targetFile, 'file')
        pause(1);
    end
    
    disp(['[GPU2] Found ' targetFile '. Loading...']);
    pause(0.5); 
    try
        load(targetFile);
        disp(['[GPU2] Loaded ' targetFile]);
    catch e
        disp(['[GPU2] Error loading x2.mat: ' e.message]);
    end
    
    try
        delete(targetFile);
        disp(['[GPU2] Deleted ' targetFile]);
    catch e
        disp(['[GPU2] Error deleting x2.mat: ' e.message]);
    end
    if x2(1)<=1
        x2(1)=1;
    end
    if x2(1)>=32
        x2(1)=32;
    end
    if x2(2)<=1e-6
        x2(2)=1e-6;
    end
    if x2(2)>=1e-2
        x2(2)=1e-2;
    end    
    mBS=ceil(x2(1));
    lr=x2(2);
%     mBS=x2(iterCnt,1);
%     lr=x2(iterCnt,2);
    options = trainingOptions('sgdm', ...
        'VerboseFrequency', 50, ...
        'GradientThreshold', 1, ...
        'GradientThresholdMethod', 'l2norm', ...
        'ExecutionEnvironment','gpu', ...
        'MiniBatchSize', mBS, ....
        'InitialLearnRate',lr, ...
        'MaxEpochs',50,...
        'Shuffle','every-epoch',...
        'OutputFcn',@(info)stopIfLossOK(info,1e-1));
    
    % Use robust network loading with fallback
    try
        lgraph = get_lgraph_safe(gen, imageSize, numClasses, anchorBoxes);
    catch e
        disp(['CRITICAL ERROR: ' e.message]);
        quit; % detailed in log
    end

    [detector,info] = trainYOLOv2ObjectDetector(trainingData,lgraph,options);
    
    % --- Step 6: Competitive Random Model Exchange (User Requirement) ---
    % 1. Evaluate LOCAL model first (5-Fold CV)
    disp(['[GPU2] Evaluating LOCAL model with 5-Fold CV...']);
    c2 = evaluate_model_cost(detector, validData);
    
    % --- NaN/Inf Handling for Local ---
    if isnan(c2) || isinf(c2)
        disp('[WARN] Local Cost is NaN/Inf.');
        if iterCnt > 1
             c2 = cost(iterCnt-1); % Use history
        else
             c2 = 1000; % Max penalty
        end
        disp(['[WARN] Fallback Local Cost: ' num2str(c2)]);
    end
    
    % 2. Save local model for exchange
    if ~exist('.\exchange','dir')
        mkdir('.\exchange');
    end
    exchangeFile = sprintf('.\\exchange\\model_gpu2_iter%d.mat', iterCnt);
    save(exchangeFile, 'detector');
    disp(['[GPU2] Saved model for exchange: ' exchangeFile]);
    
    % 3. Wait for other GPUs (Synchronization Barrier)
    otherFile1 = sprintf('.\\exchange\\model_gpu1_iter%d.mat', iterCnt);
    otherFile3 = sprintf('.\\exchange\\model_gpu3_iter%d.mat', iterCnt);
    
    disp('[GPU2] Waiting for model exchange...');
    while (~exist(otherFile1, 'file') || ~exist(otherFile3, 'file'))
        pause(1);
    end
    
    % 4. Select Peer to Compare
    exchangeCandidates = {otherFile1, otherFile3};
    chosenIdx = randi(2);
    peerFile = exchangeCandidates{chosenIdx};
    
    disp(['[GPU2] Comparing with Peer: ' peerFile]);
    loaded = load(peerFile, 'detector');
    peer_detector = loaded.detector;
    
    % 5. Evaluate PEER (5-Fold CV)
    c2_peer = evaluate_model_cost(peer_detector, validData);
    
    % --- NaN/Inf Handling for Peer ---
    if isnan(c2_peer) || isinf(c2_peer)
        c2_peer = 1000; % Peer is broken
    end
    
    % 6. Competitive Swap
    if c2_peer < c2
        disp(['[GPU2] Peer is BETTER (' num2str(c2_peer) ' < ' num2str(c2) '). SWAPPING.']);
        detector = peer_detector;
        c2 = c2_peer;
    else
        disp(['[GPU2] Peer is WORSE or EQUAL (' num2str(c2_peer) ' >= ' num2str(c2) '). KEEPING LOCAL.']);
    end
    % ---------------------------------------------------------

    save('.\cFolder\c2.mat','c2');
    
    % --- Save Best Model for this Iteration (User Request) ---
    if ~exist('.\best_models','dir')
        mkdir('.\best_models');
    end
    bestModelFile = sprintf('.\\best_models\\best_model_gpu2_iter%d.mat', iterCnt);
    save(bestModelFile, 'detector');
    disp(['[GPU2] Saved BEST model (Post-Exchange) to: ' bestModelFile]);
    % ---------------------------------------------------------

    th1(iterCnt)=mBS;
    th2(iterCnt)=lr; 
    cost(iterCnt)=c2;
    
    if mod(gen,7)==1
        save(strcat('.\th1_th2_cost_yolov2mobilenetv2gpu2_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==2
        save(strcat('.\th1_th2_cost_yolov2resnet18gpu2_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end

    if mod(gen,7)==3
        save(strcat('.\th1_th2_cost_yolov2resnet50gpu2_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==4
        save(strcat('.\th1_th2_cost_yolov2resnet101gpu2_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==5
        save(strcat('.\th1_th2_cost_yolov2vgg16gpu2_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==6
        save(strcat('.\th1_th2_cost_yolov2vgg19gpu2_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==0
        save(strcat('.\th1_th2_cost_yolov2googlenetGpu2_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end    
    
    if c2<=1.2
        break
    end
end  
disp('Done');
reset(gpuD);

% --- Helper Function for 5-Fold CV ---
function cost = evaluate_model_cost(detector, validData)
    numFolds = 5;
    numRows = height(validData);
    
    foldIndices = repmat(1:numFolds, 1, ceil(numRows/numFolds));
    foldIndices = foldIndices(1:numRows);
    foldIndices = foldIndices(randperm(numRows)); % Shuffle
    
    p_folds = zeros(numFolds,1);
    r_folds = zeros(numFolds,1);
    
    for k = 1:numFolds
        testIdx = (foldIndices == k)';
        testTbl = validData(testIdx, :);
        [p, r] = evaluate_fold(detector, testTbl);
        p_folds(k) = p;
        r_folds(k) = r;
    end
    precision = mean(p_folds);
    recall = mean(r_folds);
    
    cost = (1/precision + 1/recall)/2;
end
