clear all; clc; close all;warning off;
disp('Step 1: Started script');
% load trainingDataGpu1
disp('Step 2: Reading CSV 1');
trainingData = read_yolo_csv('tr1_fix.csv');
disp('Step 3: Reading CSV 2');
validData = read_yolo_csv('valid1_fix.csv');
disp('Step 4: CSVs read');
% trainingData1=trainingData;
% trainingData = trainingData(1:10:10000,:); % speed up
numClasses = width(trainingData)-1;
%gpuD=gpuDevice(1); % gpu1: 2080ti
disp('Step 5: Initializing GPU');
gpuD=gpuDevice(2); % Use OS GPU 1 (1050 Ti)
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
%             if strcmp(filename,'x1.mat')
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
    targetFile = fullfile(pwd, 'xFolder', 'x1.mat');
    disp(['[GPU1] Waiting for ' targetFile]);
    
    while ~exist(targetFile, 'file')
        pause(1);
    end
    
    disp(['[GPU1] Found ' targetFile '. Loading...']);
    pause(0.5); 
    try
        load(targetFile);
        disp(['[GPU1] Loaded ' targetFile]);
    catch e
        disp(['[GPU1] Error loading x1.mat: ' e.message]);
    end
    
    try
        delete(targetFile);
        disp(['[GPU1] Deleted ' targetFile]);
    catch e
        disp(['[GPU1] Error deleting x1.mat: ' e.message]);
    end
    if x1(1)<=1
        x1(1)=1;
    end
    if x1(1)>=32
        x1(1)=32;
    end
    if x1(2)<=1e-6
        x1(2)=1e-6;
    end
    if x1(2)>=1e-2
        x1(2)=1e-2;
    end    
    mBS=ceil(x1(1));
    lr=x1(2);
%     mBS=x1(iterCnt,1);
%     lr=x1(iterCnt,2);
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
    disp('[GPU1] Evaluating LOCAL model...');
    c1 = evaluate_model_cost(detector, validData);
    
    % --- NaN/Inf Handling for Local ---
    if isnan(c1) || isinf(c1)
        disp('[WARN] Local Cost is NaN/Inf.');
        if iterCnt > 1
             c1 = cost(iterCnt-1); % Use history
        else
             c1 = 1000; % Max penalty
        end
        disp(['[WARN] Fallback Local Cost: ' num2str(c1)]);
    end
    
    % 2. Save local model for exchange
    exchangeFile = sprintf('.\\exchange\\model_gpu1_iter%d.mat', iterCnt);
    save(exchangeFile, 'detector');
    disp(['[GPU1] Saved model for exchange: ' exchangeFile]);
    
    % 3. Wait for other GPUs (Synchronization Barrier)
    otherFile2 = sprintf('.\\exchange\\model_gpu2_iter%d.mat', iterCnt);
    otherFile3 = sprintf('.\\exchange\\model_gpu3_iter%d.mat', iterCnt);
    
    disp('[GPU1] Waiting for model exchange...');
    while (~exist(otherFile2, 'file') || ~exist(otherFile3, 'file'))
        pause(1);
    end
    
    % 4. Select Peer to Compare
    % "If a peer model yields lower... it replaces"
    exchangeCandidates = {otherFile2, otherFile3};
    chosenIdx = randi(2);
    peerFile = exchangeCandidates{chosenIdx};
    
    disp(['[GPU1] Comparing with Peer: ' peerFile]);
    loaded = load(peerFile, 'detector');
    peer_detector = loaded.detector;
    
    % 5. Evaluate PEER (5-Fold CV)
    c1_peer = evaluate_model_cost(peer_detector, validData);
    
    % --- NaN/Inf Handling for Peer ---
    if isnan(c1_peer) || isinf(c1_peer)
        c1_peer = 1000; % Peer is broken
    end
    
    % 6. Competitive Swap
    if c1_peer < c1
        disp(['[GPU1] Peer is BETTER (' num2str(c1_peer) ' < ' num2str(c1) '). SWAPPING.']);
        detector = peer_detector;
        c1 = c1_peer;
    else
        disp(['[GPU1] Peer is WORSE or EQUAL (' num2str(c1_peer) ' >= ' num2str(c1) '). KEEPING LOCAL.']);
    end
    % ---------------------------------------------------------

    save('.\cFolder\c1.mat','c1');
    
    % --- Save Best Model for this Iteration (User Request) ---
    if ~exist('.\best_models','dir')
        mkdir('.\best_models');
    end
    bestModelFile = sprintf('.\\best_models\\best_model_gpu1_iter%d.mat', iterCnt);
    save(bestModelFile, 'detector');
    disp(['[GPU1] Saved BEST model (Post-Exchange) to: ' bestModelFile]);
    % ---------------------------------------------------------

    th1(iterCnt)=mBS;
    th2(iterCnt)=lr; 
    cost(iterCnt)=c1;
    
    if mod(gen,7)==1
        save(strcat('.\th1_th2_cost_yolov2mobilenetv2gpu1_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==2
        save(strcat('.\th1_th2_cost_yolov2resnet18gpu1_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end

    if mod(gen,7)==3
        save(strcat('.\th1_th2_cost_yolov2resnet50gpu1_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==4
        save(strcat('.\th1_th2_cost_yolov2resnet101gpu1_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==5
        save(strcat('.\th1_th2_cost_yolov2vgg16gpu1_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==6
        save(strcat('.\th1_th2_cost_yolov2vgg19gpu1_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end
    
    if mod(gen,7)==0
        save(strcat('.\th1_th2_cost_yolov2googlenetGpu1_',num2str(iterCnt),'.mat'),'th1','th2','cost');
    end    
    
    if c1<=1.2
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
