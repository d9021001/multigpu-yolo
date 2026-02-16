function lgraph = get_lgraph_safe(gen, imageSize, numClasses, anchorBoxes)
    % Attempt to get the network based on 'gen'
    % If missing, fall back to other available networks
    
    % Define network configs: name, constructor, featureLayer
    % Order matches the mod(gen, 7) logic in original script roughly
    % 0: googlenet
    % 1: mobilenetv2
    % 2: resnet18
    % 3: resnet50
    % 4: resnet101
    % 5: vgg16
    % 6: vgg19
    
    idx = mod(gen, 7);
    
    % Order of preference for loading: specific requested one first, then fallbacks
    preferredIdx = idx;
    
    % Map indices to configs
    configs = {
        0, 'googlenet', 'inception_4d-output';
        1, 'mobilenetv2', 'block_13_expand_relu';
        2, 'resnet18', 'res4b_relu';
        3, 'resnet50', 'activation_40_relu';
        4, 'resnet101', 'res4b22_relu';
        5, 'vgg16', 'relu5_3';
        6, 'vgg19', 'relu5_4';
    };

    lgraph = [];
    
    % Try the preferred one first
    config = get_config_by_id(configs, preferredIdx);
    if ~isempty(config)
        lgraph = try_load_network(config, imageSize, numClasses, anchorBoxes);
    end
    
    if ~isempty(lgraph)
        return;
    end
    
    disp(['[WARN] Preferred network ID ' num2str(preferredIdx) ' failed. Trying fallbacks...']);
    
    % Fallback priority list (ResNet18 is usually available and fast)
    fallbackIds = [2, 3, 5, 1, 4, 6, 0]; 
    
    for i = 1:length(fallbackIds)
        fbId = fallbackIds(i);
        if fbId == preferredIdx
            continue; % Already tried
        end
        
        config = get_config_by_id(configs, fbId);
        lgraph = try_load_network(config, imageSize, numClasses, anchorBoxes);
        
        if ~isempty(lgraph)
            disp(['[INFO] Successfully fell back to network ID ' num2str(fbId) ' (' config{2} ').']);
            return;
        end
    end
    
    % FINAL FALLBACK: Create a simple custom Tiny YOLO-like network from scratch
    % This is used when NO support packages are installed.
    disp('[WARN] All pretrained networks failed. Constructing a simple custom network (random weights)...');
    try
        lgraph = create_simple_yolov2_graph(imageSize, numClasses, anchorBoxes);
        disp('[INFO] Successfully created custom random network.');
        return;
    catch e
        disp(['[ERROR] Failed to create custom network: ' e.message]);
        % Rethrow to let the user see the error if even this fails
        rethrow(e);
    end
end

function lgraph = create_simple_yolov2_graph(imageSize, numClasses, anchorBoxes)
    % Create a very simple CNN feature extractor
    layers = [
        imageInputLayer(imageSize, 'Name', 'input')
        
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3') % Feature extraction output
    ];

    % Convert to layer graph
    lgraph = layerGraph(layers);
    
    % Manually add YOLOv2 detection head layers
    % This bypasses yolov2Layers() which seems to reject custom LayerGraph inputs in some versions.
    numAnchors = size(anchorBoxes, 1);
    numPreds = 5 + numClasses;
    
    detectionLayers = [
        convolution2dLayer(1, numAnchors * numPreds, 'Name', 'yolov2Convweights')
        yolov2TransformLayer(numAnchors, 'Name', 'yolov2Transform')
        yolov2OutputLayer(anchorBoxes, 'Name', 'yolov2Output')
    ];
    
    lgraph = addLayers(lgraph, detectionLayers);
    lgraph = connectLayers(lgraph, 'relu3', 'yolov2Convweights');
end

function config = get_config_by_id(configs, id)
    config = {};
    for i = 1:size(configs, 1)
        if configs{i, 1} == id
            config = configs(i, :);
            return;
        end
    end
end

function lgraph = try_load_network(config, imageSize, numClasses, anchorBoxes)
    lgraph = [];
    netName = config{2};
    layerName = config{3};
    
    try
        % Construct the network using eval to handle dynamic names
        baseNetwork = eval(netName);
        lgraph = yolov2Layers(imageSize, numClasses, anchorBoxes, baseNetwork, layerName);
    catch e
        % specific error catching for missing support header
        % disp(['Failed to load ' netName ': ' e.message]);
    end
end
