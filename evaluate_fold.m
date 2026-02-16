function [precision, recall] = evaluate_fold(detector, data)
    % Evaluate detector on fold data
    % data: Table returned by read_yolo_csv(fold_csv) [imageFilename, Boxes]
    
    tp = 0;
    fp = 0;
    fn = 0;
    
    % Thresholds
    CONF_THRESH = 0.5;
    IOU_THRESH = 0.5;
    
    for i = 1:height(data)
        imgFile = data.imageFilename{i};
        gtBoxes = data.Boxes{i}; % Mx4
        
        try
            img = imread(imgFile);
            [bboxes, scores] = detect(detector, img, 'Threshold', CONF_THRESH);
            
            if isempty(gtBoxes)
                fp = fp + size(bboxes, 1);
                continue;
            end
            
            if isempty(bboxes)
                fn = fn + size(gtBoxes, 1);
                continue;
            end
            
            % Sort by score
            [scores, sortIdx] = sort(scores, 'descend');
            bboxes = bboxes(sortIdx, :);
            
            % Calculate IoU (NxM matrix)
            ious = bboxOverlapRatio(bboxes, gtBoxes); 
            
            matched_gt = false(size(gtBoxes, 1), 1);
            
            for j = 1:size(bboxes, 1)
                best_iou = 0;
                best_gt_idx = 0;
                
                % Find best matching GT for this detection
                for k = 1:size(gtBoxes, 1)
                    if ious(j, k) > best_iou
                        best_iou = ious(j, k);
                        best_gt_idx = k;
                    end
                end
                
                if best_iou >= IOU_THRESH
                    if ~matched_gt(best_gt_idx)
                        tp = tp + 1;
                        matched_gt(best_gt_idx) = true;
                    else
                        fp = fp + 1; % Duplicate detection
                    end
                else
                    fp = fp + 1;
                end
            end
            
            fn = fn + sum(~matched_gt);
            
        catch ME
            warning('Error processing image %s: %s', imgFile, ME.message);
        end
    end
    
    precision = tp / (tp + fp + eps);
    recall = tp / (tp + fn + eps);
end
