function stop = stopIfLossOK(info)
    % STOPIFLOSSOK  Output function for trainNetwork
    %   stop = stopIfLossOK(info) returns true to stop training if a condition is met.
    %   This prevents "Undefined function" errors when used in trainingOptions.

    stop = false;
    
    % Option: Stop if loss is very low (e.g. < 0.01)
    % if ~isempty(info.TrainingLoss) && info.TrainingLoss < 0.005
    %     disp('Loss is very low. Stopping early.');
    %     stop = true;
    % end
end
