function stop = stopIfLossOK(info,N)
stop = false;
if info.State == "start"  
    
elseif (~isempty(info.TrainingLoss)) & (~isempty(info.ValidationLoss))
    if (info.TrainingLoss<N) & (info.ValidationLoss<N)
	   stop = true;
    end  
elseif ~isempty(info.TrainingLoss)
    if info.TrainingLoss<N
	   stop = true;
    end  	
end

end