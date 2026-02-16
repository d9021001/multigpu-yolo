disp('Checking for required functions...');
check_func('trainYOLOv2ObjectDetector');
check_func('yolov2TransformLayer');
check_func('yolov2OutputLayer');
check_func('trainingOptions');
check_func('gpuDevice');

disp('Listing installed toolboxes:');
v = ver;
for i = 1:length(v)
    disp(v(i).Name);
end

function check_func(name)
    if exist(name)
        disp(['[OK] ' name ' exists.']);
    else
        disp(['[MISSING] ' name ' is NOT found.']);
    end
end
