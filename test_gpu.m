disp('Checking GPU Device Count...');
try
    c = gpuDeviceCount;
    disp(['GPU Count: ', num2str(c)]);
    if c > 0
        for i=1:c
            disp(['Checking GPU ', num2str(i)]);
            d = gpuDevice(i);
            disp(d);
            reset(d);
        end
    else
        disp('No GPUs found by MATLAB');
    end
catch e
    disp('Error checking GPU:');
    disp(getReport(e));
end
quit;
