disp('--- GPU Device Check ---');
try
    count = gpuDeviceCount;
    disp(['Total GPU Count: ' num2str(count)]);
    for i = 1:count
        dev = gpuDevice(i);
        disp(['MATLAB GPU Index ' num2str(i) ': ' dev.Name ' (Compute Capability ' dev.ComputeCapability ')']);
        % Reset to avoid locking
        reset(dev);
    end
catch e
    disp(['Error checking GPUs: ' e.message]);
end
disp('--- End GPU Check ---');
exit;
