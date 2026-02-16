disp('--- Checking Available Networks ---');
nets = {'googlenet', 'resnet101', 'resnet50', 'resnet18', 'vgg16', 'vgg19', 'mobilenetv2', 'inceptionv3'};
for i = 1:length(nets)
    name = nets{i};
    try
        eval([name ';']);
        disp(['[OK] ' name ' is available.']);
    catch e
        disp(['[FAIL] ' name ' is NOT available: ' e.message]);
    end
end
disp('--- End Check ---');
exit;
