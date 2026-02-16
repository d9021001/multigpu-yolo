try
    disp('Testing read_yolo_csv with tr1_fix.csv...');
    data = read_yolo_csv('tr1_fix.csv');
    disp(['Success! Rows: ', num2str(height(data))]);
    disp('First row image:');
    disp(data.imageFilename{1});
    disp('First row boxes:');
    disp(data.Boxes{1});
    
    disp('Testing read_yolo_csv with valid1_fix.csv...');
    data = read_yolo_csv('valid1_fix.csv');
    disp(['Success! Rows: ', num2str(height(data))]);
    
    disp('Verification Complete.');
catch ME
    disp(['Error: ', ME.message]);
    disp(ME.stack(1));
    exit(1);
end
exit(0);
