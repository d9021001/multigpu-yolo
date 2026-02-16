disp('--- CSV DIAGNOSIS START ---');
filename = 'tr1_fix.csv';
if ~exist(filename, 'file')
    disp(['File not found: ' filename]);
else
    disp(['File found: ' filename]);
    
    % Check file contents (first few lines)
    disp('First 5 lines of file:');
    fid = fopen(filename, 'r');
    for i=1:5
        line = fgetl(fid);
        disp(['L' num2str(i) ': ' line]);
    end
    fclose(fid);

    % Check default readtable
    disp('Attempting basic readtable...');
    try
        T_basic = readtable(filename);
        disp('Basic readtable vars:');
        disp(T_basic.Properties.VariableNames);
    catch e
        disp(['Basic readtable failed: ' e.message]);
    end

    % Check detectImportOptions
    disp('Checking detectImportOptions...');
    try
        opts = detectImportOptions(filename);
        disp(['VariableNamesLine: ' num2str(opts.VariableNamesLine)]);
        disp(['DataLine: ' num2str(opts.DataLine)]);
        disp('Detected VariableTypes:');
        disp(opts.VariableTypes);
        disp('Detected VariableNames:');
        disp(opts.VariableNames);
        
        opts.VariableNamingRule = 'preserve';
        T_opts = readtable(filename, opts);
        disp('Options readtable vars:');
        disp(T_opts.Properties.VariableNames);
    catch e
        disp(['Options check failed: ' e.message]);
    end
end
disp('--- CSV DIAGNOSIS END ---');
exit;
