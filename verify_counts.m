try
    % Use detectImportOptions to speed up and avoid parsing errors
    opts = detectImportOptions('tr1_fix.csv');
    opts.VariableNamingRule = 'preserve';
    
    t1 = readtable('tr1_fix.csv', opts); 
    n1 = height(t1);
    disp(['tr1: ' num2str(n1)]);
    
    t2 = readtable('tr2_fix.csv', opts); 
    n2 = height(t2);
    disp(['tr2: ' num2str(n2)]);
    
    t3 = readtable('tr3_fix.csv', opts); 
    n3 = height(t3);
    disp(['tr3: ' num2str(n3)]);
    
    % Parent file
    parentFile = fullfile('..', 'tr_fix.csv');
    if exist(parentFile, 'file')
        optsP = detectImportOptions(parentFile);
        optsP.VariableNamingRule = 'preserve';
        t = readtable(parentFile, optsP); 
        n = height(t);
        disp(['tr_fix: ' num2str(n)]);
        
        sum_split = n1 + n2 + n3;
        disp(['Sum of splits: ' num2str(sum_split)]);
        
        if sum_split == n
            disp('VERIFICATION: MATCH');
        else
            disp(['VERIFICATION: MISMATCH (Diff: ' num2str(sum_split - n) ')']);
        end
    else
        disp(['MISSING: ' parentFile]);
    end
catch e
    disp(['ERROR: ' e.message]);
end
