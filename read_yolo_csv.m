function data = read_yolo_csv(filename)
    % Read CSV file for YOLOv2 training
    % Expects columns: imageFilename, bboxes

    % Check if file exists, try parent directory if not
    if ~exist(filename, 'file')
        if exist(fullfile('..', filename), 'file')
            filename = fullfile('..', filename);
        else
            error('File not found: %s', filename);
        end
    end

    % Explicitly define options to handle quoted JSON fields
    opts = delimitedTextImportOptions('NumVariables', 2);
    opts.DataLines = [2, Inf];
    opts.Delimiter = ',';
    opts.VariableNames = {'imageFilename', 'bboxes'};
    opts.VariableTypes = {'char', 'char'};
    opts.PreserveVariableNames = true;
    
    % Read table with explicit options
    try
        T = readtable(filename, opts);
    catch e
        % Fallback for older MATLAB versions or if options fail
        warning('Explicit options failed: %s. Reverting to detectImportOptions.', e.message);
        opts = detectImportOptions(filename);
        opts.VariableNamingRule = 'preserve';
        T = readtable(filename, opts);
    end

    % Verify columns again just in case
    validCols = T.Properties.VariableNames;
    if ~ismember('imageFilename', validCols)
         error('Table does not contain "imageFilename". Columns found: %s', strjoin(validCols, ', '));
    end
    
    % Prepare Output Table
    imageFilename = cell(height(T), 1);
    Boxes = cell(height(T), 1);
    
    % Determine Root for Images (Assume parallel to CSV if relative)
    % If CSV is in '..', images are likely in '..\rgb_...'
    % T.imageFilename contains 'rgb_\...'
    
    fileDir = fileparts(filename); % e.g. '..'
    
    for i = 1:height(T)
        % Parse Image Path
        relPath = T.imageFilename{i};
        imageFilename{i} = fullfile(fileDir, relPath);
        
        % If file not found in CSV dir, check parent dir
        if ~exist(imageFilename{i}, 'file')
            parentPath = fullfile('..', relPath);
            if exist(parentPath, 'file')
                imageFilename{i} = parentPath;
            end
        end
        
        % Parse BBoxes
        str = T.bboxes{i};
        
        % Normalize string format for jsondecode
        % Handle Numpy string format: "[0.1 0.2]" -> "[0.1, 0.2]"
        % Replace space with comma
        str = strrep(str, ' ', ',');
        
        % Cleanup potential double commas or leading commas
        % (Regex is safer but strrep is faster if format is simple)
        % "[," -> "["
        str = strrep(str, '[,', '[');
        % ",," -> ","
        while contains(str, ',,')
            str = strrep(str, ',,', ',');
        end
        % ",]" -> "]"
        str = strrep(str, ',]', ']');
        
        try
            val = jsondecode(str);
            if isempty(val)
                Boxes{i} = [];
            else
                % Usually jsondecode returns Mx4 double for [[...],[...]]
                % Or 4x1 for single box if shape is weird? 
                % Python list of list -> MATLAB Mx4 double matrix
                if size(val, 2) ~= 4 && size(val, 1) == 4
                     % Sometimes single box [1,2,3,4] comes as 4x1
                     val = val'; 
                end
                Boxes{i} = double(val);
            end
        catch
            warning('Failed to parse bbox at row %d: %s', i, str);
            Boxes{i} = [];
        end
    end
    
    data = table(imageFilename, Boxes);
end
