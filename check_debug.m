disp('--- DEBUG START ---');
disp(['Current Directory: ', pwd]);
w = which('read_yolo_csv');
disp(['Which read_yolo_csv: ', w]);

if isempty(w)
    disp('read_yolo_csv not found in path.');
else
    d = dir(w);
    disp(['File Size: ', num2str(d.bytes)]);
    disp('File Content Preview (First 20 lines):');
    dbtype(w, '1:20');
end

disp('--- DEBUG END ---');
exit;
