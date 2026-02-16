L=2; folder='.\cFolder\';
% Wait until all three cost files exist
c1_file = fullfile(folder, 'c1.mat');
c2_file = fullfile(folder, 'c2.mat');
c3_file = fullfile(folder, 'c3.mat');

disp('[LoadCost] Waiting for c1.mat, c2.mat, c3.mat...');
while ~exist(c1_file, 'file') || ~exist(c2_file, 'file') || ~exist(c3_file, 'file')
    pause(1);
end
disp('[LoadCost] All cost files found. Loading...');
pause(0.5); % Ensure write completion
load(c1_file);
load(c2_file);
load(c3_file);

cost(1)=c1;
cost(2)=c2;
cost(3)=c3;

delete('.\cFolder\c1.mat');
delete('.\cFolder\c2.mat');
delete('.\cFolder\c3.mat');  