x23(1,find(x23(1,:)>=32))=32;
x23(1,find(x23(1,:)<=1))=1;
x23(2,find(x23(2,:)>=(1e-2)))=1e-2;
x23(2,find(x23(2,:)<=(1e-6)))=1e-6;

x1=x23(:,1)';
x2=x23(:,2)';
x3=x23(:,3)';

save('.\xFolder\x1.mat','x1');
if exist('.\xFolder\x1.mat', 'file'), disp('[Main] x1.mat created successfully.'); else, disp('[Main] ERROR: x1.mat NOT created.'); end

save('.\xFolder\x2.mat','x2');
if exist('.\xFolder\x2.mat', 'file'), disp('[Main] x2.mat created successfully.'); else, disp('[Main] ERROR: x2.mat NOT created.'); end

save('.\xFolder\x3.mat','x3');
if exist('.\xFolder\x3.mat', 'file'), disp('[Main] x3.mat created successfully.'); else, disp('[Main] ERROR: x3.mat NOT created.'); end