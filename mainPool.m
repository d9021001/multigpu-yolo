clear all; clc; close all; warning off
disp(['[MainPool] Started at ' datestr(now)]);
disp('[MainPool] Cleaning up old files...');
% gpuD1=gpuDevice(1) % 2080TI
% reset(gpuD1)
% gpuD2=gpuDevice(2) % 2080TI
% reset(gpuD2)
% gpuD3=gpuDevice(3) % 2080
% reset(gpuD3)
% gpuD4=gpuDevice(4) % 1080TI is slower than 2080
% reset(gpuD4)
% return

delete('.\xFolder\x1.mat');
delete('.\xFolder\x2.mat');
delete('.\xFolder\x3.mat');
delete('.\cFolder\c1.mat');
delete('.\cFolder\c2.mat');
delete('.\cFolder\c3.mat');
costHistory=[]; k=0; minCost=inf;

% 3) Random Initialization
% Randomly initialize three learning rates (mu1, mu2, mu3) from range (0, 0.01)
% Randomly initialize three batch sizes (sigma1, sigma2, sigma3) as integers [1, 32]

x = zeros(2,3);
x(1,:) = randi([1, 32], 1, 3); % Batch Size (sigma)
x(2,:) = rand(1, 3) * 0.01;    % Learning Rate (mu)

x23=x;
save_x1_x2_x3
disp('[MainPool] Initial X saved. Loading initial costs...');
load_cost  
n=1;
Xx{n}=x23;

disp(['[MainPool] Starting optimization loop. Max iters: 10']);

for iter=1:10
    disp(['[MainPool] Iteration ' num2str(iter) '...']);
    [cost_max,Index_high] = max(cost);
    [cost_min,Index_low] = min(cost);
    if cost_min<minCost
        k=k+1;
        costHistory(k)=cost_min;
        minCost=cost_min;
    end
    
    % Step 5.3: Check Convergence (User Requirement)
    % If current best cost meets goal, terminate.
    if minCost <= 1.0 
        disp(['[MainPool] Convergence Reached! minCost: ' num2str(minCost)]);
        break;
    end
    x_high = x(:,Index_high);
    x_low  = x(:,Index_low) ;   
    cost_high = cost(Index_high);
    cost_low = cost(Index_low);
    x_base = ((sum(x'))'-x_high)/size(x,1);
    
    alfa = 0.9+rand*0.2;    
    xReflect(:,1) = x_base+alfa*(x_base-x_high); 
    alfa = 0.9+rand*0.2;    
    xReflect(:,2) = x_base+alfa*(x_base-x_high); 
    alfa = 0.9+rand*0.2;    
    xReflect(:,3) = x_base+alfa*(x_base-x_high);
       
    x23=xReflect;    
    save_x1_x2_x3
    load_cost 
    n=n+1;
    Xx{n}=x23;
    
    costReflect=cost;    
    [cost_reflect,Index_low_reflect] = min(costReflect);
    x_reflect=xReflect(:,Index_low_reflect);
    
    if cost_reflect<cost_high
        if cost_reflect<cost_low
            
            gama = 1.7+rand*0.6;
            xExtend(:,1) = x_base+gama*(x_reflect-x_base); 
            gama = 1.7+rand*0.6;
            xExtend(:,2) = x_base+gama*(x_reflect-x_base); 
            gama = 1.7+rand*0.6;
            xExtend(:,3) = x_base+gama*(x_reflect-x_base); 
            
            x23=xExtend;               
            save_x1_x2_x3
            load_cost
            n=n+1;
            Xx{n}=x23;
            
            costExtend=cost;  
            [cost_extend,Index_low_extend] = min(costExtend);
            x_extend=xExtend(:,Index_low_extend);
    
            if cost_extend<cost_reflect
               x(:,Index_high) = x_extend;
               cost(Index_high)=cost_extend;
            else
               x(:,Index_high) = x_reflect;    
               cost(Index_high)=cost_reflect;
            end
        else                
            x(:,Index_high) = x_reflect;
            cost(Index_high)=cost_reflect;
        end
    else
        
        beta = 0.3+rand*0.4;
        xCentral(:,1) = x_base+beta*(x_high-x_base);
        beta = 0.3+rand*0.4;
        xCentral(:,2) = x_base+beta*(x_high-x_base);
        beta = 0.3+rand*0.4;
        xCentral(:,3) = x_base+beta*(x_high-x_base);
        
        x23=xCentral;             
        save_x1_x2_x3
        load_cost
        n=n+1;
        Xx{n}=x23;
    
        costCentral=cost;
        [cost_central,Index_low_central] = min(costCentral);
        x_central=xCentral(:,Index_low_central);
            
        if cost_central < cost_high
            x(:,Index_high) = x_central;
            cost(Index_high)=cost_central;
        else        
            x = (x+(x_low*ones(1,size(x,2))))/2;       
            
            x23=x;            
            save_x1_x2_x3
            load_cost
            n=n+1;
            Xx{n}=x23;
        end
    end
end
[cost_min,Index_low] = min(cost);
xsol=x(:,Index_low);
save xsol_Xx_costHistory xsol Xx costHistory
figure, plot(costHistory)

