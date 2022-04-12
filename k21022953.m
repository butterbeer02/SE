%% Coursework 2022 -- Part 1 (Due on March 6)

%% Main Body -- Do NOT edit

close all; clear; clc;
load('dataset_wisconsin.mat'); % loads full data set: X and t

% Section 1
[X_tr,t_tr,X_te,t_te] = split_tr_te(X, t, 0.7); % tr stands for training, te for test

% Section 2
t_hat1_tr =             hard_predictor1(X_tr);
t_hat1_te =             hard_predictor1(X_te);
t_hat2_tr =             hard_predictor2(X_tr);
t_hat2_te =             hard_predictor2(X_te);

% Section 3
[sens1_te, spec1_te] =  sensitivity_and_specificity( t_hat1_te, t_te );
[sens2_te, spec2_te] =  sensitivity_and_specificity( t_hat2_te, t_te );

% Section 4
loss1_te =              detection_error_loss( t_hat1_te, t_te );
loss2_te =              detection_error_loss( t_hat2_te, t_te );

% Section 5
discussionA();

% Section 6
theta_ls3 =             LSsolver(               X3(X_tr) ,      t_tr    );
theta_ls4 =             LSsolver(               X4(X_tr) ,      t_tr    );

% Section 7
Ngrid =                 101; % number of ponts in grid
[mRadius,mTexture] =    meshgrid(linspace(5,30,Ngrid),linspace(8,40,Ngrid));
X_gr =                  [mRadius(:),mTexture(:)]; % gr for grid

t_hat_ls3_gr =          linear_combiner(        X3(X_gr) ,  theta_ls3 );
t_hat_ls4_gr =          linear_combiner(        X4(X_gr) ,  theta_ls4 );

figure; hold on;
contourf(mRadius,mTexture,max(0,min(1,reshape(t_hat_ls3_gr,[Ngrid,Ngrid]))),'ShowText','on','DisplayName','LS solution'); inc_vec = linspace(0,1,11).'; colormap([inc_vec,1-inc_vec,0*inc_vec]);
plot(X_te(t_te==0,1),X_te(t_te==0,2),'o','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','c','DisplayName','t=0 test');
plot(X_te(t_te==1,1),X_te(t_te==1,2),'^','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','m','DisplayName','t=1 test');
contour (mRadius,mTexture,max(0,min(1,reshape(t_hat_ls3_gr,[Ngrid,Ngrid]))),[0.5,0.5],'y--','LineWidth',3,'DisplayName','Decision line');
xlabel('$x^{(1)}$ radius mean','interpreter','latex'); ylabel('$x^{(2)}$ texture mean','interpreter','latex'); colorbar; title('$\hat{t}_3(X|\theta_3)$','interpreter','latex'); legend show;
figure; hold on;
contourf(mRadius,mTexture,max(0,min(1,reshape(t_hat_ls4_gr,[Ngrid,Ngrid]))),'ShowText','on','DisplayName','LS solution'); inc_vec = linspace(0,1,11).'; colormap([inc_vec,1-inc_vec,0*inc_vec]);
plot(X_te(t_te==0,1),X_te(t_te==0,2),'o','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','c','DisplayName','t=0 test');
plot(X_te(t_te==1,1),X_te(t_te==1,2),'^','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','m','DisplayName','t=1 test');
contour (mRadius,mTexture,max(0,min(1,reshape(t_hat_ls4_gr,[Ngrid,Ngrid]))),[0.5,0.5],'y--','LineWidth',3,'DisplayName','Decision line');
xlabel('$x^{(1)}$ radius mean','interpreter','latex'); ylabel('$x^{(2)}$ texture mean','interpreter','latex'); colorbar; title('$\hat{t}_4(X|\theta_4)$','interpreter','latex'); legend show;

% Section 8
t_hat_ls3_te =          linear_combiner(        X3(X_te) ,              theta_ls3   );
t_hat_ls4_te =          linear_combiner(        X4(X_te) ,              theta_ls4   );
mse_loss3_te =          mse_loss(               t_hat_ls3_te ,          t_te        );
mse_loss4_te =          mse_loss(               t_hat_ls4_te ,          t_te        );
det_loss3_te =          detection_error_loss(   (t_hat_ls3_te>0.5) ,    t_te        );
det_loss4_te =          detection_error_loss(   (t_hat_ls4_te>0.5) ,    t_te        );

% Section 9
discussionB();

% Section 10
theta_ls5 =             LSsolver(               X5(X_tr) ,              t_tr        );
t_hat_ls5_te =          linear_combiner(        X5(X_te) ,              theta_ls5   );
loss5_te =              detection_error_loss(   (t_hat_ls5_te>0.5) ,    t_te        );

% Section 11
v_ratio_tr =            (10:3:100)/100;
v_loss_LS =             loss_vs_training_size( X_tr, t_tr, X_te, t_te, v_ratio_tr );
figure; plot(v_ratio_tr,v_loss_LS);
xlabel('percentage of used first training samples'); ylabel('Test loss'); title('Detection Error test loss vs. training size');

function out = LSsolver(X,t) % Least Square solver
    out = ( X.' * X ) \ (X.' * t);
end


%% Functions -- Fill in the functions with your own code from this point

% Function 1
function [X_tr,t_tr,X_te,t_te] = split_tr_te(X, t, eta)
n = round(eta*size(X,1));
X_tr = X(1:n,:);
t_tr = t(1:n);
X_te = X(n+1:end,:);
t_te = t(n+1:end);
end

% Function 2
function t_hat1 = hard_predictor1(X)
x_hat1 = X(:,1);
t_hat1 = (x_hat1 > 14).';
end

% Function 3
function t_hat2 = hard_predictor2(X)
x_hat2 = X(:,2);
t_hat2 = (x_hat2 > 20).';
end

% Function 4
function [sens, spec] = sensitivity_and_specificity( t_hat, t )
TP = 0;
FN = 0;
TN = 0;
FP = 0;
for i = 1 : length(t_hat.')
    if t_hat(i) == 1 && t(i)==1
        TP = TP + 1;
    else
        if t_hat(i) == 0 && t(i)==1
            FN = FN + 1;
        else
            if t_hat(i) == 1 && t(i)==0
                TN = TN + 1;
            else
                if t_hat(i) == 0 && t(i)==0
                FP = FP + 1;
                end
            end
        end
    end
end
sens = TP/(TP+FN);
spec = FP/(TN+FP);
end                       

% Function 5
function loss = detection_error_loss( t_hat, t )
T = 0;
for i = 1 : length(t.') 
    if t_hat(i) == t(i)
            T = T + 1;
    end
end
loss = 1-(T/length(t.'));
end

% Function 6
function discussionA()
    disp('discussion A:');
    disp('<<The predictor t1 is better to predict breast cancer on its own, since it has a higher sensitivity and specificity than the predictor t2. ')
    disp('In terms of loss rates, the predictor t1 possesses a lower probability of detection error than the predictor t2, which means greater accuracy>>');
end

% Function 7
function out = X3(X)
    x1 = ones(size(X,1),1);
    x2 = X(:,1);
    x3 = X(:,2);
    out = [x1,x2,x3];
end

% Function 8
function out = X4(X)
    x1 = ones(size(X,1),1);
    x2 = X(:,1);
    x3 = X(:,2);
    x4 = x2.^2;
    x5 = x3.^2;
    x6 = x2.*x3;
    out = [x1,x2,x3,x4,x5,x6];
end

% Function 9
function out = linear_combiner( X ,  theta )
out = X*theta;
end

% Function 10
function out = mse_loss( t_hat ,  t )
m = 0;
n = size(t,1);
for i = 1: n
    L = (t_hat(i)-t(i))^2;
    m = m + L;
end
out = m/n;
end

% Function 11
function discussionB()
    disp('discussion B:');
    disp('<<The predictor t4 has higher model capacity, since the data shows that it has a lower mean square error loss (0.1036) than that of t3 (0.1045).') 
    disp('The higher complexity is useful for this problem. Due to the fact that more feature parameters are used in t4, it has a higher accuracy than t3 .>>');
end

% Function 12
function out = X5(X)
    out = [ones(size(X,1),1) X];
end

% Function 13
function v_loss_LS = loss_vs_training_size( X_tr, t_tr, X_te, t_te, v_ratio_tr )
Loss=zeros(1,size(v_ratio_tr,1));
Tr = zeros(size(v_ratio_tr));
for i = 1:size(v_ratio_tr,2)
Tr(i)=round(size(X_tr,1)*v_ratio_tr(i));
X_Train = X_tr(1:Tr(i),1:size(X_tr,2));
t_Train = t_tr(1:Tr(i),1);
Loss(i) = detection_error_loss((linear_combiner(X5(X_te),LSsolver(X5(X_Train), t_Train))>0.5),t_te);
end
v_loss_LS= Loss;
end