%% SGD with unbalanced transport for KL EBM 
%
% E(x) = \sum_i w_i c_i (x,z_i)_+
%
% with |x|=|z_i|=1, c_i \in R, w_i \ge 0.
%
% Expectations over the data (teacher) computed with the full data set
% Expectations over the student computed on evolving walkers
% Unabalanced transport in WFR.
%
% SGD performed with full data set
% Number of walkers = twice size of data set

%% options & input parameters
% clear all;
reset_data = true;                                 % reset training data set
reset_test = reset_data;                           % reset test data set
stop_restart = false;                               % stop restart after Nt/2 iterations
reg_w = true;                                      % enforce |w|<beta
n_print  = 5;

scr_siz = get(0,'ScreenSize') ;
fig_size = floor([.1*scr_siz(3) .1*scr_siz(4) .9*scr_siz(3) .9*scr_siz(4)]);

Nt = 1e3;                                       % number of step of training
N_restart = Nt;                                 % restart walkers every N_restart (if N_restart=1 => SM)

d1 = 15;                                         % input dimension >3 (if d1=3, use EBMdual3d.m)

hx = 2e-2;                                      % time step for walkers x
alph = (1+.1/N_restart/hx);
hz = .2*alph;                                        % time step for features z (set to zero for lazy training)
hw = .2*alph;                                        % time step for weight w (unbalanced transport)

n_t = 2;                                        % number of teachers
n_unit = 2^6;                                   % number of students
n_test = 1e5;                                   % test data size
n_test2 = 1e5;                                  % test data size for SM computation
n_data = 1e4;                                   % training data sise
n_walker = 2e4;                                 % number of walkers


%% energy function & gradient wrt x
f1 = @(x,z,c) mean(max(x'*z,0).*(repmat(c,size(x,2),1)),2);
df1 = @(x,z,c) z*(repmat(c',1,size(x,2)).*(z'*x>0))/size(z,2);

%% teacher

if reset_data 
    beta = 10;
    ct = -beta*ones(1,n_t);
    % zt = [0;1;0];
%     zt = randn(d1,n_t);
    % zt(:,1) = [1;.8;0];
    % zt(:,2) = [-1;.8;0];
    zt = [-0.1581,  0.5860;   %old teacher
        -0.1232,  0.1702;
        -0.1913,  -0.1274;
        0.4161,  0.0954;
        0.4344,  0.1495;
        0.1045,  0.1055;
        0.3137,  0.2156;
        0.2839,  0.1797;
        0.1072,  -0.1128;
        0.1849,  0.0606;
        -0.2719,  -0.1669;
        -0.1624,  0.0621;
        -0.0897,  -0.4499;
        -0.2070,  0.4556;
        0.4179,  0.1786];
%     zt = [-0.1581    0.1901
%         -0.1232    0.0444
%         -0.1913    0.1050
%         0.4161   -0.4427
%         0.4344   -0.4244
%         0.1045   -0.0265
%         0.3137   -0.2217
%         0.2839   -0.2021
%         0.1072    0.0412
%         0.1849   -0.2631
%         -0.2719    0.3599
%         -0.1624    0.1848
%         -0.0897    0.0721
%         -0.2070    0.2579
%         0.4179   -0.4216];
    zt = zt./sqrt(repmat(sum(zt.^2,1),d1,1));
end


%% data set created by rejection method using box optimized to min f(x)
if reset_test 
    dt = 1e-2;
    tol = 1e-8;
    norm1 = 1;
    n_gd = 1e5;
    n_trial = 4e7;
    
    x1 = randn(d1,n_gd);
    x1 = x1./sqrt(ones(d1,1)*sum(x1.^2,1));
    
    while norm1>tol
        x1 = x1 - dt*df1(x1,zt,ct);                            % GD
        x1 = x1./sqrt(repmat(sum(x1.^2,1),d1,1));
        [~,im] = min(f1(x1,zt,ct));
        x1m = x1(:,im);
        df1m = df1(x1m,zt,ct);
        norm1 = norm(df1m-(df1m'*x1m)/(x1m'*x1m)*x1m);
    end
    
    [f1m,im] = min(f1(x1,zt,ct));
    ef1_max = exp(-f1m);
    
    n_test0 = 0;
    while n_test0 < n_test
        x0 = randn(d1,n_trial);
        x0 = x0./sqrt(repmat(sum(x0.^2,1),d1,1));
        x0 = x0(:,rand(1,n_trial)*ef1_max<exp(-f1(x0,zt,ct)'));
        n_test0 = size(x0,2);
    end
    
    x_test = x0(:,1:n_test);
    
    n_test0 = 0;
    while n_test0 < n_data
        x0 = randn(d1,n_trial);
        x0 = x0./sqrt(repmat(sum(x0.^2,1),d1,1));
        x0 = x0(:,rand(1,n_trial)*ef1_max<exp(-f1(x0,zt,ct)'));
        n_test0 = size(x0,2);
    end
    xt = x0(:,1:n_data);
    
    fprintf('New data set created\n')
else
    fprintf('Old data set used\n')
end

f1_t = f1(x_test,zt,ct);
f1_t = f1_t-min(f1_t);


%% initial weigths, features, walkers 
if reset_data
    z = randn(d1,n_unit);
    z = z./sqrt(repmat(sum(z.^2,1),d1,1));
    c = [-ones(1,n_unit/2),ones(1,n_unit/2)];
    w = ones(1,n_unit);
else
    z = z0; c = c0; w = w0;
end
z0 = z; c0 = c;  w0 = w;                             % save initial values


wc = w.*c;
xb1 = xt;
xb = [xb1,xb1];

%% initial KL, score and TV norm
KL1 = zeros(Nt,1);
SM1 = zeros(Nt,1);
mw1 = zeros(Nt,1);

f1_s = f1(x_test,z,wc);
f1_s = f1_s-min(f1_s);
KL1(1) = log(mean(exp(-f1_s+f1_t)))+mean(f1_s-f1_t);
df1_t = df1(x_test,zt,ct);
df1_t = df1_t - repmat(sum(x_test.*df1_t,1),d1,1).*x_test;
df1_s = df1(x_test,z,wc);
df1_s = df1_s - repmat(sum(x_test.*df1_s,1),d1,1).*x_test;
SM1(1) = mean(sum((df1_t-df1_s).^2,1));
mw1(1) = mean(w);
%%  training   

for i = 1:Nt
    if i > 2000
        n_print = 500;
    elseif i > 200
        n_print = 1e2;
    elseif i > 40
        n_print = 2e1;
    end
    
    % walkers
%     if mod(i,N_restart)==0; xb1 = xt(:,randi(n_data,n_walker/2,1)); xb = [xb1,xb1]; end
    if mod(i,N_restart)==0; xb = [xt,xt]; end
    if and(stop_restart==1,i>Nt/2); N_restart = Nt; end
    dx = df1(xb,z,wc);
    r1 = randn(d1,n_walker/2); 
    xb = xb - hx*dx + sqrt(2*hx)*[r1,-r1];                            
    xb = xb./sqrt(repmat(sum(xb.^2,1),d1,1));
    
    % weights & features
    dz = (xt*((xt'*z>0).*(repmat(c,size(xt,2),1))))/n_data ...
        - (xb*((xb'*z>0).*(repmat(c,size(xb,2),1))))/n_walker;
    dz = dz - repmat(sum(z.*dz,1),d1,1).*z;
    dc = mean(max(xt'*z,0),1) - mean(max(xb'*z,0),1);
    dw = c.*dc;
    z = z - dz*hz;
    z = z./sqrt(repmat(sum(z.^2,1),d1,1));
    w = w.*exp(-hw*dw);
    if and(reg_w,mean(w)>beta); w = beta*w/mean(w); end
    wc = w.*c;
    
    
    f1_s = f1(x_test,z,wc);
    f1_s = f1_s-min(f1_s);
    KL1(i) = log(mean(exp(-f1_s+f1_t)))+mean(f1_s-f1_t);
    df1_s = df1(x_test,z,wc);
    df1_s = df1_s - repmat(sum(x_test.*df1_s,1),d1,1).*x_test;
    SM1(i) = mean(sum((df1_t-df1_s).^2,1));
    mw1(i) = mean(w);
    if mod(i,n_print)==0
        fprintf('Iteration %u: KL = %5f ; score =  %5f ; TV norm = %5f \n', i, KL1(i), SM1(i), mw1(i))
    end
   
end

%% KL, score & TV norm
fig2=figure(2);clf;
subplot(1,3,1)
semilogy(KL1,'Linewidth',2);
hold on
grid on
xlabel('SGD iteration','FontSize',16,'FontAngle','italic');
ylabel('KL div','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
subplot(1,3,2)
semilogy(SM1,'Linewidth',2);
hold on
grid on
xlabel('SGD iteration','FontSize',16,'FontAngle','italic');
ylabel('SM','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
subplot(1,3,3)
plot(mw1,'Linewidth',2);
hold on
grid on
xlabel('SGD iteration','FontSize',16,'FontAngle','italic');
ylabel('TV','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
fig2.Position = fig_size;