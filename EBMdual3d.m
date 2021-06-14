%% SGD with unbalanced transport for KL EBM 
%
% E(x) = \sum_i w_i c_i (x,z_i)_+
%
% with |x|=|z_i|=1, c_i \in R, w_i \ge 0.
%
% Expectations over the data (teacher) computed with the full data set
% Expectations over the student computed on evolving walkers
% Unabalanced transport in WFR -- z_i & c_i evolved, c_i fixed
%
% SGD performed with full data set
% Number of walkers = twice size of data set

%% options & input parameters
% clear all;
reset_data = true;                                 % reset training data set
stop_restart = false;                               % stop restart after Nt/2 iterations
reg_w = true;                                      % enforce |w|<beta
vid1 = true;                                       % video output, 0 otherwise
scr_siz = get(0,'ScreenSize') ;
fig_size = floor([.1*scr_siz(3) .1*scr_siz(4) .9*scr_siz(3) .9*scr_siz(4)]);

d1 = 3;                                         % input dimension (needs to be 3 right now for the graphics)
 
Nt = 1e4;                                       % number of step of training
N_restart = Nt;                                 % restart walkers every N_restart (if N_restart=1 => SM)
hx = 2e-2;                                      % time step for walkers x
alph = (1+.1/N_restart/hx);
hz = .04*alph;                                        % time step for features z (set to zero for lazy training)
hw = .02*alph;                                        % time step for weight w (unbalanced transport)
n_plot = 5;

n_t = 2;                                        % number of teachers
n_unit = 2^6;                                   % number of students
n_data = 1e2;                                   % training data sise
n_walker = 2e2;                                 % number of walkers

if vid1 
    writerObj = VideoWriter('KLdualmonomodlowdata2.mp4','MPEG-4');
    writerObj.FrameRate = 4;
    open(writerObj);
end

%% energy function & gradient wrt x
f1 = @(x,z,c) mean(max(x'*z,0).*(repmat(c,size(x,2),1)),2);
df1 = @(x,z,c) z*(repmat(c',1,size(x,2)).*(z'*x>0))/size(z,2);

%% teacher
beta = 10;
ct = -beta*ones(1,n_t);
% zt = [0;1;0];
% zt = randn(d1,n_t);
% zt(:,1) = [1;.8;0];
% zt(:,2) = [-1;.8;0];
% zt = [-0.4907,  0.4253;0.7621, -0.3558;0.4224, -0.8321];
zt = [-0.1685   -0.3475; 0.9065    0.3090; -0.3871    0.8853];
zt = zt./sqrt(repmat(sum(zt.^2,1),d1,1));

KL1 = zeros(Nt,1);
SM1 = zeros(Nt,1);
mw1 = zeros(Nt,1);


%% data set created by rejection method using box optimized to min f(x)
if reset_data
    dt = 1e-2;
    tol = 1e-8;
    norm1 = 1;
    n_gd = 1e5;
    n_trial = 1e8;
    
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

%% initial weigths, features, walkers 
if reset_data
    z = randn(d1,n_unit);
    z = z./sqrt(repmat(sum(z.^2,1),d1,1));
    c = [-ones(1,n_unit/2),-ones(1,n_unit/2)]; % pick positive vs negative units
    w = ones(1,n_unit);
else
    z = z0; c = c0; w = w0;
end
z0 = z; c0 = c;   w0 = w;                              % save initial values
wc = w.*c;
% xb1 = xt(:,randi(n_data,n_walker/2,1));
xb1 = xt;
xb = [xb1,xb1];

%% variables for graphical output
n1 = 1e2;
theta1 = linspace(0,pi,n1);
phi1 = linspace(0,2*pi,n1);
[theta,phi] = meshgrid(theta1,phi1);

x2 = sin(theta).*cos(phi);
y2 = sin(theta).*sin(phi);
z2 = cos(theta);

f_t = 0*z2;
for k = 1:n_t
    f_t = f_t+ct(k)*max(x2*zt(1,k)+y2*zt(2,k)+z2*zt(3,k),0)/n_t;
end
f_t = f_t - min(f_t(:));

%% initial KL, score, and TV norm
theta3 = reshape(theta,1,n1*n1);
phi3 = reshape(phi,1,n1*n1);

x3 = sin(theta3).*cos(phi3);
y3 = sin(theta3).*sin(phi3);
z3 = cos(theta3);

f3_t = f1([x3;y3;z3],zt,ct)';
f3_t = f3_t-min(f3_t);
Z3_t = mean(sin(theta3).*exp(-f3_t));
lZ3_t = log(Z3_t);
f3_s = f1([x3;y3;z3],z,wc)';
f3_s = f3_s-min(f3_s);
KL1(1) = log(mean(sin(theta3).*exp(-f3_s))) - lZ3_t ...
    + mean(sin(theta3).*(f3_s-f3_t).*exp(-f3_t))/Z3_t;

df3_t = df1([x3;y3;z3],zt,ct);
df3_t = df3_t - repmat(sum([x3;y3;z3].*df3_t,1),d1,1).*[x3;y3;z3];
df3_s = df1([x3;y3;z3],z,wc);
df3_s = df3_s - repmat(sum([x3;y3;z3].*df3_s,1),d1,1).*[x3;y3;z3];
SM1(1) = mean(sin(theta3).*sum((df3_t-df3_s).^2,1).*exp(-f3_t))/Z3_t;

mw1(1) = mean(w);

%% initial figure

fig1 = figure(1);clf;
subplot(1,2,1)
s = surf(x2,y2,z2,exp(-f_t));
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',16);
cm = max(abs(wc.*sqrt(sum(z.^2,1))));
zn = z./sqrt(repmat(sum(z.^2,1),d1,1));
for j = 1:n_unit
    cs = 1+abs(wc(j)*norm(z(:,j)))/cm;
    if c(j)>0
        plot3([0 cs*zn(1,j)],[0 cs*zn(2,j)],[0 cs*zn(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*zn(1,j)],[0 cs*zn(2,j)],[0 cs*zn(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title('Teacher density')
drawnow
f_s = 0*z2;
for k =1:n_unit
    f_s = f_s + wc(k)*max(x2*z(1,k)+y2*z(2,k)+z2*z(3,k),0);
end
f_s = f_s/n_unit;
f_s = f_s - min(f_s(:));
f0_s = f_s;
f_s(1,1) = max(f_t(:));
drawnow
subplot(1,2,2)
s = surf(x2,y2,z2,exp(-f_s));
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',16);
cm = max(abs(wc.*sqrt(sum(z.^2,1))));
zn = z./sqrt(repmat(sum(z.^2,1),d1,1));
for j = 1:n_unit
    cs = 1+abs(wc(j)*norm(z(:,j)))/cm;
    if c(j)>0
        plot3([0 cs*zn(1,j)],[0 cs*zn(2,j)],[0 cs*zn(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*zn(1,j)],[0 cs*zn(2,j)],[0 cs*zn(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title(['Student density at iteration ',num2str(0)])
fig1.Position = fig_size;
%%
if vid1
    frame = getframe(gcf);
    writeVideo(writerObj,frame);
end

%%  training   
for i = 1:Nt
    if i > 2000
        n_plot = 500;
    elseif i > 200
        n_plot = 1e2;
    elseif i > 40
        n_plot = 2e1;
    end

    % walkers
    if mod(i,N_restart)==0; xb = [xt,xt]; end
    if and(stop_restart==1,i>Nt/2); N_restart = Nt; end
    dx = df1(xb,z,wc);
%     dx = dx - repmat(sum(xb.*dx,1),d1,1).*xb;
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
    
    f3_s = f1([x3;y3;z3],z,wc)';
    f3_s = f3_s-min(f3_s);
    KL1(i) = log(mean(sin(theta3).*exp(-f3_s))) - lZ3_t ...
    + mean(sin(theta3).*(f3_s-f3_t).*exp(-f3_t))/Z3_t;
    
    df3_s = df1([x3;y3;z3],z,wc);
    df3_s = df3_s - repmat(sum([x3;y3;z3].*df3_s,1),d1,1).*[x3;y3;z3];
    SM1(i) = mean(sin(theta3).*sum((df3_t-df3_s).^2,1).*exp(-f3_t))/Z3_t;
    
    mw1(i) = mean(w);
    
    if mod(i,n_plot)==0
        fprintf('Iteration %u: KL = %5f ; score =  %5f ; TV norm = %5f \n', i, KL1(i), SM1(i), mw1(i))
        figure(1);clf;
        subplot(1,2,1)
        s = surf(x2,y2,z2,exp(-f_t));
        s.EdgeColor = 'none';
        colorbar
        hold on
        axis equal
        axis off
        xlim([-1.3,1.3])
        ylim([-1.3,1.3])
        zlim([-1.3,1.3])
        hold on
        for k=1:n_t
            plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
        end
        view([180,-20])
        set(gca,'FontSize',16);
        cm = max(abs(wc.*sqrt(sum(z.^2,1))));
        zn = z./sqrt(repmat(sum(z.^2,1),d1,1));
        for j = 1:n_unit
            cs = 1+abs(wc(j)*norm(z(:,j)))/cm;
            if wc(j)>0
                plot3([0 cs*zn(1,j)],[0 cs*zn(2,j)],[0 cs*zn(3,j)],'Color',[1,.5,0], 'Linewidth',2)
            else
                plot3([0 cs*zn(1,j)],[0 cs*zn(2,j)],[0 cs*zn(3,j)],'Color',[0,.5,1], 'Linewidth',2)
            end
        end
        title('Teacher density')
        drawnow
        f_s = 0*z2;
        for k =1:n_unit
            f_s = f_s + wc(k)*max(x2*z(1,k)+y2*z(2,k)+z2*z(3,k),0);
        end
        f_s = f_s/n_unit;
        f_s = f_s - min(f_s(:));
        f_s(1,1) = max(f_t(:));
        subplot(1,2,2)
        s = surf(x2,y2,z2,exp(-f_s));
        s.EdgeColor = 'none';
        colorbar
        hold on
        axis equal
        axis off
        xlim([-1.3,1.3])
        ylim([-1.3,1.3])
        zlim([-1.3,1.3])
        hold on
        for k=1:n_t
            plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
        end
        view([180,-20])
        set(gca,'FontSize',16);
        cm = max(abs(wc.*sqrt(sum(z.^2,1))));
        zn = z./sqrt(repmat(sum(z.^2,1),d1,1));
        for j = 1:n_unit
            cs = 1+abs(wc(j)*norm(z(:,j)))/cm;
            if wc(j)>0
                plot3([0 cs*zn(1,j)],[0 cs*zn(2,j)],[0 cs*zn(3,j)],'Color',[1,.5,0], 'Linewidth',2)
            else
                plot3([0 cs*zn(1,j)],[0 cs*zn(2,j)],[0 cs*zn(3,j)],'Color',[0,.5,1], 'Linewidth',2)
            end
        end
        title(['Student density at iteration ',num2str(i)])
        drawnow
        if vid1
            frame = getframe(gcf); 
            writeVideo(writerObj,frame); 
        end
    end
end

if vid1; close(writerObj); end

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

%% density and energy
fig3=figure(3);clf;
subplot(1,2,1)
s = surf(x2,y2,z2,f_t);
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',16);
cm = max(abs(wc));
for j = 1:n_unit
    cs = 1+abs(wc(j))/cm;
    if c(j)>0
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title('Teacher energy ')
drawnow
f_s = 0*z2;
for k =1:n_unit
    f_s = f_s + wc(k)*max(x2*z(1,k)+y2*z(2,k)+z2*z(3,k),0);
end
f_s = f_s/n_unit;
drawnow
subplot(1,2,2)
s = surf(x2,y2,z2,f_s);
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',16);
cm = max(abs(wc));
for j = 1:n_unit
    cs = 1+abs(wc(j))/cm;
    if c(j)>0
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title('Student energy')
drawnow
fig3.Position = fig_size;

fig4=figure(4);clf;
subplot(1,2,1)
s = surf(x2,y2,z2,exp(-f_t));
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',24);
cm = max(abs(wc));
for j = 1:n_unit
    cs = 1+abs(wc(j))/cm;
    if c(j)>0
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title('Teacher density')
drawnow
f_s = 0*z2;
for k =1:n_unit
    f_s = f_s + wc(k)*max(x2*z(1,k)+y2*z(2,k)+z2*z(3,k),0);
end
f_s = f_s/n_unit;
f_s = f_s - min(f_s(:));
f_s(1,1) = max(f_t(:));
drawnow
subplot(1,2,2)
s = surf(x2,y2,z2,exp(-f_s));
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',24);
cm = max(abs(wc));
for j = 1:n_unit
    cs = 1+abs(wc(j))/cm;
    if c(j)>0
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[1,.5,0], 'Linewidth',2)
    else 
        plot3([0 cs*z(1,j)],[0 cs*z(2,j)],[0 cs*z(3,j)],'Color',[0,.5,1], 'Linewidth',2)
    end
end
title('Student density')
drawnow
fig4.Position = fig_size;

%% error on energy
fig5=figure(5);clf;
subplot(1,2,1)
f40 = 0*z2;
for k =1:n_t
    f40 = f40 + ct(k)*max(x2*zt(1,k)+y2*zt(2,k)+z2*zt(3,k),0);
end
f40 = f40/n_t;
f40 = f40 - min(f40(:));
f4 = 0*z2;
for k =1:n_unit
    f4 = f4 + wc(k)*max(x2*z(1,k)+y2*z(2,k)+z2*z(3,k),0);
end
f4 = f4/n_unit;
f4 = f4 - min(f4(:));
df4 = f4-f40;
df4 = df4-min(df4(:));
s = surf(x2,y2,z2,df4);
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',16);
title('Error in student energy ')
subplot(1,2,2);
df4(1,1) = max(f_t(:));
s = surf(x2,y2,z2,df4);
s.EdgeColor = 'none';
colorbar
hold on
axis equal
axis off
xlim([-1.3,1.3])
ylim([-1.3,1.3])
zlim([-1.3,1.3])
hold on
for k=1:n_t
    plot3([0 1.2*zt(1,k)],[0 1.2*zt(2,k)],[0 1.2*zt(3,k)],'k', 'Linewidth',4)
end
view([180,-20])
set(gca,'FontSize',16);
title('Error in student on teacher energy scale')
fig5.Position = fig_size;
