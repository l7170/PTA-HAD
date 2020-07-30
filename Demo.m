clear all;
close all;
clc;
addpath(genpath('./实验数据'));
addpath(genpath('./inexact_alm_rpca'));

load Sandiego_new
load Sandiego_gt
data=hsi;
mask=hsi_gt;



f_show=data(:,:,[37,18,8]);
for i=1:3
    max_f=max(max(f_show(:,:,i)));
    min_f=min(min(f_show(:,:,i)));
    f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
end
figure,imshow(f_show);imwrite(f_show,'im.jpg');
figure,imshow(mask,[]);imwrite(mask,'gt.jpg');
DataTest=data;
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end

NBOXPLOT=zeros(H*W,8);

%%%%
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';
%%%%

%%%%PTA with LTV-norm

tol1=1e-4;
tol2=1e-6;
maxiter=400;
truncate_rank=1;
alphia=1.7;
beta=0.069;
tau=0.1;

 tic;
 [X,S,area] = AD_Tensor_LILU1(DataTest,alphia,beta,tau,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);
toc

f_show=sqrt(sum(S.^2,3));
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));

figure('name','Proposed'),imshow(f_show);imwrite(f_show,'PTA.jpg');
NBOXPLOT(:,1)=f_show(:);

r_max = max(f_show(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (f_show(:)> tau)';
  PF0(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD0(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area03=sum((PF0(1:end-1)-PF0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);
%%%%

%%%%PTA with anisotropic TV-norm

% tol1=1e-4;
% tol2=1e-3;
% maxiter=29;
% truncate_rank=34;
% alphia=2.76;%2.76;
% beta=0.17;%1/sqrt(H*W);
% tau=0.009;
% tic;
%    [X,S,area] = AD_Tensor_LILU3(DataTest,alphia,beta,tau,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);
% toc
% 
% f_show=sqrt(sum(S.^2,3));
% f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% 
% %figure('name','Proposed'),imshow(f_show);imwrite(f_show,'PTR.jpg');
% NBOXPLOT(:,1)=f_show(:);
% 
% r_max = max(f_show(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (f_show(:)> tau)';
%   PF02(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD02(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area02=sum((PF02(1:end-1)-PF02(2:end)).*(PD02(2:end)+PD02(1:end-1))/2);
% 
% %%%%PTA with isotropic TV-norm
% 
% tol1=1e-4;
% tol2=1e-3;
% maxiter=400;
% truncate_rank=10;
% alphia=0.01;%2.76;
% beta=172;%1/sqrt(H*W);
% tau=10;
% tic;
%    [X,S,area] = AD_Tensor_LILU2(DataTest,alphia,beta,tau,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);
% toc
% 
% f_show=sqrt(sum(S.^2,3));
% f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
% 
% %figure('name','Proposed'),imshow(f_show);imwrite(f_show,'PTR.jpg');
% NBOXPLOT(:,2)=f_show(:);
% 
% r_max = max(f_show(:));
% taus = linspace(0, r_max, 5000);
% for index2 = 1:length(taus)
%   tau = taus(index2);
%   anomaly_map_rx = (f_show(:)> tau)';
%   PF01(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
%   PD01(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
% end
% area01=sum((PF01(1:end-1)-PF01(2:end)).*(PD01(2:end)+PD01(1:end-1))/2);
% 
% 
% figure,
% plot(PF01, PD01, 'b-', 'LineWidth', 2);hold on;
% plot(PF02, PD02, 'r-', 'LineWidth', 2);  
% plot(PF03, PD03, 'g-', 'LineWidth', 2);  
% hold off;
% xlabel('False alarm rate'); ylabel('Probability of detection');
% legend('anisotropic','isotropic','LTV-norm');
% axis([0 0.1 0 1]);hold off;
% 
% figure,
% semilogx(PF01, PD01, 'b-', 'LineWidth', 2);hold on;
% semilogx(PF02, PD02, 'r-', 'LineWidth', 2);  
% semilogx(PF03, PD03, 'g-', 'LineWidth', 2);  
% hold off;
% xlabel('False alarm rate'); ylabel('Probability of detection');
% legend('anisotropic','isotropic','LTV-norm');
% axis([0 0.1 0 1]);hold off;

%%%%%%%GTVLRR
tic;
Dict=ConstructionD_lilu(Y,15,20);
lambda = 0.5;
beta = 0.2;
gamma =0.05;% 0;%
display = true;
[X,S] = lrr_tv_manifold(Y,Dict,lambda,beta,gamma,[H,W],display);
toc
r_gtvlrr=sqrt(sum(S.^2));
r_max = max(r_gtvlrr(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_gtvlrr > tau);
  PF_gtvlrr(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD_gtvlrr(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r_gtvlrr,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','GTVLRR'), imshow(f_show);imwrite(f_show,'GTVLRR.jpg');

area_GTVLRR = sum((PF_gtvlrr(1:end-1)-PF_gtvlrr(2:end)).*(PD_gtvlrr(2:end)+PD_gtvlrr(1:end-1))/2);
NBOXPLOT(:,8)=f_show(:);


% %%%%%%LRASR

beta=0.1;
lamda=1;
tic;
[S,E]=LRASR(Y,Dict,beta,lamda,1);
toc
r_new=sqrt(sum(E.^2,1));
r_max = max(r_new(:));
taus = linspace(0, r_max, 5000);
PF_40=zeros(1,5000);
PD_40=zeros(1,5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_new> tau);
  PF_40(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD_40(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_LRASR = sum((PF_40(1:end-1)-PF_40(2:end)).*(PD_40(2:end)+PD_40(1:end-1))/2);
f_show=reshape(r_new,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','LRASR'), imshow(f_show);imwrite(f_show,'LRASR.jpg');
NBOXPLOT(:,4)=f_show(:)';



%%%%%%
tic;
[T_all,Anomaly]=AD_lilu5(DataTest);
toc
Y0=reshape(Anomaly, num, Dim)';
r4 = RX(Y0);  % input: num_dim x num_sam    rx
r_max = max(r4(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r4 > tau);
  PF4(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD4(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r4,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','TDAD'), imshow(f_show);imwrite(f_show,'TDAD.jpg');
area_TDAD = sum((PF4(1:end-1)-PF4(2:end)).*(PD4(2:end)+PD4(1:end-1))/2);
NBOXPLOT(:,6)=f_show(:);


tic;
Anomaly=AD_lilu7(DataTest);
toc
r5 = RX(abs(Anomaly)');  % input: num_dim x num_sam    rx
r_max = max(r5(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r5 > tau);
  PF7(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD7(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r5,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','TPCA'), imshow(f_show);imwrite(f_show,'TPCA.jpg');
area_TPCA = sum((PF7(1:end-1)-PF7(2:end)).*(PD7(2:end)+PD7(1:end-1))/2);
NBOXPLOT(:,7)=f_show(:);


tic;
r3 = RX(Y);  % input: num_dim x num_sam    rx
toc
r_max = max(r3(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r3 > tau);
  PF3(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD3(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r3,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','RX'), imshow(f_show);imwrite(f_show,'RX.jpg');
area_RX = sum((PF3(1:end-1)-PF3(2:end)).*(PD3(2:end)+PD3(1:end-1))/2);
NBOXPLOT(:,2)=f_show(:);
% 
% LSMAD
tic;
[L,S,RMSE,error]=GoDec(Y',28,floor(0.0022*Dim)*9,2);
toc
L=L';
S=S';

mu=mean(L,2);
r_new2=(diag((Y-repmat(mu,[1,num]))'*pinv(cov(L'))*(Y-repmat(mu,[1,num]))))';

r_max = max(r_new2(:));
taus = linspace(0, r_max, 5000);
PF_41=zeros(1,5000);
PD_41=zeros(1,5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_new2> tau);
  PF_41(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD_41(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_LSMAD = sum((PF_41(1:end-1)-PF_41(2:end)).*(PD_41(2:end)+PD_41(1:end-1))/2);
f_show=reshape(r_new2,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','LSMAD'), imshow(f_show);imwrite(f_show,'LSMAD.jpg');
NBOXPLOT(:,5)=f_show(:);

%%%rpca   rx
tic;
[r0 ,Output_S, Output_L] = Unsupervised_RPCA_Detect_v1(DataTest);
toc
XS = reshape(Output_S, num, Dim);
r2 = RX(XS');  % input: num_dim x num_sam

r_max = max(r2(:));                                              
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r2 > tau);
  PF2(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD2(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r2,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','RPCA-RX'), imshow(f_show);imwrite(f_show,'RPCA-RX.jpg');
NBOXPLOT(:,3)=f_show(:);
area_RPCA = sum((PF2(1:end-1)-PF2(2:end)).*(PD2(2:end)+PD2(1:end-1))/2);



figure,
plot(PF3, PD3, 'b-', 'LineWidth', 2);hold on;
plot(PF2, PD2, 'r-', 'LineWidth', 2);  
plot(PF_41, PD_41, 'y-', 'LineWidth', 2);
plot(PF_40, PD_40, 'm-', 'LineWidth', 2);
plot(PF4, PD4, 'c-', 'LineWidth', 2);
plot(PF7, PD7, 'k-', 'LineWidth', 2);
plot(PF_gtvlrr, PD_gtvlrr, 'Color',[0.5,0.5,1], 'LineWidth', 2);
plot(PF0, PD0, 'g-', 'LineWidth', 2);  
hold off;
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('GRXD','RPCA-RX','LSMAD','LRASR','TDAD','TPCA','GTVLRR','PTA');
axis([0 0.1 0 1]);hold off;

figure, 
semilogx(PF3, PD3, 'b-', 'LineWidth', 3); hold on;
semilogx(PF2, PD2, 'r-', 'LineWidth', 2);  
semilogx(PF_41, PD_41, 'y-', 'LineWidth', 2);
semilogx(PF_40, PD_40, 'm-', 'LineWidth', 2);
semilogx(PF4, PD4, 'c-', 'LineWidth', 2);
semilogx(PF7, PD7, 'k-', 'LineWidth', 2);
semilogx(PF_gtvlrr, PD_gtvlrr, 'Color',[0.5,0.5,1], 'LineWidth', 2);
semilogx(PF0, PD0, 'g-', 'LineWidth', 4);
hold off;
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('GRXD','RPCA-RX','LSMAD','LRASR','TDAD','TPCA','GTVLRR','PTA');
axis([0 1 0 1]);hold off;
