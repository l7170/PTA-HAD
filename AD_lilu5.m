function [T,Anomaly]=AD_lilu5(X)
addpath(genpath('./tensor_toolbox_2.5'));
[H,W,Dim]=size(X);
X=tensor(X);
[T,~,core,U]=tucker_als(X,[H,W,Dim]);
temp_H=tenmat(core,1);
temp_H=double(temp_H.data);
idx_H=slice_determin(temp_H);
temp_W=tenmat(core,2);
temp_W=double(temp_W.data);
idx_W=slice_determin(temp_W);
temp_D=tenmat(core,3);
temp_D=double(temp_D.data);
idx_D=slice_determin(temp_D);

H_1=max(idx_H-1,1);
W_1=max(idx_W-1,1);
D_1=max(idx_D-1,1);

H_2=min(idx_H+1,H);
W_2=min(idx_W+1,W);
D_2=min(idx_D+1,Dim);

E=zeros(H_2-H_1+1,W_2-W_1+1,D_2-D_1+1);
for i=H_1:H_2
    for j=W_1:W_2
        for k=D_1:D_2
            core_new=core(1:i,1:j,1:k);
            U_new{1}=U{1}(:,1:i);
            U_new{2}=U{2}(:,1:j);
            U_new{3}=U{3}(:,1:k);
            T_an = ttensor(core_new, U_new);
            E(1+i-H_1,1+j-W_1,1+k-D_1)=energe_fun(double(T_an));
        end
    end
end
ind=find(E(:)==max(E(:)));
[x,y,z] = ind2sub([H_2-H_1+1,W_2-W_1+1,D_2-D_1+1],ind);
core_new=core(1:x+H_1-1,1:y+W_1-1,1:z+D_1-1);
U_new{1}=U{1}(:,1:x+H_1-1);
U_new{2}=U{2}(:,1:y+W_1-1);
U_new{3}=U{3}(:,1:z+D_1-1);
Anomaly = double(ttensor(core_new, U_new));
T=double(T);
end
function E=energe_fun(T_an)
G=sqrt(sum(double(T_an).^2,3));
m=filter2(fspecial('average',3),G);
E=sum(sum(G.^2./m));
end
function idx=slice_determin(slice)
temp=std(slice');
u=mean(temp);
M=max(temp);
delta=u+(M-u)*sqrt(u/M);
ind=find(temp>delta);
idx=max(ind(1)-1,1);
end