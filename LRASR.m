function [S,E]=LRASR(X,Dict,beta,lamda,display)
%%min||S||_*+beta||S||_1+lamda||E||_2,1
%%s.t. X=Dict*S+E
%%%% X DimxNum Dict DimxnumDict S numDictxNum
if nargin<5
    display = false;
end

tol1= 1e-6;
tol2= 1e-2;
maxIter = 100;%1e6;
mu=0.01;
mu_max=1e10;
ita1=1.0/((norm(Dict,2))^2);
k=0;

[dim num] = size(X);
numDict = size(Dict,2);
S=zeros(numDict,num);
J=zeros(numDict,num);
E=zeros(dim,num);
Y1=zeros(dim,num);
Y2=zeros(numDict,num);
DtX = Dict'*X;
DtD=Dict'*Dict;
iter = 0;
X_F=norm(X,'fro');
if display
    disp(['initial,rank=' num2str(rank(S))]);
end

while iter<maxIter
    iter = iter + 1;
    %updata S
    temp=S+ita1*DtX-ita1*DtD*S-ita1*Dict'*E+ita1*Dict'*Y1/mu-ita1*S+ita1*J-ita1*Y2/mu;
    S1=svd_threshold(temp,ita1/mu);
    %updata J
    temp=S1+Y2/mu;
    J1=shrink(temp,beta/mu);
    %updata E
    temp=X-Dict*S1+Y1/mu;
    E1=solve_l1l2(temp,lamda/mu);
    %updata Y1 Y2
    RES=X-Dict*S1-E1;
    Y1=Y1+mu*(RES);
    Y2=Y2+mu*(S1-J1);
    %updata mu
   ktt2=mu*max(max(sqrt(1/ita1)*norm(S1-S,'fro'),norm(J1-J,'fro')),norm(E1-E,'fro'))/X_F;
   if ktt2<tol2
       rou=1.1;
   else
       rou=1;
   end
   mu=min(mu_max,rou*mu);
   ktt1=norm(RES,'fro')/X_F;
   S=S1;
   J=J1;
   E=E1;
   if display
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(S,1e-3*norm(S,2))) ',stopC1=' num2str(ktt1,'%2.3e') ',stopC2=' num2str(ktt2,'%2.3e')]);
    end
   if(ktt1<tol1&&ktt2<tol2)
       break;
   end
end


end
function Y=svd_threshold(X,r)
[U,S,V] = svd(X, 'econ'); % stable 
 sigma = diag(S);
 svp = length(find(sigma>r));
    if svp>=1
        sigma = sigma(1:svp)-r;
    else
        svp = 1;
        sigma = 0;
    end
    Y = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
end
function Y=shrink(X,r)
Y=max(X-r,0)-max(-X-r,0);
end