function Anomaly=AD_lilu7(Data)
%%%%TPCA
[H,W,Dim]=size(Data);
Y=reshape(Data,[H*W,Dim]);
S=svd(Y);
E_tol=sqrt(sum(S.^2));
thre=(E_tol*0.005)^2;
np=find(S.^2<thre);
np=np(1);
Ten=[];
for j=1:W
    for i=1:H
        Temp=zeros(3,3,Dim);
        idt=i-1;
        idb=i+1;
        if i==1
            idt=H;
            idb=2;
        end
        if i==H
            idt=H-1;
            idb=1;
        end
        idl=j-1;
        idr=j+1;
        if j==1
            idl=W;
            idr=2;
        end
        if j==W
            idl=W-1;
            idr=1;
        end
        Temp(1,1,:)=Data(idt,idl,:);
        Temp(1,2,:)=Data(idt,j,:); 
        Temp(1,3,:)=Data(idt,idr,:); 
        
        Temp(2,1,:)=Data(i,idl,:);
        Temp(2,2,:)=Data(i,j,:); 
        Temp(2,3,:)=Data(i,idr,:);
        
        Temp(3,1,:)=Data(idb,idl,:);
        Temp(3,2,:)=Data(idb,j,:); 
        Temp(3,3,:)=Data(idb,idr,:); 
        
        Ten=cat(4,Ten,Temp);
    end
end
num=size(Ten,4);
X_m=mean(Ten,4);
Ten_fft=[];
for i=1:num
    temp=squeeze(Ten(:,:,:,i))-X_m;
    Temp_fft=fft(temp,[],3);
    Ten_fft=cat(4,Ten_fft,Temp_fft);
end
Ten_hat_fft=Ten_fft(:,:,np:end,:);
Ten_hat_ifft=zeros(3,3,Dim,num);
for i=1:3
    for j=1:3
        temp=squeeze(Ten_fft(i,j,:,1));
        G=temp*temp';
        for k=2:num
           temp=squeeze(Ten_fft(i,j,:,k));
           G=G+temp*temp';
        end
        G=G/(num-1);
        [u,s,v]=svd(G);
        u_eps=u(:,np:end);        
        for k=1:num
           temp=squeeze(Ten_hat_fft(i,j,:,k));
           temp=u_eps*temp;
           Ten_hat_ifft(i,j,:,k)=ifft(temp);
        end
    end
end
Anomaly=zeros(num,Dim);
for i=1:num
    for j=1:Dim
        temp=Ten_hat_ifft(:,:,j,i);
        Anomaly(i,j)=sum(temp(:));
    end
end
end