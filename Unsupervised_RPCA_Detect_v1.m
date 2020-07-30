function [result Data_tmp1, Data_tmp2] = Unsupervised_RPCA_Detect_v1(Data)
%  


[a b c] = size(Data);        
result = zeros(a, b);
DataTest = reshape(Data, a*b, c);
[L_hat S_hat iter] = inexact_alm_rpca(DataTest); % DataTest: num_sam x num_dim
Data_tmp1 = reshape(S_hat, a, b, c);
Data_tmp2 = reshape(L_hat, a, b, c);

% for i = 1: b 
%     for j = 1: a
%         y = squeeze(Data_tmp1(j, i, :));
%         result(j, i) = norm(y, 1);
%     end
% end
