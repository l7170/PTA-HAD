function D = RX(X)

[N M] = size(X);

X_mean = mean(X,2);

X = X - repmat(X_mean, [1 M]);

Sigma = (X * X')/(M-1);

Sigma_inv = inv(Sigma);
D=zeros(1,M);
for m = 1:M
 D(m) = X(:, m)' * Sigma_inv * X(:, m);
end
