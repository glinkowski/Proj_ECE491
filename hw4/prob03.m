
%% Part 1
x = [-2:0.2:2];
y = [-2:0.2:2];

X = repmat(x, numel(x), 1);
Y = repmat(y, numel(y), 1)';

Z = X.^2 - 4.*X.*Y + Y.^2;
figure();
surf(Z)

%% Part 2
x = [-2:0.2:2];
y = [-2:0.2:2];

X = repmat(x, numel(x), 1);
Y = repmat(y, numel(y), 1)';

Z = X.^4 - 4.*X.*Y + Y.^4;
figure();
surf(Z)

%% Part 3
x = [-2:0.2:2];
y = [-2:0.2:2];

X = repmat(x, numel(x), 1);
Y = repmat(y, numel(y), 1)';

Z = 2.*X.^3 - 3.*X.^2 - 6.*X.*Y.*(X - Y - 1);
figure();
surf(Z)

%% Part 4
x = [-5:0.2:5];
y = [-5:0.2:5];

X = repmat(x, numel(x), 1);
Y = repmat(y, numel(y), 1)';

Z = (X - Y)^4 + X.^2 - Y.^2 - 2*X + 2*Y + 1;
figure();
surf(Z)