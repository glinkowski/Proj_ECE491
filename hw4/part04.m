close all;

%% Part 1
x = [-5:0.2:5];
y = [-5:0.2:5];

X = repmat(x, numel(x), 1);
Y = repmat(y, numel(y), 1)';

% Unsure if this maps out the function quite right
Z = ( ((X.^2) - Y).^2 + (1 - X).^2 ) / 2;
figure();
surf(Z)

%X = [-1,0,1,2]; Y = [-1,0,1,2];
%( ((X.^2) - Y).^2 + (1 - X).^2 ) / 2
%X = 1; Y = 1;
%( ((X.^2) - Y).^2 + (1 - X).^2 ) / 2

%% Part 2

syms x1 x2;
gradF(x1, x2) = [ (2*x1^3 + x1 - 1 - 2*x1*x2) ; (-x1^2 + x2) ];
Hf(x1, x2) = [(6*x1 + 1 - 2*x2), 2*x1; -2*x1, 1];

x0 = [2;2];
x1 = x0 - inv( Hf(x0(1), x0(2)) )*gradF(x0(1),x0(2));
double(x1)

s = linsolve( Hf(x0(1), x0(2)), -gradF(x0(1),x0(2)) );
xk = x0 + s;
double(xk)

for n = 1:10
    s = linsolve( Hf(xk(1), xk(2)), -gradF(xk(1),xk(2)) );
    xk = xk + s;
end
double(xk)


