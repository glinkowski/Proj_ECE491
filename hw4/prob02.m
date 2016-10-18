
syms x y
s = 3.5;

f0(x, y) = [ (-x / tan(x) - y); (s^2 - x^2 - y^2)];
Jf0(x,y) = jacobian(f0(x,y), [x, y])

f1(x, y) = [ (-1/(x * tan(x)) - x^(-2) + y^(-1) + y^(-2)); (s^2 - x^2 - y^2)];
Jf1(x,y) = jacobian(f1(x,y), [x, y])