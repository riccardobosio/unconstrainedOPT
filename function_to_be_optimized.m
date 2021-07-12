%Create the function to be optimized
function y = function_to_be_optimized(x,n)
%
%INPUTS:
%x=column vector of length n;
%n=number of dimensions of x;
%OUTPUTS:
%y=real scalar value equal to f(x)
%
y=0;
for i=1:n
    y=y+(1/4*x(i)^4+1/2*x(i)^2+x(i));
end
end