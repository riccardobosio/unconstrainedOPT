%Create gradient function
function gradfx=grad(x,n)
%
%INPUTS:
%x=column vector of size n;
%n=number of dimensions;
%
%OUTPUTS:
%gradf=column vector where i-th component is the partial derivative of f 
%with respect to the i-th component of x.
%
gradfx=zeros(n,1);
for i=1:n
    gradfx(i)=x(i)^3+x(i)+1;
end
end