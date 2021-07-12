
clear all;
clc;
%
%
%Unconstrained optimization
%3)Inexact Newton method
%
%

%Initializations
n=10^4; %We run U3 also with n=10^5
x0=zeros(n,1);
alpha=1;
kmax=100;
tolgrad=1e-12;
alpha0=alpha;
c1=1e-4;
rho=0.8;
btmax=50;
max_pcgiters=50;

%Create the function f
f=@(x) function_to_be_optimized(x,n);

%Create the gradient function
gradf=@(x) grad(x,n);

%Create the hessian function
Hess_f=@(x) diag(sparse(3.*x(:,1).^2+1)); 

%Run newton method
tic
[xk,fk,gradfk_norm,k,xseq,btseq]=...
    newton_method(x0,f,gradf,...
    Hess_f,alpha,kmax,...
    tolgrad,c1,rho,btmax);
toc
display('Number of iterations in the Newton method:')
display(k)
display('Value of fk:')
display(fk)
display('Number of inner iterations:')
display(btseq)


%Run inexact newton method
%Linear convergence
tic
[xk,fk,gradfk_norm,k,xseq,btseq,pcg_iter]=...
    inexact_newton_method(x0,f,...
    gradf,Hess_f,alpha0,kmax,tolgrad,c1,...
    rho,btmax,'l',max_pcgiters);
toc
display('Number of iterations in the inexact Newton method:') 
display('with linear convergence of forcing terms:')
display(k)
display('Value of fk:')
display(fk)
display('Number of inner iterations:')
display(btseq)
display(pcg_iter)

%Superinear convergence
tic
[xk,fk,gradfk_norm,k,xseq,btseq,pcg_iter]=...
    inexact_newton_method(x0,f,...
    gradf,Hess_f,alpha0,kmax,tolgrad,c1,...
    rho,btmax,'s',max_pcgiters);
toc
display('Number of iterations in the inexact Newton method:') 
display('with superlinear convergence of forcing terms:')
display(k)
display('Value of fk:')
display(fk)
display('Number of inner iterations:')
display(btseq)
display(pcg_iter)

%Quadratic convergence
tic
[xk,fk,gradfk_norm,k,xseq,btseq,pcg_iter]=...
    inexact_newton_method(x0,f,...
    gradf,Hess_f,alpha0,kmax,tolgrad,c1,...
    rho,btmax,'q',max_pcgiters);
toc
display('Number of iterations in the inexact Newton method:') 
display('with quadratic convergence of forcing terms:')
display(k)
display('Value of fk:')
display(fk)
display('Number of inner iterations:')
display(btseq)
display(pcg_iter)
