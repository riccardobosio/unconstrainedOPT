%Newton method

function [xk,fk,gradfk_norm,k,xseq, btseq]=newton_method(x0,f,gradf,...
    Hess_f,alpha0,c1,kmax,tolgrad,rho,btmax)
%
%[xk,fk,gradfk_norm,k,xseq, btseq]=newton_method(x0,f,gradf,...
%Hess_f,alpha0,c1,kmax,tolgrad,rho,btmax)
%
%INPUTS:
%x0=column vector of dimension n;
%f=function R^n->R;
%gradf=gradient of f;
%Hess_f=Hessian of f;
%alpha0=the initial factor that multiplies the descent direction at each
%iteration;
%c1=factor of the Armijo condition that must be a scalar in (0,1);
%kmax=maximum number of iterations;
%tolgrad=value used as stopping criterion with respect to the norm of the
%gradient;
%rho=fixed factor, lesser than 1, used for reducing alpha0;
%btmax=maximum number of steps for updating alpha during the backtracking 
%strategy.
%
%OUTPUTS:
%xk=the last x computed by the method;
%fk=f(xk);
%gradfk_norm=norm of gradf(xk);
%k=last iteration performed;
%xseq=n-by-k matrix where the columns are the xk computed during the 
%iterations.
%btseq=1-by-k vector with the number of backtracking iterations at each 
%optimization step.
%

%Initializations
xseq=zeros(length(x0), kmax);
btseq=zeros(1, kmax);
xk=x0;
fk=f(xk);
k=0;
gradfk_norm=norm(gradf(xk));

% Function handle for the armijo condition
f_armijo=@(fk, alpha, xk, pk) fk+c1*alpha*gradf(xk)'*pk;

while k<kmax && gradfk_norm>=tolgrad
    %Compute the descent direction as solution of
    %Hess_f(xk) p = - graf(xk)
    pk=-Hess_f(xk)\gradf(xk);
    %Reset the value of alpha
    alpha=alpha0;
    %Compute the candidate new xk
    xnew=xk+alpha*pk;
    %Compute the value of f in the candidate new xk
    fnew=f(xnew);
    
    bt = 0;
    %Backtracking:
    while bt<btmax && fnew>f_armijo(fk, alpha, xk, pk)
        %Reduce the value of alpha
        alpha=rho*alpha;
        %Update xnew and fnew
        xnew=xk+alpha*pk;
        fnew=f(xnew);
        
        bt = bt + 1;
    end
    
    xk=xnew;
    fk=fnew;
    gradfk_norm=norm(gradf(xk));
    
    %Increase the step by one
    k=k+1;
    
    %Store current xk in xseq
    xseq(:,k)=xk;
    %Store bt iterations in btseq
    btseq(k)=bt;
end

%"Cut" xseq and btseq to the correct size
xseq=xseq(:,1:k);
btseq=btseq(1:k);

end