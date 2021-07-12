%Inexact Newton method
function [xk,fk,gradfk_norm,k,xseq,btseq,pcg_iter]=...
    inexact_newton_method(x0,f,gradf,Hess_f,...
    alpha0,kmax,tolgrad,c1,rho,btmax,...
    type_fterms,max_pcgiters)
%
%function [xk,fk,grafk_norm,k,xseq,btseq]=...
%   inexact_newton_method(x0,f,gradf,Hess_f,...
%   alpha0,kmax,tolgrad,c1,rho,btmax,...
%   fterms,maxpcgiters)
%
%INPUTS:
%x0=column vector of dimension n;
%f=function R^n->R;
%gradf=gradient of f;
%Hess_f=Hessian of f;
%alpha0=initial factor that multiplies the descent direction at each 
%iteration;
%kmax=maximum number of iterations;
%tolgrad=value used as stopping criterion;
%c1=scalar factor of the Armijo condition, it must be in (0,1);
%rho=fixed factor used to reduce alpha, lesser than 1;
%btmax=maximum number of steps of backtracking strategy;
%type_fterms='l','s'or'q', string that specifies the type of forcing terms;
%max_pcgiters=maximum number of iterations for the pcg solver to compute
%pk.
%
%OUTPUTS:
%xk=the last x computed by the method;
%fk=f(xk);
%grafk_norm=norm of gradf(xk);
%k=last iteration;
%xseq=n-by-k matrix where the columns are the xk;
%btseq=1-by-k vector where elements are the number of backtracking
%iterations at each optimization step.
%pcg_iter=number of iterations of pcg
%

%Initializations
xseq=zeros(length(x0),kmax);
btseq=zeros(1,kmax);
pcg_iter=zeros(1,kmax);
xk=x0;
fk=f(xk);
k=0;
gradfk_norm=norm(gradf(xk));

%Function handle for the Armijo condition
f_armijo=@(fk,alpha,xk,pk) fk+c1*alpha*gradf(xk)'*pk;

while k<kmax && gradfk_norm>=tolgrad
    %Compute the descent direction as solution of
    %Hessf(xk) p = - graf(xk)
    eta_k=0;
    switch type_fterms
        case 'l'
            eta_k=0.5;
        case 's'
            eta_k=min(0.5,sqrt(norm(gradf(xk))));
        case 'q'
            eta_k=min(0.5,norm(gradf(xk)));
        otherwise
            eta_k=0.5;
    end
    
    %Tolerance varying with respect to forcing terms
    epsilon_k=eta_k*norm(gradf(xk));
    [pk,~,~,iter]=pcg(Hess_f(xk),...
          -gradf(xk),epsilon_k,max_pcgiters);
    pcg_iter(k+1)=iter;
    
    %Reset the value of alpha
    alpha=alpha0;
    
    %Compute the candidate new xk
    xnew=xk+alpha*pk;
    
    %Compute f in the candidate new xk
    fnew=f(xnew);
    
    %Use backtracking strategy
    bt=0;
    while bt<btmax && fnew>f_armijo(fk,alpha,xk,pk)
        %Reduce alpha
        alpha=rho*alpha;
        
        %Update xnew and fnew
        xnew=xk+alpha*pk;
        fnew=f(xnew);
        
        bt=bt+1;
    end
    
    %Update xk,fk,gradfk_norm
    xk=xnew;
    fk=fnew;
    gradfk_norm=norm(gradf(xk));
    
    k=k+1;
    
    %Store xk in xseq
    xseq(:,k)=xk;
    
    %Store bt in btseq
    btseq(k)=bt;
end

%Resize xseq and btseq
xseq=xseq(:,1:k);
btseq=btseq(1:k);
pcg_iter=pcg_iter(1:k);

end