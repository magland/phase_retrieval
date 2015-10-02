function f=ap1d(xx,u,opts)

if (nargin==0) test_ap1d; return; end;

alpha1=opts.mu1/opts.beta;
alpha2=opts.mu2*alpha1/(opts.mu1+opts.mu2*alpha1);
if (isfield(opts,'f_ref'))
    fhat_ref=fftb(opts.f_ref);
end;

%Start with a random real function 
g_current=rand(size(u));
if (isfield(opts,'f_init')) g_current=opts.f_init; end;

resid1=ones(1,opts.numit)*inf;
resid2=ones(1,opts.numit)*inf;
residsum=ones(1,opts.numit)*inf;

fig_timer=tic;
for it=1:opts.numit

    f_current=pi_1(g_current,xx,alpha1);
    ghat_next=pi_2(fftb(f_current),u,alpha2);
    g_next=real(ifftb(ghat_next));

    resid1(it)=opts.mu1*sqrt(sum(abs(g_current-pi_1(g_current,xx,1)).^2));
    resid2(it)=opts.mu2*sqrt(sum(abs(g_current-ifftb(pi_2(fftb(g_current),u,1))).^2));
    residsum(it)=resid1(it)+resid2(it);
    if (it>1)&&(residsum(it)>residsum(it-1))
        fprintf('%g>%g\n',residsum(it),residsum(it-1));
        error('Unexpected increase in residual');
    end;

    if ((isfield(opts,'fig')) && ((it==1)||(it==opts.numit)||(toc(fig_timer)>0.1)) )

        figure(opts.fig);
        subplot(1,4,1);
        plot(xx,g_next,'b',xx,opts.f_ref,'r'); hold on;
        hold off;
        title('Recon');
        subplot(1,4,2);
        semilogy(1:opts.numit,residsum,'k',1:opts.numit,resid1,'b',1:opts.numit,resid2,'r',2:opts.numit,1e3*abs(diff(residsum)),'m');
        legend('resid sum','resid 1','resid 2','1000 x diff resid sum');
        title('Residuals');
        if (isfield(opts,'f_ref'))
            subplot(1,4,3);
            phase_diff=unwrap(angle(ghat_next.*conj(fhat_ref)));
            coeffs=polyfitweighted(xx,phase_diff,1,abs(fhat_ref));
            phase_diff=phase_diff-(coeffs(1)*xx+coeffs(2));
            phase_diff=angle(exp(i*phase_diff));
            plot(xx,phase_diff,'b',xx,phase_diff,'b.');
            title('Phase error');
            
            if (it>2000)
                %disp('debug');
            end;
            
            subplot(1,4,4);
            semilogx(abs(fhat_ref),phase_diff,'b.');
            title('Phase error vs fhat mag');
        end;
        
        drawnow;
        fig_timer=tic;

    end;

    g_current=g_next;

end

f=g_current;

end

function f_new=pi_1(f,xx,alpha1)

mask1=(f>=0)&(abs(xx)<=1);

f_new=alpha1*f.*mask1 + (1-alpha1)*f;

end

function fhat_new=pi_2(fhat,u,alpha2)

ph=angle(fhat);
fhat_new=alpha2*u.*exp(i*ph) + (1-alpha2)*fhat;

end

function Y=fftb(X)
Y=fftshift(fft(ifftshift(X)));
end

function Y=ifftb(X)
Y=fftshift(ifft(ifftshift(X)));
end




function Y=create_gaussian(xx,sigma)
Y=exp(-(xx/sigma).^2);
end

function p = polyfitweighted(x,y,n,w)
% polyfitweighted.m 
% -----------------
%
% Find a least-squares fit of 1D data y(x) with an nth order 
% polynomial, weighted by w(x).
%
% By S.S. Rogers (2006), based on polyfit.m by The MathWorks, Inc. - see doc
% polyfit for more details.
%
% Usage
% -----
%
% P = polyfitweighted(X,Y,N,W) finds the coefficients of a polynomial 
% P(X) of degree N that fits the data Y best in a least-squares sense. P 
% is a row vector of length N+1 containing the polynomial coefficients in
% descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). W is
% a vector of weights. 
%
% Vectors X,Y,W must be the same length.
%
% Class support for inputs X,Y,W:
%    float: double, single
%

% The regression problem is formulated in matrix format as:
%
%    yw = V*p    or
%
%          3    2
%    yw = [x w  x w  xw  w] [p3
%                            p2
%                            p1
%                            p0]
%
% where the vector p contains the coefficients to be found.  For a
% 7th order polynomial, matrix V would be:
%
% V = [w.*x.^7 w.*x.^6 w.*x.^5 w.*x.^4 w.*x.^3 w.*x.^2 w.*x w];

if ~isequal(size(x),size(y),size(w))
    error('X and Y vectors must be the same size.')
end

x = x(:);
y = y(:);
w = w(:);


% Construct weighted Vandermonde matrix.
%V(:,n+1) = ones(length(x),1,class(x));
V(:,n+1) = w;
for j = n:-1:1
   V(:,j) = x.*V(:,j+1);
end

% Solve least squares problem.
[Q,R] = qr(V,0);
ws = warning('off','all'); 
p = R\(Q'*(w.*y));    % Same as p = V\(w.*y);
warning(ws);
if size(R,2) > size(R,1)
   warning('polyfitweighted:PolyNotUnique', ...
       'Polynomial is not unique; degree >= number of data points.')
elseif condest(R) > 1.0e10
    if nargout > 2
        warning('polyfitweighted:RepeatedPoints', ...
            'Polynomial is badly conditioned. Remove repeated data points.')
    else
        warning('polyfitweighted:RepeatedPointsOrRescale', ...
            ['Polynomial is badly conditioned. Remove repeated data points\n' ...
            '         or try centering and scaling as described in HELP POLYFIT.'])
    end
end
p = p.';          % Polynomial coefficients are row vectors by convention.

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function test_ap1d

close all;

opts.mu1=1; opts.mu2=1; opts.beta=opts.mu1*1.5;
opts.numit=20000;
oversamp=3;
N=500;
rand_init_factor=1;
example_num=1.8;

xx=linspace(-oversamp,oversamp,N);

if example_num==1
    f_exact=create_gaussian(xx,0.1);
elseif example_num==1.5
    f_exact=(abs(xx)<0.2);
elseif example_num==1.8
    f_exact=(abs(xx)<0.2).*exp(0.1*xx);
elseif example_num==2
    F1=create_gaussian(xx,0.3);
    F2=create_gaussian(xx-0.5,0.1).*(abs(xx-0.5)<0.1);
    f_exact=F1+F2;
elseif example_num==3
    f_exact=(abs(xx-0.1)<0.6).*abs(xx) + create_gaussian(xx,0.2) + rand(size(xx))*0.02;
end;

f_exact=f_exact.*(abs(xx)<=1)*1;
u=abs(fftb(f_exact));

opts.f_ref=f_exact;
opts.f_init=f_exact + pi*(rand(size(f_exact))*2-1) * rand_init_factor;
opts.fig=figure; set(opts.fig,'position',[100,100,1600,400]);
figure; plot(xx,opts.f_ref);
f=ap1d(xx,u,opts);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%