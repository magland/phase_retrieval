function f=ap2d(xx,yy,u,opts)

if (nargin==0) test_ap2d; return; end;

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
error=ones(1,opts.numit)*inf;

fig_timer=tic;
for it=1:opts.numit

    f_current=pi_1(g_current,xx,yy,alpha1);
    ghat_next=pi_2(fftb(f_current),u,alpha2);
    g_next=real(ifftb(ghat_next));
    g_next_registered=register_with_reference(g_next,opts.f_ref);

    tmp1=abs(g_current-pi_1(g_current,xx,yy,1)).^2; resid1(it)=opts.mu1*sqrt(sum(tmp1(:)));
    tmp2=abs(g_current-ifftb(pi_2(fftb(g_current),u,1))).^2; resid2(it)=opts.mu2*sqrt(sum(tmp2(:)));
    residsum(it)=resid1(it)+resid2(it);
    error(it)=sqrt(sum(abs(g_next_registered(:)-opts.f_ref(:)).^2));
    if (it>1)&&(residsum(it)>residsum(it-1))
        fprintf('%g>%g\n',residsum(it),residsum(it-1));
        error('Unexpected increase in residual');
    end;

    if ((isfield(opts,'fig')) && ((it==1)||(it==opts.numit)||(toc(fig_timer)>1)) )

        figure(opts.fig);
        subplot(1,4,1);
        imagesc(g_next_registered); colormap('gray');
        title('Recon');
        subplot(1,4,2);
        semilogy(1:opts.numit,residsum,'k',1:opts.numit,resid1,'b',1:opts.numit,resid2,'r',2:opts.numit,1e3*abs(diff(residsum)),'m',1:opts.numit,error,'c');
        legend('resid sum','resid 1','resid 2','1000 x diff resid sum','error');
        title('Residuals');
        if (isfield(opts,'f_ref'))
            subplot(1,4,3);
            phase_diff=angle(fftb(g_next_registered).*conj(fhat_ref));
            %coeffs=polyfitweighted(xx,phase_diff,1,abs(fhat_ref));
            %phase_diff=phase_diff-(coeffs(1)*xx+coeffs(2));
            imagesc(phase_diff); colormap('gray');
            title('Phase error');
            
            subplot(1,4,4);
            semilogx(abs(u(:)),phase_diff(:),'b.');
            title('Phase error vs fhat mag');
        end;
        
        drawnow;
        fig_timer=tic;

    end;

    g_current=g_next;

end

f=g_current;

end

function A2=register_with_reference(A,A_ref)

N=size(A_ref,1);
M=ceil((N+1)/2);
[GX,GY]=ndgrid((0:N-1)-M,(0:N-1)-M);


A1_CX=sum(A_ref(:).*GX(:))/sum(A_ref(:));
A1_CY=sum(A_ref(:).*GY(:))/sum(A_ref(:));

A2_CX=sum(A(:).*GX(:))/sum(A(:));
A2_CY=sum(A(:).*GY(:))/sum(A(:));

DX=A2_CX-A1_CX;
DY=A2_CY-A1_CY;

A2=real(ifftb(fftb(A).*exp(2*pi*i*(DX*GX+DY*GY)/N)));



end

function f_new=pi_1(f,xx,yy,alpha1)

mask1=(f>=0)&(abs(xx)<=1)&(abs(yy)<=1);

f_new=alpha1*f.*mask1 + (1-alpha1)*f;

end

function fhat_new=pi_2(fhat,u,alpha2)

ph=angle(fhat);
fhat_new=alpha2*u.*exp(i*ph) + (1-alpha2)*fhat;

end

function Y=fftb(X)
Y=fftshift(fft2(ifftshift(X)));
end

function Y=ifftb(X)
Y=fftshift(ifft2(ifftshift(X)));
end


function Y=create_gaussian(xx,yy,sigma)
Y=exp(-(xx/sigma).^2).*exp(-(yy/sigma).^2);
end

function d0=gaussian_apodize(d,sigma)

N=size(d,1);
M=ceil((N+1)/2);
aa=((0:N-1)-M); [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
d0=d.*exp(-GR.^2/sigma^2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function test_ap2d

rng(1);

close all;

opts.mu1=1; opts.mu2=1; opts.beta=opts.mu1*1.5;
opts.numit=50000;
oversamp=2;
N=500;
rand_init_factor=1;
example_num=2;

[xx,yy]=ndgrid(linspace(-oversamp,oversamp,N),linspace(-oversamp,oversamp,N));

if example_num==1
    f_exact=create_gaussian(xx,yy,0.3);
elseif example_num==1.5
    f_exact=(abs(xx)<=0.4).*(abs(yy)<=0.4);
elseif example_num==2
    F1=create_gaussian(xx,yy,0.3);
    F2=create_gaussian(xx-0.5,(yy-0.2),0.1).*(abs(xx-0.5).^2+abs(yy-0.25).^2<0.1^2);
    f_exact=F1+F2;
elseif example_num==3
    f_exact=(abs(xx-0.1)<0.6).*abs(xx) + create_gaussian(xx,yy,0.2) + rand(size(xx))*0.02;
end;

f_exact=(f_exact).*(abs(xx)<=1).*(abs(yy)<=1);

%f_exact= real(ifftb(gaussian_apodize(fftb(f_exact),30)));
xx=xx/2; yy=yy/2;

u=abs(fftb(f_exact));
u(u(:)<max(u(:))*0.1)=0;
figure; hist(log10(u(:)),1000); xlim([-10,10]);

opts.f_ref=f_exact;
opts.f_init=f_exact + pi*(rand(size(f_exact))*2-1) * rand_init_factor;
opts.fig=figure; set(opts.fig,'position',[100,100,1600,400]);
figure; imagesc(opts.f_ref); colormap('gray');
f=ap2d(xx,yy,u,opts);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%