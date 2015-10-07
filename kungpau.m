function f=kungpau(xx,yy,u,opts)

if (nargin==0) test_kungpau; return; end;

%Start with a random guess
f=rand(size(u));

%Set algorithm parameters
alpha1=0.5; alpha2=0.5;
apodization=5;

%initialize the positivity tolerance
positivity_tolerance=0.01;

fA=figure;
fB=figure;

success_score=0;

done=false;
num_fails_at_this_apodization=0;
fig_timer=tic;
while (~done)    
    
    u0=apodize(u,apodization);
    
    if (isfield(opts,'f_exact'))
        f=register_with_reference(f,opts.f_exact);
        f_exact_apodized=ifftb(apodize(fftb(opts.f_exact),apodization));
    end;
    
    %Do a few iterations
    for it=1:40
        f_next=real(ifftb(pi_2(fftb(pi_1(f,xx,yy,alpha1)),u0,alpha2)));
        f=f_next;
    end;
    
    if (toc(fig_timer)>0.2)
        figure(fA); imagesc(f); colormap('gray'); drawnow;
        figure(fB); imagesc(f_exact_apodized); colormap('gray');
        fig_timer=tic;
    end;
    %Are we within tolerance?
    fhat=fftb(f);
    image_resid=max(0,-min(f(:)));
    fourier_resid=max(abs(abs(fhat(:))-u0(:)));
    if (image_resid<positivity_tolerance)
        success_score=success_score+1;
        %good, now we can better enforce the Fourier constraint
        if (alpha2<1)
            alpha2=min(1,alpha2+0.02);
        else
            %can't enforce any more, so let's relax the image constraint
            if (alpha1>0)
                alpha1=max(0,alpha1-0.02);
            else
                figure; imagesc(f); colormap('gray'); title(sprintf('ap=%d',apodization));
                %Hurray, let's increase the apodization!
                if (apodization>200)
                    break;
                end;
                apodization=apodization*2;
                alpha1=0.5; alpha2=0.5;
                success_score=0;
                num_fails_at_this_apodization=0;
            end;
        end;
    else
        success_score=success_score-2;
        %not to worry, we just need to relax the Fourier constraint a bit
        alpha2=max(0,alpha2-0.01);
        %and enforce the image constraint
        alpha1=min(1,alpha1+0.01);
        
        if (success_score<-100)
            f=f+rand(size(u))*3;
        elseif (success_score<-50)
            f=f+rand(size(u))*1;
        end
        
        if (success_score<-150)
            num_fails_at_this_apodization=num_fails_at_this_apodization+1;
            if (num_fails_at_this_apodization>=3)
                disp('##############################################');
                apodization=apodization/2;    
                num_fails_at_this_apodization=0;
                f=rand(size(u));
            else
                f=rand(size(u));
            end;
            alpha1=0.5; alpha2=0.5;
            success_score=0;
            
        end;
    end;
    fprintf('Fourier resid: %.g, image resid: %.g, alpha1: %g, alpha2: %g, ap: %g (%g)\n',fourier_resid,image_resid,alpha1,alpha2,apodization,success_score);
end;

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

function d0=apodize(d,sigma)

N=size(d,1);
M=ceil((N+1)/2);
aa=((1:N)-M); [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
d0=d.*exp(-GR.^2/sigma^2);

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


function [xx,yy,f_exact]=create_example(num)

oversamp=1;
N=100;
[xx,yy]=ndgrid(linspace(-oversamp,oversamp,N),linspace(-oversamp,oversamp,N));

if num==1
    f_exact=create_gaussian(xx,yy,0.3);
elseif num==1.5
    f_exact=(abs(xx)<=0.4).*(abs(yy)<=0.4);
elseif num==1.51
    f_exact=(abs(xx)<=0.4).*(abs(yy)<=0.4);
    f_exact=f_exact+create_gaussian(xx-0.3,yy-0.4,0.3)*1;
elseif num==2
    F1=create_gaussian(xx,yy,0.3);
    F2=create_gaussian(xx-0.5,(yy-0.2),0.1).*(abs(xx-0.5).^2+abs(yy-0.25).^2<0.1^2);
    f_exact=F1+F2;
elseif num==3
    f_exact=(abs(xx-0.1)<0.6).*abs(xx) + create_gaussian(xx,yy,0.2) + rand(size(xx))*0.02;
elseif num==4
    f_exact=zeros(size(xx));
    for kk=1:20
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        f_exact=f_exact+create_gaussian(xx-cc(1),yy-cc(2),rr/2).*((xx-cc(1)).^2+(yy-cc(2)).^2<=rr^2);
    end;
    %f_exact=f_exact+create_gaussian(xx,yy,0.5)*0.2;
    f_exact=f_exact+randn(size(xx))*0.02;
elseif num==4.1
    f_exact=zeros(size(xx));
    for kk=1:10
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.25;
        f_exact=f_exact+create_gaussian(xx-cc(1),yy-cc(2),rr/2).*((xx-cc(1)).^2+(yy-cc(2)).^2<=rr^2);
    end;
    f_exact=f_exact+create_gaussian(xx,yy,0.5)*0.2;
    f_exact=f_exact+rand(size(xx))*0.02;
end;

end

function test_kungpau

close all;

[xx,yy,f_exact]=create_example(4);
u=abs(fftb(f_exact));

figure; imagesc(f_exact); colormap('gray');

opts.f_exact=f_exact;
kungpau(xx,yy,u,opts);

end
