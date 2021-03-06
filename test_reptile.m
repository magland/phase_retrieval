function test_reptile

close all;

rng(1);

[xx,yy,f_exact]=create_example(4);
u=abs(fftb(f_exact));

figure; imagesc(f_exact); colormap('gray'); set(gcf,'position',[100,100,500,400]);
drawnow;

title(sprintf('min = %g\n',min(f_exact(:))));

opts.f_exact=f_exact;
f=reptile(xx,yy,u,opts);

figure; imagesc(f); colormap('gray'); set(gcf,'position',[700,100,500,400]);

end

function [xx,yy,f_exact]=create_example(num)

if (num==-1)
    oversamp=1;
    N=2;
    [xx,yy]=ndgrid(linspace(-oversamp,oversamp,N),linspace(-oversamp,oversamp,N));
    f_exact=rand(N,N);
    %f_exact=[0,0;1,0];
    %f_exact=[0.5,0.0001;0.7,0.3];
    return;
end;

oversamp=1.25;
N=64*oversamp;
[xx,yy]=ndgrid(linspace(-oversamp,oversamp,N),linspace(-oversamp,oversamp,N));

if num==0
    f_exact=zeros(size(xx));
    f_exact(ceil((N+1)/2),ceil((N+1)/2))=1;
elseif num==1
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
    f_exact=(abs(xx-0.1)<0.4).*(abs(yy-0.3)<0.6).*abs(xx) + create_gaussian(xx,yy,0.2);
elseif num==4
    f_exact=zeros(size(xx));
    for kk=1:20
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        f_exact=f_exact+create_gaussian(xx-cc(1),yy-cc(2),rr/2).*((xx-cc(1)).^2+(yy-cc(2)).^2<=rr^2);
    end;
    for kk=1:5
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        f_exact=f_exact+(abs(xx-cc(1))<=rr).*(abs(yy-cc(2))<=rr);
    end;
    %f_exact=f_exact+create_gaussian(xx,yy,0.5)*0.2;
    %f_exact=f_exact+randn(size(xx))*0.02;
elseif num==4.01
    f_exact=zeros(size(xx));
    for kk=1:5
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        f_exact=f_exact+create_gaussian(xx-cc(1),yy-cc(2),rr/2).*((xx-cc(1)).^2+(yy-cc(2)).^2<=rr^2);
    end;
    
elseif num==4.1
    f_exact=zeros(size(xx));
    for kk=1:20
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        f_exact=f_exact+create_gaussian(xx-cc(1),yy-cc(2),rr/2).*((xx-cc(1)).^2+(yy-cc(2)).^2<=rr^2);
    end;
    for kk=1:5
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        f_exact=f_exact+(abs(xx-cc(1))<=rr).*(abs(yy-cc(2))<=rr);
    end;
    f_exact=f_exact+create_gaussian(xx,yy,0.3)*1;
    %f_exact=f_exact+rand(size(xx))*0.02;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REMEMBER TO REMOVE THIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%xx=xx/2;
%yy=yy/2;

end

function Y=create_gaussian(xx,yy,sigma)
Y=exp(-(xx/sigma).^2).*exp(-(yy/sigma).^2);
end

function Y=fftb(X)
Y=fftshift(fft2(ifftshift(X)));
end

function Y=ifftb(X)
Y=fftshift(ifft2(ifftshift(X)));
end

function d0=apodize(d,sigma,N2)

N=size(d,1);
M=ceil((N+1)/2);
aa=((1:N)-M); [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
d0=d.*exp(-GR.^2/sigma^2);

KK=ceil(sigma*2);

d0=d0(M-N2/2:M+N2/2-1,M-N2/2:M+N2/2-1);

d0=d0*(N2/N)^2;

end