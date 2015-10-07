function amphibian_test

close all;

rng(1);

[xx,yy,f_exact]=create_example(4);
f_exact=real(ifftb(apodize(fftb(f_exact),20)));
[xx,yy]=ndgrid(linspace(-1,1,size(f_exact,1)),linspace(-1,1,size(f_exact,2)));
u=abs(fftb(f_exact));

figure; imagesc(f_exact); colormap('gray'); set(gcf,'position',[100,100,500,400]);
drawnow;

title(sprintf('min = %g\n',min(f_exact(:))));

figure; imagesc(f_exact); colormap('gray'); set(gcf,'position',[100,100,500,400]);
drawnow;

opts.f_exact=f_exact;
f=amphibian(xx,yy,u,opts);

figure; imagesc(f); colormap('gray'); set(gcf,'position',[700,100,500,400]);

end

function [xx,yy,f_exact]=create_example(num)

oversamp=1;
N=500;
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
    %f_exact=f_exact+create_gaussian(xx,yy,0.5)*0.2;
    %f_exact=f_exact+randn(size(xx))*0.02;
elseif num==4.1
    f_exact=zeros(size(xx));
    for kk=1:10
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.25;
        f_exact=f_exact+create_gaussian(xx-cc(1),yy-cc(2),rr/2).*((xx-cc(1)).^2+(yy-cc(2)).^2<=rr^2);
    end;
    f_exact=f_exact+create_gaussian(xx,yy,0.2)*0.2;
    f_exact=f_exact+rand(size(xx))*0.02;
end;

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

function d0=apodize_kb(d,dx)

[N1,N2]=size(d);
M1=ceil((N1+1)/2);
M2=ceil((N2+1)/2);

ff=2;

d0=d(M1-dx:M1+dx-1,M2-dx:M2+dx-1);
N1b=dx*2;
N2b=dx*2;
for k=1:N2b
    for j=1:N1b
        opts.fac1=1; opts.fac2=1;
        d0(j,k)=d0(j,k)*kb_kernel(j-dx-1,dx*2*ff,opts)*kb_kernel(k-dx-1,dx*2*ff,opts);
    end;
end;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Here is the Kaisser-Bessel kernel
function [val,nspread]=kb_kernel(x,nspread,opts)

W=nspread*opts.fac1;
beta=pi*sqrt(W*W/4-0.8)*opts.fac2;

y=beta*sqrt(1-(2*x/W).^2);
val=besseli(0,y);
val=val/besseli(0,beta);

end


function d0=apodize(d,sigma)

N=size(d,1);
M=ceil((N+1)/2);
aa=((1:N)-M); [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
d0=d.*exp(-GR.^2/sigma^2);

KK=ceil(sigma*2);

d0=d0(M-KK:M+KK-1,M-KK:M+KK-1);

d0=d0*((KK*2)/N)^2;

end