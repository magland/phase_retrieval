function ap_test_01

close all;

numit=100;
numtries=10;

N=300;
[xx,yy]=ndgrid(linspace(-2.5,2.5,N),linspace(-3,3,N));

F=create_function(xx,yy,N);
F=ifftb(gaussian_apodize(fftb(F),0.05));
u=abs(fftb(F));
ph=angle(fftb(F));
ph0=ph+(rand(size(ph))*2-1)*pi*1;

fF=figure; set(fF,'position',[100,100,300,200]); imagesc(F); colormap('gray');
fU=figure; set(fU,'position',[450,100,300,200]); imagesc(log10(u)); colormap('gray');
fResid=figure; set(fResid,'position',[800,100,250,200]);

[F2,resid]=do_ap(xx,yy,u,ph0,numit,numtries,fResid);

fF2=figure; set(fF2,'position',[100,400,300,200]); 
fU2=figure; set(fU2,'position',[450,400,300,200]);
figure(fF2); imagesc(F2); colormap('gray');
figure(fU2); imagesc(abs(fftb(F2))); colormap('gray');

F2hat=fftb(F2);
Fhat=fftb(F);
angle_diff=angle(F2hat.*conj(Fhat));
figure; imagesc(angle_diff); colormap('gray');
figure; imagesc(abs(abs(F2hat)-abs(Fhat))); colormap('gray');

end

function [F,resid]=do_ap(xx,yy,u,ph,numit,numtries,fResid)

best_final_resid=inf;

best_ph=ph;

for tt=1:numtries
    resid0=ones(1,numit)*inf;
    ph1=best_ph + (rand(size(ph))*2-1)*pi*0.1;
    for j=1:numit
        F1=real(ifftb(u.*exp(i*ph1)));
        %F1(:)=F1(:).*(F1(:)>=0).*(abs(xx(:))<1).*(abs(yy(:))<1);
        %F1(:)=F1(:).*(F1(:)>=0).*(xx(:).^2+yy(:).^2<1);
        F1(:)=F1(:).*(F1(:)>=0);
        F1hat=fftb(F1);
        ph1=angle(F1hat);
        resid0(j)=sqrt(sum(abs(u(:)-abs(F1hat(:))))/length(u(:)));
        if ((mod(j-1,20)==0)||(j==numit)) figure(fResid); semilogy(1:numit,resid0); drawnow; pause(0.01); end;
    end
    disp(resid0(end));
    if (resid0(end)<best_final_resid)
        F=F1;
        resid=resid0;
        best_ph=ph1;
        best_final_resid=resid0(end);
    end;
end;

figure(fResid); semilogy(1:numit,resid); drawnow;
disp(resid(end));

end

function F=create_function(xx,yy,N)

A=exp(-(xx.^2+yy.^2)*6).*(xx.^2+yy.^2<1);
B=(abs(xx)<0.234).*(abs(yy)<0.309);
C=(abs(xx-0.3)<0.15).*(abs(yy+0.4)<0.1);

F=A+C;

end

function Y=fftb(X)
Y=fftshift(fft2(fftshift(X)));
end

function Y=ifftb(X)
Y=fftshift(ifft2(fftshift(X)));
end

function d0=gaussian_apodize(d,frac)

N=size(d,1);
aa=((0:N-1)*2-N)/N; [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
d0=d.*exp(-GR.^2/frac^2);

end

