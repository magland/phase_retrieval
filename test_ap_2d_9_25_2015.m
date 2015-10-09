function test_ap_2d_9_25_2015

close all;

rng(1);

N=200; [xx,yy]=ndgrid(linspace(-3,3,N),linspace(-3,3,N));
numit=1000;

F=create_2d_example(xx,yy);
F=ifftb(gaussian_apodize(fftb(F),0.1));
figure; imagesc(F); colormap('gray');
set(gcf,'position',[100,100,600,600]);

u=compute_fft_mag_data(F);
ph=compute_fft_phase_data(F);
figure; imagesc(log10(u)); colormap('gray');
set(gcf,'position',[700,100,600,600]);
title('log magnitude of fourier data');

ph_rand=ph+(rand(size(ph))-0.5)*pi*0.00000000;

[F2,resids]=do_ap(xx,yy,u,ph_rand,numit);
figure; plot(log10(resids));
set(gcf,'position',[1400,100,600,600]);

figure; imagesc(F2); colormap('gray');
set(gcf,'position',[2100,100,600,600]);

disp(resids(end));

figure; imagesc(angle(fftb(F)./fftb(F2))); colormap('gray');
set(gcf,'position',[2100,700,600,600]);
figure; imagesc(log10(abs(fftb(F)./fftb(F2)))); colormap('gray');
set(gcf,'position',[2100,700,600,600]);
figure; imagesc(abs(fftb(F))-abs(fftb(F2))); colormap('gray');
set(gcf,'position',[2800,700,600,600]);

figure; imagesc(log10(abs(fftb(F2)))); colormap('gray');
set(gcf,'position',[700,700,600,600]);


figure; imagesc(F-F2); colormap('gray');
set(gcf,'position',[2100,700,600,600]);

end

function [F2,resids]=do_ap(xx,yy,u,ph0,numit)

ph1=ph0;
for it=1:numit
F2=real(ifftb(u.*exp(i*ph1)));
F2(:)=F2(:).*(abs(xx(:))<=1).*(abs(yy(:))<=1).*(F2(:)>=0);
u2=abs(fftb(F2));
resid=sqrt(sum((u2(:)-u(:)).^2)/length(u));
resids(it)=resid;
ph1=angle(fftb(F2));
end;

end

function F=create_2d_example(xx,yy)

%A=((abs(xx)<0.2)+(abs(xx-0.4)<0.3)).*((abs(yy)<0.2)+(abs(yy-0.2)<0.3));
A=(abs(xx)<0.305).*(abs(yy)<0.404);
B=(abs(xx-0.2)<0.207).*(abs(yy-0.2)<0.315);
F=A+B;
%F=A;

%A=(abs(xx)<0.2)*1.0;
%B=(abs(xx-0.2)<0.3)*1.0;

%C=(abs(yy)<0.2)*1.0;
%D=(abs(yy-0.2)<0.3)*1.0;

%F=(A+B).*(C+D);

end

function u=compute_fft_mag_data(F)
u=abs(fftb(F));
end

function ph=compute_fft_phase_data(F)
ph=angle(fftb(F));
end

function ph=create_random_phase(SS)
ph=rand(SS)*2*pi;
end

function Y=fftb(X) %for convenience
Y=fftshift(fft2(fftshift(X)));
end

function Y=ifftb(X) % for convenience
Y=ifftshift(ifft2(ifftshift(X)));
end

function d0=gaussian_apodize(d,frac)

N=size(d,1);
aa=((0:N-1)*2-N)/N; [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
d0=d.*exp(-GR.^2/frac^2);

end
