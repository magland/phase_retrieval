function test_ap_1d_9_25_2015

close all;

rng(1);

N=200; xx=linspace(-3,3,N);
numit=1000;

F=create_1d_example(xx);
figure; plot(xx,F);
set(gcf,'position',[100,100,600,600]);

u=compute_fft_mag_data(F);
ph=compute_fft_phase_data(F);
figure; plot(xx,u);
set(gcf,'position',[700,100,600,600]);

%ph_rand=create_random_phase(size(u));
ph_rand=ph+(rand(size(ph))-0.5)*1.0;

[F2,resids]=do_ap(xx,u,ph_rand,numit);
figure; plot(log(resids));
set(gcf,'position',[1400,100,600,600]);

figure; plot(xx,real(F2),'b',xx,imag(F2),'r');
set(gcf,'position',[2100,100,600,600]);

disp(resids(end));

end

function [F2,resids]=do_ap(xx,u,ph0,numit)

ph1=ph0;
for it=1:numit
F2=real(ifftb(u.*exp(i*ph1)));
F2(:)=F2(:).*(abs(xx(:))<=1).*(F2(:)>=0);
u2=abs(fftb(F2));
resid=sqrt(sum((u2(:)-u(:)).^2)/length(u));
resids(it)=resid;
ph1=angle(fftb(F2));
end;

end

function F=create_1d_example(xx)

A=(abs(xx)<0.2)*1.0;
B=(abs(xx-0.2)<0.3)*1.0;
%F=ifftb(fftb(A).^2.*fftb(B))/length(xx) + A + B;

F=A+B;

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
Y=fftshift(fft(fftshift(X)));
end

function Y=ifftb(X) % for convenience
Y=ifftshift(ifft(ifftshift(X)));
end

