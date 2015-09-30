function ap1d_tests

close all;

example=2;

if example==1
    N=300;
    [xx,yy]=ndgrid(linspace(-2,2,N),linspace(-2,2,N));
    F=create_gaussian(xx,yy,0.2);
    F=F+create_gaussian(xx-0.5,yy+0.2,0.2)*0.3.*((xx-0.5).^2+(yy+0.2).^2<=0.2^2);
    noise_factor=0.01;
elseif example==2
    N=300;
    [xx,yy]=ndgrid(linspace(-2,2,N),linspace(-2,2,N));
    F=create_gaussian(xx,yy,0.2);
    F=F+create_gaussian(xx-0.5,yy+0.2,0.2)*0.3.*((xx-0.5).^2+(yy+0.2).^2<=0.2^2);
    %F=F.*(yy<0.25);
    noise_factor=0.01;
end;

u=abs(fftb(F));
Fph=angle(fftb(F));
Fph=normalize_phase(Fph);
F=real(ifftb(u.*exp(i*Fph)));

figure; imagesc(F); colormap('gray');

ph=angle(fftb(F));

apfig=figure; plot(1:10); set(apfig,'position',[100,100,1500,400]);

numit=100;
ph0=ph + (rand(size(u))*2-1)*pi*noise_factor;
[f,err]=ap2d(xx,yy,u,ph0,numit,F,apfig);


end

function [f,positivity_deviation]=ap2d(xx,yy,u,ph,numit,ref,apfig)

ph0_ref=angle(fftb(ref));
ph0_ref=normalize_phase(ph0_ref);
ref=real(ifftb(u.*exp(i*ph0_ref)));

ph0=ph;
positivity_deviations=ones(1,numit)*inf;
err_image=ones(1,numit)*inf;
err_fhat_mag=ones(1,numit)*inf;
for it=1:numit
    fhat=u.*exp(i*ph0);
    f=real(ifftb(fhat));
    positivity_deviation=max(0,-min(f(:)));
    positivity_deviations(it)=positivity_deviation;
    err_image(it)=max(abs(f(:)-ref(:)));
    fproj=f.*(f>=0).*(abs(xx)<1).*(abs(yy)<1);
    new_fhat=fftb(fproj);
    ph0=angle(new_fhat);
    ph0=normalize_phase(ph0);
    err_fhat_mag(it)=max(abs(abs(new_fhat(:))-u(:)));
    if nargin>=4
        figure(apfig);
        subplot(1,3,1);
        imagesc(f); hold on;
        title('recon');
        
        subplot(1,3,2);
        semilogy(1:numit,positivity_deviations,'b'); hold on;
        semilogy(1:numit,err_fhat_mag,'g'); hold on;
        semilogy(1:numit,err_image,'r'); hold off;
        title('Error');
        legend('Max dev. from positivity','Max err abs(fhat)','Max err image');
        subplot(1,3,3);
        imagesc(ph0-ph0_ref); title('phase error');
        pause(0.0001); drawnow;
    end;
end;

end

function ph2=normalize_phase(ph)

[N1,N2]=size(ph);
M1=ceil((N1+1)/2);
M2=ceil((N2+1)/2);
slope1=(ph(M1+1,M2)-ph(M1-1,M2))/2;
slope2=(ph(M1,M2+1)-ph(M1,M2-1))/2;
[GX,GY]=ndgrid(((0:N1-1)-M1),((0:N2-1)-M2));
ph2=ph-GX*slope1+-GY*slope2;

end

function Y=create_gaussian(xx,yy,sigma)

Y=exp(-(xx/sigma).^2).*exp(-(yy/sigma).^2);

end

function Y=fftb(X)
Y=fftshift(fft2(fftshift(X)));
end

function Y=ifftb(X)
Y=fftshift(ifft2(fftshift(X)));
end