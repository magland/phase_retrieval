function ap2d_tests

close all;

rng(2);

% Select example=1 or example=2
example=2;

if example==1
    N=200;
    [xx,yy]=ndgrid(linspace(-3,3,N),linspace(-3,3,N));
    F=create_gaussian(xx,yy,0.2);
    noise_factor=0.01; %How far away from the true solution we start -- set to 1 to randomize completely
elseif example==2
    N=200;
    [xx,yy]=ndgrid(linspace(-2,2,N),linspace(-2,2,N));
    %F=create_gaussian(xx,yy,0.2);
    F=create_gaussian(xx-0.5,yy+0.2,0.2)*0.3.*((xx-0.5).^2+(yy+0.2).^2<=0.2^2);
    %F=F.*(yy<0.25);
    noise_factor=1e-7; %How far away from the true solution we start -- set to 1 to randomize completely
end;

u=abs(fftb(F));
figure; imagesc(F); colormap('gray');
ph=angle(fftb(F));

figure; imagesc(log10(u)); colormap('gray');

apfig=figure; plot(1:10); set(apfig,'position',[100,100,1500,400]);
numit=300;
ph0=ph + (rand(size(u))*2-1)*pi*noise_factor;

%Here's the actual algorithm!
[f,err]=ap2d(xx,yy,u,ph0,numit,F,apfig);


end

function [f,positivity_deviation]=ap2d(xx,yy,u,ph,numit,ref,apfig)

ph0_ref=angle(fftb(ref));

alpha=0.3;
beta=0.8;

ph0=ph;
positivity_deviations=ones(1,numit)*inf;
err_image=ones(1,numit)*inf;
err_image2=ones(1,numit)*inf;
err_fhat_mag=ones(1,numit)*inf;
fprev=zeros(size(u));
for it=1:numit
    fhat=u.*exp(i*ph0);
    f=real(ifftb(fhat));
    positivity_deviation=max(0,-min(f(:)));
    positivity_deviations(it)=positivity_deviation;
    err_image(it)=max(abs(f(:)-ref(:)));
    err_image2(it)=max(abs(f(:)-fprev(:)));
    fproj=(1-alpha)*fprev+alpha*f.*(f>=0);
    fproj=fproj.*(abs(xx)<1).*(abs(yy)<1);
    %fproj=f;
    %fproj=f.*(f>=0).*(abs(xx)<1).*(abs(yy)<1) +(fprev-beta*fprev).*((f<0)|(abs(xx)>=1)|(abs(yy)>=1));
    %fproj=f.*(f>=0);
    %fproj=f.*(abs(xx)<1).*(abs(yy)<1) + (fprev-f).*((abs(xx)>=1)|(abs(yy)>=1));
    fprev=fproj;
    
    new_fhat=fftb(fproj);
    ph0=angle(new_fhat);
    ph0=normalize_phase(ph0,ph0_ref);
    err_fhat_mag(it)=max(abs(abs(new_fhat(:))-u(:)));
    if nargin>=4
        figure(apfig);
        subplot(1,3,1);
        imagesc(f); hold on;
        title('recon');
        
        subplot(1,3,2);
        semilogy(1:numit,positivity_deviations,'b'); hold on;
        semilogy(1:numit,err_fhat_mag,'g'); hold on;
        semilogy(1:numit,err_image,'r'); hold on;
        semilogy(2:numit,err_image2(2:end),'m'); hold on;
        semilogy(2:numit,abs(diff(positivity_deviations)),'k'); hold off;
        title('Error');
        legend('Max dev. from positivity','Max err abs(fhat)','Max err image');
        subplot(1,3,3);
        phase_error=angle(exp(i*(ph0-ph0_ref)));
        imagesc(phase_error); title('phase error');
        pause(0.0001); drawnow;
    end;
end;

figure; plot(log10(u(:)),phase_error(:),'.');
xlabel('Log10 of magnitude'); ylabel('Phase error');
title('Pixel-wise phase error as a function of Fourier magnitude');

end

function ph2=normalize_phase(ph,ph_ref)

ph_diff=ph-ph_ref;
[N1,N2]=size(ph);
M1=ceil((N1+1)/2);
M2=ceil((N2+1)/2);
slope1=(ph_diff(M1+1,M2)-ph_diff(M1-1,M2))/2;
slope2=(ph_diff(M1,M2+1)-ph_diff(M1,M2-1))/2;
[GX,GY]=ndgrid(((0:N1-1)-M1),((0:N2-1)-M2));
ph2=ph-GX*slope1-GY*slope2;

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