function ap1d_tests

close all;

%select example=1 or example=2
example=1;

if example==1
    N=300;
    xx=linspace(-2,2,N);
    F=create_gaussian(xx,0.2);
    noise_factor=0.1; %How far away from the true solution we start -- set to 1 to randomize completely
elseif example==2
    N=300;
    xx=linspace(-2,2,N);
    F=create_gaussian(xx,0.2);
    F=F.*(xx<0.2);
    noise_factor=0.001; %How far away from the true solution we start -- set to 1 to randomize completely
end;

u=abs(fftb(F));
figure; plot(xx,F,'b');
ph=angle(fftb(F));

apfig=figure; plot(1:10); set(apfig,'position',[100,100,1500,400]);
numit=1000;
ph0=ph + (rand(size(u))*2-1)*pi*noise_factor;

[f,err]=ap1d(xx,u,ph0,numit,F,apfig);

end

function [f,positivity_deviation]=ap1d(xx,u,ph,numit,ref,apfig)

ph0_ref=angle(fftb(ref));

ph0=ph;
positivity_deviations=ones(1,numit)*inf;
err_image=ones(1,numit)*inf;
err_fhat_mag=ones(1,numit)*inf;
for it=1:numit
    fhat=u.*exp(i*ph0);
    f=real(ifftb(fhat));
    positivity_deviation=max(0,-min(f(:)));
    positivity_deviations(it)=positivity_deviation;
    err_image(it)=max(abs(f-ref));
    fproj=f.*(f>=0).*(abs(xx)<=1);
    new_fhat=fftb(fproj);
    ph0=angle(new_fhat);
    ph0=normalize_phase(ph0,ph0_ref);
    err_fhat_mag(it)=max(abs(abs(new_fhat)-u));
    if nargin>=4
        figure(apfig);
        subplot(1,3,1);
        plot(xx,ref,'b'); hold on;
        plot(xx,f,'r'); hold off;
        title('recon');
        
        subplot(1,3,2);
        semilogy(1:numit,positivity_deviations,'b'); hold on;
        semilogy(1:numit,err_fhat_mag,'g'); hold on;
        semilogy(1:numit,err_image,'r'); hold off;
        title('Error');
        legend('Max dev. from positivity','Max err abs(fhat)','Max err image');
        subplot(1,3,3);
        plot(xx,angle(exp(i*(ph0-ph0_ref))),'b'); title('phase error');
        pause(0.01); drawnow;
    end;
end;

end

function ph2=normalize_phase(ph,ph_ref)

ph_diff=ph-ph_ref;

N=length(ph);
M=ceil((N+1)/2);
slope=(ph_diff(M+1)-ph_diff(M-1))/2;
ph2=ph-((0:N-1)-M)*slope;

end

function Y=create_gaussian(xx,sigma)

Y=exp(-(xx/sigma).^2);

end

function Y=fftb(X)
Y=fftshift(fft(fftshift(X)));
end

function Y=ifftb(X)
Y=fftshift(ifft(fftshift(X)));
end