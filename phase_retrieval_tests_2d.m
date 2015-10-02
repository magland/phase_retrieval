function phase_retrieval_tests_2d

close all;

image_example=2; %Use 1,2,3 (see below)
N=60; %The oversampled image will be NxN
oversamp=2; %The oversampling factor
initialization_distance=1; %Use 1 to use a fully random starting point. Use 0 to start at the solution.
numit=60; %Number of iterations
opts.method='ap';
%opts.method='ap_miao'; opts.beta=0.6;
%opts.method='ap_damp'; opts.beta=0.8;

[xx,yy]=ndgrid(linspace(-oversamp,oversamp,N),linspace(-oversamp,oversamp,N));

if image_example==1
    F=create_gaussian(xx,yy,0.2);
elseif image_example==2
    F1=create_gaussian(xx,yy,0.3);
    F2=create_gaussian(xx-0.5,yy+0.2,0.5).*((xx-0.5).^2+(yy+0.2).^2<=0.5^2);
    F3=((xx+0.5).^2+(yy+0.2).^2<=0.1^2);
    F=F1+F2+F3;
    F=ifftb(gaussian_apodize(fftb(F),0.4));
elseif image_example==3
    F0=create_gaussian(xx,yy,0.5);
    %F1=((xx-0.5).^2+(yy+0.2).^2<=0.5^2);
    F2=((xx-0.6).^2+(yy+0.3).^2<=0.3^2);
    %F=(F1+F2).*F0.*rand(size(F0));
    F=F0;
end

u=abs(fftb(F));
ph=angle(fftb(F));

figure; imagesc(F); colormap('gray'); title('True F');
figure; imagesc(log10(u)); colormap('gray'); title('True log magnitude Fhat');
figure; imagesc(ph); colormap('gray'); title('True phase Fhat');

apfig=figure; plot(1:10); set(apfig,'position',[100,100,2000,400]);
ph0=ph + (rand(size(u))*2-1)*pi*initialization_distance;
[f,err]=ap2d(xx,yy,u,ph0,numit,opts,F,apfig);

apfig=figure; plot(1:10); set(apfig,'position',[100,100,2000,400]);
ph0=ph + (rand(size(u))*2-1)*pi*initialization_distance;
[f2,err2]=ap2d(xx,yy,u,ph0,numit,opts,F,apfig);

phase_error=angle(fftb(f).*conj(fftb(f2)));
figure; plot(log10(u(:)),phase_error(:),'.');

 
% apfig=figure; plot(1:10); set(apfig,'position',[100,100,2000,400]);
% ph0=ph + (rand(size(u))*2-1)*pi*initialization_distance;
% [f3,err3]=ap2d(xx,yy,u,ph0,numit,opts,F,apfig);


 
%test=angle(fftb(f).*conj(fftb(f2)));
% figure; imagesc(test);
% 
% test=angle(fftb(f).*conj(fftb(f3)));
% figure; imagesc(test);
% 
% test=angle(fftb(f).*conj(fftb(F)));
% figure; imagesc(test);

end

function [f,positivity_deviation]=ap2d(xx,yy,u,ph,numit,opts,ref,apfig)

u(find(u(:)<10^1.0))=0;

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
    positivity_deviation=sqrt(sum(f(:).^2.*(f(:)<=0))/length(f(:)));
    positivity_deviations(it)=positivity_deviation;
    err_image(it)=sqrt(sum(abs(f(:)-ref(:)).^2)/length(f(:)));
    err_image2(it)=sqrt(sum(abs(f(:)-fprev(:)).^2)/length(f(:)));
    
    %eps=min(0,max(-f(:)));
    eps=0;
    
    if (strcmp(opts.method,'ap_miao'))
        fproj=f.*(f>=eps).*(abs(xx)<1).*(abs(yy)<1) +(fprev-opts.beta*fprev).*((f<eps)|(abs(xx)>=1)|(abs(yy)>=1));
    elseif (strcmp(opts.method,'ap'))
        fproj=f.*(f>=eps).*(abs(xx)<1).*(abs(yy)<1);
    elseif (strcmp(opts.method,'ap_damp'))
        fproj=f.*(f>=eps).*(abs(xx)<1).*(abs(yy)<1)+(f*opts.beta).*((f<eps)|(abs(xx)>=1)|(abs(yy)>=1));;
    end;
    
    fprev=fproj;
    
    new_fhat=fftb(fproj);
    ph0=angle(new_fhat);
    ph0=normalize_phase(ph0,ph0_ref,u); %Important for registration
    err_fhat_mag(it)=max(abs(abs(new_fhat(:))-u(:)));
    if nargin>=4
        figure(apfig);
        subplot(1,4,1);
        imagesc(f); hold on;
        title('recon');
        
        subplot(1,4,2);
        semilogy(1:numit,positivity_deviations,'b'); hold on;
        %semilogy(1:numit,err_fhat_mag,'g'); hold on;
        semilogy(1:numit,err_image,'r'); hold on;
        semilogy(2:numit,err_image2(2:end),'m'); hold on;
        %semilogy(2:numit,abs(diff(positivity_deviations)),'k'); hold on;
        hold off;
        title('Error');
        legend('|f-pi(f)|','|f-ftrue|');
        subplot(1,4,3);
        phase_error=angle(exp(i*(ph0-ph0_ref)));
        imagesc(phase_error); title('phase error');
        %imagesc(ph0); title('phase');
        
        
        mag_error=(abs(new_fhat)-u);
        subplot(1,4,4);
        %imagesc(mag_error); colormap('gray');
        plot(log10(u(:)),phase_error(:),'.'); colormap('gray');
        
        drawnow;
    end;
end;

figure; plot(log10(u(:)),phase_error(:),'.');
xlabel('Log10 of magnitude'); ylabel('Phase error');
title('Pixel-wise phase error as a function of Fourier magnitude');

end

function ph2=normalize_phase(ph,ph_ref,w)

ph_diff=ph-ph_ref;

dx=ph_diff(2:end,:)-ph_diff(1:end-1,:); wx=(w(2:end,:)+w(1:end-1,:))/2;
dy=ph_diff(:,2:end)-ph_diff(:,1:end-1); wy=(w(:,2:end)+w(:,1:end-1))/2;

aa=angle(sum(exp(i*dx(:)).*wx(:))/sum(wx(:)));
bb=angle(sum(exp(i*dy(:)).*wy(:))/sum(wy(:)));

[N1,N2]=size(ph);
M1=ceil((N1+1)/2);
M2=ceil((N2+1)/2);
[GX,GY]=ndgrid(((0:N1-1)-M1),((0:N2-1)-M2));
ph2=ph-GX*aa-GY*bb;
ph2=ph2-ph2(M1,M2);
ph2=angle(exp(i*ph2));

% ph_diff=ph-ph_ref;
% [N1,N2]=size(ph);
% M1=ceil((N1+1)/2);
% M2=ceil((N2+1)/2);
% slope1=(ph_diff(M1+1,M2)-ph_diff(M1-1,M2))/2;
% slope2=(ph_diff(M1,M2+1)-ph_diff(M1,M2-1))/2;
% [GX,GY]=ndgrid(((0:N1-1)-M1),((0:N2-1)-M2));
% ph2=ph-GX*slope1-GY*slope2;

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

function d0=gaussian_apodize(d,frac)

N=size(d,1);
aa=((0:N-1)*2-N)/N; [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
d0=d.*exp(-GR.^2/frac^2);

end