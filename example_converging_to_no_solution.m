function phase_retrieval_tests_1d

close all;

image_example=3; %Use 1,2,3 (see below)
N=200; %The oversampled image will be Nx1
oversamp=2; %The oversampling factor
initialization_distance=1; %Use 1 to use a fully random starting point. Use 0 to start at the solution.
numit=5000; %Number of iterations
%opts.method='ap';
%opts.method='ap_miao'; opts.beta=0.6;
opts.method='ap_damp'; opts.beta=0.3;

xx=linspace(-oversamp,oversamp,N);

if image_example==1
    F=create_gaussian(xx,0.3);
elseif image_example==2
    F1=create_gaussian(xx,0.3);
    F2=create_gaussian(xx-0.5,0.1).*(abs(xx-0.5)<0.1);
    F=F1+F2;
elseif image_example==3
    F=(abs(xx-0.1)<0.6).*abs(xx);
end

ph=angle(fftb(F));
u=abs(fftb(F));

figure; plot(xx,F,'b');

apfig=figure; plot(1:10); set(apfig,'position',[100,100,2000,400]);
ph0=ph + (rand(size(u))*2-1)*pi*initialization_distance;
f=ap1d(xx,u,ph0,numit,opts,F,apfig);

end

function f=ap1d(xx,u,ph,numit,opts,ref,apfig)

%u(find(u(:)<10^0))=0;

ph0_ref=angle(fftb(ref));

ph0=ph;
positivity_deviations=ones(1,numit)*inf;
err_image=ones(1,numit)*inf;
err_image2=ones(1,numit)*inf;
err_fhat_mag=ones(1,numit)*inf;
fprev=zeros(size(u));
timer=tic;
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
        fproj=f.*(f>=eps).*(abs(xx)<1) +(fprev-opts.beta*fprev).*((f<eps)|(abs(xx)>=1));
    elseif (strcmp(opts.method,'ap'))
        fproj=f.*(f>=eps).*(abs(xx)<1);
    elseif (strcmp(opts.method,'ap_damp'))
        fproj=f.*(f>=eps).*(abs(xx)<1)+(f*opts.beta).*((f<eps)|(abs(xx)>=1));
    end;
    
    fprev=fproj;
    
    new_fhat=fftb(fproj);
    ph0=angle(new_fhat);
    ph0=normalize_phase(ph0,ph0_ref,u); %Important for registration
    err_fhat_mag(it)=max(abs(abs(new_fhat(:))-u(:)));
    if ((nargin>=7)&&((toc(timer)>0.1)||(it==1)||(it==numit)))
        figure(apfig);
        subplot(1,4,1);
        plot(xx,f,'b');
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
        plot(xx,phase_error); title('phase error');
        %imagesc(ph0); title('phase');
        
        mag_error=(abs(new_fhat)-u);
        subplot(1,4,4);
        %imagesc(mag_error); colormap('gray');
        semilogx(u(:),phase_error(:),'.'); title('phase error');
        xlabel('fhat magnitude'); ylabel('phase error');
        
        drawnow;
        timer=tic;
    end;
end;

end

function ph2=normalize_phase(ph,ph_ref,w)

ph_diff=ph-ph_ref;

dx=ph_diff(2:end)-ph_diff(1:end-1);
dx=angle(exp(i*dx));
wx=min(w(2:end),w(1:end-1));

aa=angle(sum(exp(i*dx(:)).*wx(:)));

N=length(ph);
M=ceil((N+1)/2);
GX=(0:N-1)-M;
ph2=ph-GX*aa;
ph2=ph2-ph2(M);
ph2=angle(exp(i*ph2));

end

function Y=create_gaussian(xx,sigma)

Y=exp(-(xx/sigma).^2);

end

function Y=fftb(X)
Y=fftshift(fft(ifftshift(X)));
end

function Y=ifftb(X)
Y=fftshift(ifft(ifftshift(X)));
end