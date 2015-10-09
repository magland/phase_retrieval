function phase_retrieval_test_1d

close all;

%rng(3);

% The entire image will be Nx1 (but the actual image is contained in within a box)
N=240*3;
noise_level=0.0;
opts.oversamp=4; % This controls how small the image box is relative to the entire image
opts.beta1=0; opts.beta2=0; opts.beta3=0; % These are zero for now
% We start out with 50 tries per step, then move to 20, then then decrease to 2
opts.num_tries_seq=[50,50,50,50,50,20,20,20,20,20, 2]; 
% We start out with 50 iterations per try, then move to 20, then then decrease to 5
opts.num_it_seq=[50,50,50,50,50,20,20,20,20,20, 5];
% This is the collection of resolution scales we step through
opts.multiscale_factors=[0.02:0.02:1];
%opts.multiscale_factors=[1];

% Create the famous image
u_true=create_image_1d(N,opts);
% Add some noise
u_true=u_true+randn(size(u_true))*noise_level;
% Let's get the magnitude k-space data
d=fftb(u_true);
d=abs(d);

u_true_original=u_true;
u_true=ifftb(gaussian_apodize(fftb(u_true),1)); %This is the real gold standard
figure; plot(u_true); set(gcf,'position',[0,0,600,600]);
title('True function');

u=zeros(size(d));
fA=figure; set(gcf,'position',[700,0,600,600]);
fB=figure; set(gcf,'position',[1400,0,600,600]);
fC=figure; set(gcf,'position',[2100,0,600,600]);

MSEs=ones(2,length(opts.multiscale_factors))*inf;
for idd=1:length(opts.multiscale_factors)
    multiscale_factor=opts.multiscale_factors(idd);
    d0=gaussian_apodize(d,multiscale_factor); % Restrict to low res
    N2=min(N,ceil(N*multiscale_factor*2.5)); % The cropped region of k-space we are looking at
    d0=cropto(d0,N2);
    u0=real(ifftb(cropto(fftb(u),N2))); % Crazy stuff, but correct
    opts.num_tries=opts.num_tries_seq(min(idd,length(opts.num_tries_seq)));
    opts.num_it=opts.num_it_seq(min(idd,length(opts.num_it_seq)));
    [u,resid,convergence]=do_recon_2(d0,u0,opts); % Here's the core procedure
    u=real(ifftb(padto(fftb(u),N))); % Crazy stuff, but correct
    MSEs(1,idd)=compute_MSE(u,u_true);
    lowres_compare=ifftb(gaussian_apodize(fftb(u_true_original),multiscale_factor));
    MSEs(2,idd)=compute_MSE(u,ifftb(gaussian_apodize(fftb(u_true_original),multiscale_factor)));
    title(sprintf('%g',multiscale_factor));
    figure(fA); plot(u); colormap('gray');  title(sprintf('%g',multiscale_factor));
    %figure; plot(lowres_compare); colormap('gray');  title(sprintf('lowres compare %g',multiscale_factor));
    figure(fB); semilogy(1:length(convergence),convergence);  title(sprintf('%g',multiscale_factor));
    figure(fC); semilogy(opts.multiscale_factors,MSEs(1,:),'b',opts.multiscale_factors,MSEs(2,:),'r'); title('MSE');
    drawnow;
    pause(0.02);
end;

end

function MSE=compute_MSE(X,Y)
% Compute the MSE after registration for translation and/or 180 deg
% rotation

MSE1=compute_MSE_helper(X,Y);
MSE2=compute_MSE_helper(X,Y(end:-1:1,end:-1:1));

MSE=min(MSE1,MSE2);

end

function MSE=compute_MSE_helper(X,Y)
% Compute the MSE after registration for translation

X2=fft(X);
Y2=fft(Y);
Z=ifft(X2.*conj(Y2));

[~,ind]=max(abs(Z(:)));
[i1,i2]=ind2sub(size(Z),ind);
%figure; imagesc(abs(Z)); colormap('gray');
%disp([i1,i2]);
Y=circshift(circshift(Y,i1-1,1),i2-1,2);
Y2=fft(Y);
Z=ifft(X2.*conj(Y2));
%figure; imagesc(abs(Z)); colormap('gray');

MSE=sum((X(:)-Y(:)).^2)/length(X(:));

end

function Y=cropto(X,N2)
N1=length(X);
center_ind1=ceil((N1+1)/2);
center_ind2=ceil((N2+1)/2);
N1,N2
%Y=X((1:N2)-center_ind2+center_ind1,(1:N2)-center_ind2+center_ind1);
Y=X((1:N2)-center_ind2+center_ind1);
end

function Y=padto(X,N2)
N1=length(X);
center_ind1=ceil((N1+1)/2);
center_ind2=ceil((N2+1)/2);
Y=zeros(N2,1);
Y((1:N1)-center_ind1+center_ind2)=X;
end

function [u,resid,convergence]=do_recon_2(d,u0,opts)

all_resids=zeros(1,opts.num_tries);
all_tries=cell(1,opts.num_tries);

% Try a bunch of times and take the one with the best resid
for j=1:opts.num_tries
%parfor j=1:opts.num_tries
    fprintf('Try %d\n',j);
    [u,resid,convergence]=do_recon(d,u0,opts); % Here's the real engine
    all_resids(j)=resid;
    all_tries{j}.u=u;
    all_tries{j}.resid=resid;
    all_tries{j}.convergence=convergence;
end;
[~,best_try]=min(all_resids);
resid=all_tries{best_try}.resid;
u=all_tries{best_try}.u;
convergence=all_tries{best_try}.convergence;

%figure; hist(all_resids,opts.num_tries);

end

function [u,resid,convergence]=do_recon(d,u0,opts)

%This is the real engine!

N=size(d,1);
aa=((0:N-1)*2-N)/N*opts.oversamp; GX=aa';
inside_box=(abs(GX)<1.5);
outside_box=~inside_box;

convergence=zeros(1,1);

u0_hat=fftb(u0);
phase_u0_hat=exp(i*angle(u0_hat));
% Add random phase according to the magnitude errors
u0_hat_mag_error_factor=min(1,abs(log(abs(u0_hat)./d))/log(2));
rand_phase_factor=u0_hat_mag_error_factor;
rand_phase=exp(2*pi*i*(rand(size(d))-0.5).*rand_phase_factor);
uhat=d.*phase_u0_hat.*rand_phase;
u=real(ifftb(uhat));
uhat=fftb(u);

% The altnerating projection algorithm!
for it=1:opts.num_it
    u=real(ifftb(uhat));
    positive=(u>0); negative=~positive;
    u= ...
        u.*inside_box.*positive*1  ...
        + u.*inside_box.*negative*opts.beta1 ...
        + u.*outside_box.*positive*opts.beta2 ...
        + u.*outside_box.*negative*opts.beta3;
    uhat=fftb(u);
    resid=sqrt(sum((abs(uhat(:))-d(:)).^2)/length(d(:)));
    convergence(it)=resid;
    phase_uhat=exp(i*angle(uhat));
    uhat=phase_uhat.*d;
    it=it+1;
end

end

function d0=gaussian_apodize(d,frac)

N=length(d);
aa=(((0:N-1)*2-N)/N)';
GR=sqrt(aa.^2);
d0=d.*exp(-GR.^2/frac^2);

end


function u_true=create_image_1d(N,opts)

aa=(((0:N-1)*2-N)/N)';
GX=aa*opts.oversamp;
M=ceil(N/2);

A=(abs(aa)<0.05)*1.0;
B=(abs(aa-0.1)<0.1)*1.0;

u_true=ifftb(fftb(A).^2.*fftb(B))/N;

beta=1;
%figure; plot(aa,u_true);
%u_true=u_true.*exp(-beta*aa);
%figure; plot(aa,u_true);

%u_true=u_true+(abs(aa-0.4)<0.05)*0.1;
u_true=u_true+circshift(u_true,150)*0.1;


%u_true=u_true+1./(1+(8*(aa-0.3)).^6);

%error('stop');

end

function Y=fftb(X) %for convenience
Y=fftshift(fft(fftshift(X)));
end

function Y=ifftb(X) % for convenience
Y=ifftshift(ifft(ifftshift(X)));
end