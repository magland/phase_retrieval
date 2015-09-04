function phase_retrieval

close all;

rng(3);

N=400; %This is the total size (including outside the FOV)
noise_level=0.01; %Too afraid to add more noise at this time
num_it=400; %Number of iterations at each resolution

%First we only do one recon at full resolution
%N0s=[400];
%do_phase_retrieval(N,noise_level,N0s,num_it);
%title('Without recursive linearization');

%Second we step up from low-res to high-res
%N0s=[50,90,130,170,210,250,290,330,370,400];
N0s=[30,100,200,300,400];
do_phase_retrieval(N,noise_level,N0s,num_it);
title('With recursive linearization');;;

end

function do_phase_retrieval(N,noise_level,N0s,num_it);

% Define the true function, which *must* stay within middle half of image
[GX,GY]=ndgrid(linspace(-2,2,N),linspace(-2,2,N));
inside_box=(abs(GX)<1).*(abs(GY)<1);

u_true=zeros(N,N);
for aa=1:4
    loc=[rand(1)*2-1,rand(1)*2-1]*0.7;
    GR=sqrt((GX-loc(1)).^2+(GY-loc(2)).^2);
    u_true = u_true + (GR<=0.2)*1;
    u_true = u_true - (GR<=0.1)*1;
end;
u_true = u_true + (abs(GX - 0.5)<0.1).*(abs(GY - 0.4)<0.2);
u_true(200,230)=3;
u_true(200,240)=3;
u_true(270,240)=3;

% add some noise and plot
u_true=u_true+randn(size(u_true))*noise_level;
u_true=u_true.*inside_box;
figure; imagesc(u_true); colormap('gray'); set(gcf,'position',[0,0,600,600]);

% Let's get the magnitude k-space data
d=fft2b(u_true);
d=abs(d);

% Keep track of results at every resolution
results={};

for iN0=1:length(N0s) % Step through the resolutions
    N0=N0s(iN0);
    d0=crop(d,N0); % Crop the k-space data to achieve lower resolution
    [GX0,GY0]=ndgrid(linspace(-2,2,N0),linspace(-2,2,N0));

    % The number of retries depends on resolution
    % We want to repeat many times at low resolution, and take the best
    num_retries=1;
    if (iN0==1) num_retries=50; end;
    tries={};
    try_end_resids=zeros(1,num_retries); %Keep track of the end resids in order to pick the best
    for rr=1:num_retries
        if (iN0==1) 
            u0=ifft2b(d0.*exp(2*pi*i*rand(size(d0)))); % at lowest resolution, randomize the phase
        else
            u0=upsample(u,N0); % Otherwise, upsample for the starting point -- this is critical to recursive linearization
        end;

        fprintf('N0=%d, try=%d\n',N0,rr);
        [tries{rr}.u,tries{rr}.resid]=phase_retrieval_engine(d0,u0,GX0,GY0,num_it); % Just do it
        try_end_resids(rr)=tries{rr}.resid(end);
    end;
    [~,best_rr]=min(try_end_resids);
    u=tries{best_rr}.u;
    u=real(u);
    resid=tries{best_rr}.resid;

    figure; imagesc(real(u)); colormap('gray'); set(gcf,'position',[700,0,600,600]);
    figure; semilogy(1:num_it,resid); set(gcf,'position',[1400,0,600,600]);
    drawnow;
    
    results{iN0}.u=u;
    results{iN0}.resid=resid;
end;

% Plot the resid as a function of iteration for all of the resolutions
figure;
for k=1:length(results)
    semilogy(1:num_it,results{k}.resid);
    hold on;
end;

end

function [u,resid]=phase_retrieval_engine(d,u0,GX,GY,num_it)

GR=sqrt(GX.^2+GY.^2);

inside_box=(abs(GX)<1).*(abs(GY)<1);
outside_box=~inside_box;

uhat=fft2b(u0);
resid=zeros(1,num_it);

for j=1:num_it
    u=ifft2b(uhat);
    %figure; imagesc(real(u)); colormap('gray');
    u=real(u);
    positive=(u>0); negative=~positive;
    % Here is the formula I was using... not exactly the formula we had
    % discussed... I like this one better -- but need to experiment with
    % coeffs, etc.
    u=u.*inside_box.*positive*1 + u.*inside_box.*negative*0.1 + u.*outside_box.*positive*0.5 + u.*outside_box.*negative*0.1;
    %u=u.*inside_box.*positive*1 + u.*inside_box.*negative*1 + u.*outside_box.*positive*0.2 + u.*outside_box.*negative*0.2;
    %u=u.*inside_box.*positive*1 + u.*inside_box.*negative*0.1 + u.*outside_box.*positive*1 + u.*outside_box.*negative*0.1;
    
    uhat=fft2b(u);
    resid(j)=sqrt(sum((abs(uhat(:))-d(:)).^2)/length(d(:)));
    uhat=uhat./abs(uhat).*d; %Enforce the magnitude
end;

end

function Y=crop(X,N2)

N=size(X,1);
M2=1+ceil((N-N2)/2);
Y=X(M2:M2+N2-1,M2:M2+N2-1);

end

function Y=upsample(X,N2)

X=fft2b(X);

N=size(X,1);
M2=1+ceil((N2-N)/2);
Y=zeros(N2,N2);
Y(M2:M2+N-1,M2:M2+N-1)=X;

Y=ifft2b(Y);

end

function Y=downsample(X,N2)

N=size(X,1);
M2=1+ceil((N-N2)/2);
A=fft2b(fftshift(X));
A=A(M2:M2+N2-1,M2:M2+N2-1);
Y=ifft2(ifftshift(A));

end

function Y=fft2b(X) %for convenience
Y=fftshift(fft2(fftshift(X)));
end

function Y=ifft2b(X) % for convenience
Y=ifftshift(ifft2(ifftshift(X)));
end
