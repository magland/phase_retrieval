function investigate_tolerance

close all;

N=100;
num_it=1000;
noise_level=0.01;
[GX,GY]=ndgrid(linspace(-2,2,N),linspace(-2,2,N));
inside_box=(abs(GX)<1).*(abs(GY)<1);
u_true=create_image(N);
u_true=u_true+randn(size(u_true))*noise_level;
%u_true=u_true.*inside_box;
figure; imagesc(u_true); colormap('gray'); set(gcf,'position',[0,0,600,600]);

% Let's get the magnitude k-space data
d=fft2b(u_true);
d=abs(d);

factors=1:-0.1:0.1;
resids=zeros(size(factors));

for ii=1:length(factors)
    factor=factors(ii);
    u0=real(downsample(u_true,factor));
    figure; imagesc(u0); colormap('gray'); set(gcf,'position',[2100,0,600,600]); title(sprintf('%g',factor));
    [u,resid]=phase_retrieval_engine(d,u0,GX,GY,num_it);
    %if (ii==1)||(ii==length(factors))
        figure; imagesc(u); colormap('gray'); set(gcf,'position',[700,0,600,600]); title(sprintf('%g',factor));
        figure; semilogy(1:num_it,resid); set(gcf,'position',[1400,0,600,600]); title(sprintf('%g',factor));
        drawnow;
    %end;

    resids(ii)=resid(end)
end;

figure; semilogy(factors,resids);

end

function Y=downsample(X,pct)

N=size(X,1);
[GX,GY]=ndgrid(linspace(-1,1,N),linspace(-1,1,N));
Y=fft2b(X);
Y((abs(GX(:))>pct)|(abs(GY(:))>pct))=0;
Y=ifft2b(Y);

end

function Y=jitter(X,level)

Y=X+randn(size(X))*level;

% Y=fft2b(X);
% Y=Y.*exp(2*pi*i*randn(size(X))*level);
% Y=ifft2b(Y);

end

function u_true=create_image(N)

[GX,GY]=ndgrid(linspace(-2,2,N),linspace(-2,2,N));
M=ceil(N/2);

locs =[
   -0.1625   -0.2847   -0.6039    0.5216;
    0.1135   -0.3677   -0.4551   -0.1034
];

u_true=zeros(N,N);
for aa=1:size(locs,2)
    loc=locs(:,aa);
    GR=sqrt((GX-loc(1)).^2+(GY-loc(2)).^2);
    u_true = u_true + (GR<=0.2)*1;
    u_true = u_true - (GR<=0.1)*1;
end;
u_true = u_true + (abs(GX - 0.5)<0.1).*(abs(GY - 0.4)<0.2);
u_true(M,M+30)=3;

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

function Y=fft2b(X) %for convenience
Y=fftshift(fft2(fftshift(X)));
end

function Y=ifft2b(X) % for convenience
Y=ifftshift(ifft2(ifftshift(X)));
end