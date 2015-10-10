function f=reptile(xx,yy,u,opts)

if (nargin==0) test_reptile; return; end;

addpath /home/magland/dev/mdaio

%sigmas=[8,16,24,32,inf];
%N2s=[8,16,24,32,64*1.25];
%solution_counts=[200,100,50,20,20];
sigmas=[8];
N2s=[16];
solution_counts=[2000];

%solution_counts=[300,50,50,50,50,50];
%sigmas=[inf];
%N2s=[100];
%solution_counts=[1000];

for iii=1:length(sigmas)
    %Downsample
    sigma=sigmas(iii); N2=N2s(iii);
    u0=apodize(u,sigma,N2);
    opts0=opts;
    opts0.f_exact=real(ifftb(apodize(fftb(opts0.f_exact),sigma,N2)));
    if (iii>1)
        M2=ceil((N2+1)/2);
        opts0.means=zeros(size(u0));
        opts0.means(M2-size(means,1)/2:M2+size(means,1)/2-1,M2-size(means,1)/2:M2+size(means,1)/2-1)=means;
        opts0.stdevs=ones(size(u0));
        opts0.stdevs(M2-size(means,1)/2:M2+size(means,1)/2-1,M2-size(means,1)/2:M2+size(means,1)/2-1)=stdevs;
    end;
    figure; imagesc(opts0.f_exact); colormap('gray'); 
    title('True'); drawnow;
    [xx0,yy0]=ndgrid(linspace(min(xx(:)),max(xx(:)),size(u0,1)),linspace(min(yy(:)),max(yy(:)),size(u0,2)));

    %Find a bunch of solutions
    num_solutions=solution_counts(iii);
    num_threads=4;
    tolerance=1e-10;
    max_iterations=50000;
    opts0.num_jobs=20;
    opts0.oversamp=max(xx(:)); % a hack
    [solutions,resids_img,errors]=generate_random_solutions(xx0,yy0,u0,num_solutions,num_threads,tolerance,max_iterations,opts0);
    %[solutions,resids_img,errors]=generate_random_solutions_cl(xx0,yy0,u0,num_solutions,num_threads,tolerance,max_iterations,opts0);
    
    f=solutions{1}.f;
    figure; imagesc(f); colormap('gray');
    title('Recon'); drawnow;

    figure; plot(resids_img,errors,'b.');
    xlabel('Resid (image)');
    ylabel('Error');
    title('Error vs. residual'); drawnow;

    [means,stdevs]=compute_solution_variations(solutions(1:10),u0);
    stdevs=min(1,stdevs);
    figure; imagesc(stdevs); colorbar;
    title('Variation in Fourier data'); drawnow;
    
    figure; imagesc(real(ifftb(means))); colormap('gray');
    title('Avg. Recon'); drawnow;
end;

end

function delete_if_exists(path)
if exist(path, 'file')==2
  delete(path);
end
end

function [solutions,resids_img,errors]=generate_random_solutions_cl(xx,yy,u,num_solutions,num_threads,tolerance,max_iterations,opts)

delete_if_exists('u.mda');
delete_if_exists('f_exact.mda');
writemda(u,'u.mda');
writemda(opts.f_exact,'f_exact.mda');

%system(sprintf('/home/magland/dev/ap2d/ap2d u.mda --out-recon=recon.mda --out-resid-err=residerr.mda --ref=f_exact.mda --count=%d --tol=%g --maxit=%d --num-threads=%d',num_solutions,tolerance,max_iterations,num_threads));
num_jobs=opts.num_jobs;
runs_per_job=ceil(num_solutions/num_jobs);

for jj=1:num_jobs
    delete_if_exists(sprintf('recon-%d.mda',jj));
    delete_if_exists(sprintf('residerr-%d.mda',jj));
end;

if (isfield(opts,'means'))
    writemda(real(opts.means),'init_means_re.mda');
    writemda(imag(opts.means),'init_means_im.mda');
    writemda(opts.stdevs,'init_stdevs.mda');
    init_means_re='init_means_re.mda';
    init_means_im='init_means_im.mda';
    init_stdevs='init_stdevs.mda';
else
    init_means_re='';
    init_means_im='';
    init_stdevs='';
end

cmd=sprintf('/mnt/xfs1/home/magland/dev/ap2d/ap2d_batch.sh %d %d %d u.mda recon residerr f_exact.mda %g %d %g %s %s %s',num_jobs,runs_per_job,num_threads,tolerance,max_iterations,opts.oversamp,init_means_re,init_means_im,init_stdevs);
disp(cmd);
system(cmd);

solutions={};
resids_img=[];
errors=[];
for jj=1:num_jobs
    path=sprintf('residerr-%d.mda',jj);
    fprintf('Reading %s...\n',path);
    residerr=readmda(path);
    recon=readmda(sprintf('recon-%d.mda',jj));
    for ii=1:size(recon,3)
        solutions{end+1}.f=recon(:,:,ii);
        resids_img(end+1)=residerr(ii,1);
        errors(end+1)=residerr(ii,2);
    end;
end;

[resids_img,sort_inds]=sort(resids_img);
sorted_solutions={};
for j=1:length(solutions)
    sorted_solutions{j}=solutions{sort_inds(j)};
end;
solutions=sorted_solutions;
errors=errors(sort_inds);

end

function [solutions,resids_img,errors]=generate_random_solutions(xx,yy,u,num_solutions,num_threads,tolerance,max_iterations,opts)

solution_sets={};
parfor ct=1:num_threads
    opts0=opts;
    opts0.jj=ct;
    solutions0=generate_random_solutions_2(xx,yy,u,ceil(num_solutions/num_threads),tolerance,max_iterations,opts0);
    solution_sets{ct}=solutions0;
end;

solutions={};
for ct=1:num_threads
    SS=solution_sets{ct};
    for j=1:length(SS);
        solutions{end+1}=SS{j};
    end;
end;

resids_img=zeros(1,length(solutions));
for j=1:length(solutions)
    resids_img(j)=solutions{j}.resid_img;
end

[resids_img,sort_inds]=sort(resids_img);
sorted_solutions={};
for j=1:length(solutions)
    sorted_solutions{j}=solutions{sort_inds(j)};
end;
solutions=sorted_solutions;

errors=zeros(1,length(solutions));
for j=1:length(solutions)
    errors(j)=solutions{j}.error;
end;

end

function solutions=generate_random_solutions_2(xx,yy,u,num_solutions,tolerance,max_iterations,opts)

u_ifft=real(ifftb(u));
u_norm=sqrt(sum(u_ifft(:).^2));

%fA=figure;
solutions={};
tA=tic;
for ii=1:num_solutions
    fprintf('.'); 
    if (mod(ii,10)==0) 
        fprintf('%d.%d (%g elapsed)\n',opts.jj,ii,toc(tA)); 
        tA=tic;
    end;
    
    opts0=opts;
        
    f=generate_random_solution(xx,yy,u,tolerance,max_iterations,opts0);
    
    %Important to compute this before registration (made mistake in past)
    f2=real(ifftb(pi_2(fftb(f),u,1)));
    f2_proj=pi_1(f2,xx,yy,1);
    XX.f=f;
    XX.resid_img=sqrt(sum((f2(:)-f2_proj(:)).^2))/u_norm;
    
    XX.f_unregistered=XX.f;
    XX.f=register_to_reference(XX.f,opts.f_exact);
    XX.error=sqrt(sum((XX.f(:)-opts.f_exact(:)).^2))/u_norm;
    
    solutions{ii}=XX;
end

end

function f=generate_random_solution(xx,yy,u,tolerance,max_iterations,opts)

u_ifft=real(ifftb(u));
u_norm=sqrt(sum(u_ifft(:).^2));

alpha1=0.9; alpha2=0.95;
%alpha1=1; alpha2=0.999;
beta=1.5;
mu1=alpha1*beta;
mu2=alpha2*mu1/(alpha1-alpha1*alpha2);

if (isfield(opts,'f_init'))
    f=opts.f_init;
elseif isfield(opts,'means')
    fhat=opts.means+randn(size(u)).*opts.stdevs./u;
    f=real(ifftb(fhat));
else
    fhat=u.*(randn(size(u))+i*randn(size(u)));
    f=real(ifftb(fhat));
end;

it=1;
last_resid=0;
num_steps_within_tolerance=0;
while 1
    f2=pi_1(f,xx,yy,alpha1);
    f2hat=fftb(f2);
    f3hat=pi_2(f2hat,u,alpha2);
    f3=real(ifftb(f3hat));
    resid0=mu1*sum(abs(f(:)-f2(:)).^2)/u_norm^2 + ...
           mu2*sum(abs(f(:)-f3(:)).^2)/u_norm^2;
    if (it>1)
        resid_diff=abs(resid0-last_resid);
        if (resid_diff<tolerance)
            num_steps_within_tolerance=num_steps_within_tolerance+1;
        else
            num_steps_within_tolerance=0;
        end;
        if (num_steps_within_tolerance>=10)
            break;
        end;
    end;
    last_resid=resid0;
    if (it>=max_iterations)
        warning('Max # iterations exceeded.');
        break;
    end;
    f=f3;
    it=it+1;
end;

end

function f_new=pi_1(f,xx,yy,alpha1)
mask1=(f>=0)&(abs(xx)<=1)&(abs(yy)<=1);
f_new=alpha1*f.*mask1 + (1-alpha1)*f;
end

function fhat_new=pi_2(fhat,u,alpha2)
ph=angle(fhat);
fhat_new=alpha2*u.*exp(i*ph) + (1-alpha2)*fhat;
end

function d0=apodize(d,sigma,N2)

N2=min(N2,size(d,1));

N=size(d,1);
M=ceil((N+1)/2);
aa=((1:N)-M); [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
if (~isinf(sigma))
    d0=d.*exp(-GR.^2/sigma^2);
else
    d0=d;
end;

d0=d0(M-N2/2:M+N2/2-1,M-N2/2:M+N2/2-1);

d0=d0*(N2/N)^2;

end

function Y=fftb(X)
Y=fftshift(fft2(ifftshift(X)));
end

function Y=ifftb(X)
Y=fftshift(ifft2(ifftshift(X)));
end

function A2=register_to_reference(A,A_ref)

N=size(A_ref,1);
M=ceil((N+1)/2);
[GX,GY]=ndgrid((1:N)-M,(1:N)-M);

tmpA=abs(ifftb(fftb(A).*conj(fftb(A_ref))));
[valA,iiA]=max(tmpA(:));
[i1A,i2A]=ind2sub(size(tmpA),iiA);

tmpB=abs(ifftb(conj(fftb(A)).*conj(fftb(A_ref))));
[valB,iiB]=max(tmpB(:));
[i1B,i2B]=ind2sub(size(tmpB),iiB);

if (valA>valB)
    xx=GX(i1A,i2A);
    yy=GY(i1A,i2A);
    A2=real(ifftb(fftb(A).*exp(2*pi*i*(xx*GX+yy*GY)/N)));
else
    xx=GX(i1B,i2B);
    yy=GY(i1B,i2B);
    A2=real(ifftb(conj(fftb(A)).*exp(2*pi*i*(xx*GX+yy*GY)/N)));
end

%Now do fine adjustment using center of mass, if in the neighborhood
A=A2;

A1_CX=sum(A_ref(:).*GX(:))/sum(A_ref(:));
A1_CY=sum(A_ref(:).*GY(:))/sum(A_ref(:));

A2_CX=sum(A(:).*GX(:))/sum(A(:));
A2_CY=sum(A(:).*GY(:))/sum(A(:));
 
DX=A2_CX-A1_CX;
DY=A2_CY-A1_CY;

if ((abs(DX)<=2)&&(abs(DY)<=2))
    A2=real(ifftb(fftb(A).*exp(2*pi*i*(DX*GX+DY*GY)/N)));
end;

end

function [means,stdevs]=compute_solution_variations(solutions,u)

V=zeros(size(solutions{1}.f,1),size(solutions{1}.f,2),length(solutions));
for j=1:length(solutions)
    V(:,:,j)=fftb(solutions{j}.f);
end;

stdevs=sqrt(var(V,[],3))./u;
means=mean(V,3);

end

