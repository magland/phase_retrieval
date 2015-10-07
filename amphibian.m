function f=amphibian(xx,yy,u,opts)

if (nargin==0) amphibian_test; return; end;

num_solutions=30;
num_threads=6;
total_num_solutions=num_solutions*num_threads;
tolerance=1e-8;
max_iterations=50000;
solution_sets={};
parfor ct=1:num_threads
    opts0=opts;
    opts0.jj=ct;
    solutions0=generate_random_solutions(xx,yy,u,num_solutions,tolerance,max_iterations,opts0);
    solution_sets{ct}=solutions0;
end;

solutions={};
for ct=1:num_threads
    SS=solution_sets{ct};
    for j=1:num_solutions
        solutions{(ct-1)*num_solutions+j}=SS{j};
    end;
end;

u_ifft=real(ifftb(u));
u_norm=sqrt(sum(u_ifft(:).^2));

epsilons=zeros(1,total_num_solutions);
for j=1:total_num_solutions
    epsilons(j)=solutions{j}.closeness;
end;
epsilons_at_10=zeros(1,total_num_solutions);
epsilons_at_30=zeros(1,total_num_solutions);
epsilons_at_50=zeros(1,total_num_solutions);
epsilons_at_70=zeros(1,total_num_solutions);
for j=1:total_num_solutions
    epsilons_at_10(j)=solutions{j}.closeness_at(1);
    epsilons_at_30(j)=solutions{j}.closeness_at(2);
    epsilons_at_50(j)=solutions{j}.closeness_at(3);
    epsilons_at_70(j)=solutions{j}.closeness_at(4);
end;
[epsilons,sort_inds]=sort(epsilons);
epsilons_at_10=epsilons_at_10(sort_inds);
epsilons_at_30=epsilons_at_30(sort_inds);
epsilons_at_50=epsilons_at_50(sort_inds);
epsilons_at_70=epsilons_at_70(sort_inds);
sorted_solutions={};
for j=1:total_num_solutions
    sorted_solutions{j}=solutions{sort_inds(j)};
end;
solutions=sorted_solutions;

errors=zeros(1,total_num_solutions);
for j=1:total_num_solutions
    errors(j)=solutions{j}.error;
end;
est_epsilons=zeros(1,40);
est_epsilons(1)=epsilons(1);
for j=2:length(est_epsilons)
    f1=solutions{j}.f;
    val=0;
    for k=1:j-1
        f2=solutions{k}.f;
        dist=sqrt(sum((f1(:)-f2(:)).^2))/u_norm;
        val=max(dist,val);
    end;
    est_epsilons(j)=max(est_epsilons(j-1),val);
end;

figure; plot(1:total_num_solutions,epsilons,'b.');
title('Closeness');

figure; plot(epsilons_at_10,epsilons_at_30,'b.');
xlabel('Closeness at a=10');
ylabel('Closeness at a=30');

figure; plot(epsilons_at_30,epsilons_at_50,'b.');
xlabel('Closeness at a=30');
ylabel('Closeness at a=50');

figure; plot(epsilons_at_50,epsilons_at_70,'b.');
xlabel('Closeness at a=50');
ylabel('Closeness at a=70');

figure; plot(epsilons_at_70,epsilons,'b.');
xlabel('Closeness at a=70');
ylabel('Closeness at full');

figure; plot(5:length(est_epsilons),est_epsilons(5:end),'k.');
title('Est epsilons');

figure; plot(1:total_num_solutions,errors,'r.');
title('Error');

figure; plot(epsilons,errors,'k.'); xlabel('closeness'); ylabel('error');

figure; hist(epsilons,200);

f=solutions{1}.f;

save test.mat

end

function [solutions]=generate_random_solutions(xx,yy,u,num_solutions,tolerance,max_iterations,opts)

u_ifft=real(ifftb(u));
u_norm=sqrt(sum(u_ifft(:).^2));

%fA=figure;
best_closeness=inf;
solutions={};
tA=tic;
for ii=1:num_solutions
    fprintf('.'); 
    if (mod(ii,10)==0) 
        fprintf('%d.%d (%g elapsed)\n',opts.jj,ii,toc(tA)); 
        tA=tic;
    end;
    
    opts0=opts;
    list=[20,40,60,80];
    for jj=1:length(list)
        f=generate_random_solution(xx,yy,apodize(u,list(jj)),tolerance,max_iterations,opts0);
        f=register_to_reference(f,opts.f_exact);
        f2=real(ifftb(pi_2(fftb(f),u,1)));
        f2_proj=pi_1(f2,xx,yy,1);
        XX.closeness_at(jj)=sqrt(sum((f2(:)-f2_proj(:)).^2))/u_norm;
        opts0.f_init=f;
    end;
        
    f=generate_random_solution(xx,yy,u,tolerance,max_iterations,opts0);
    f=register_to_reference(f,opts.f_exact);
    f2=real(ifftb(pi_2(fftb(f),u,1)));
    f2_proj=pi_1(f2,xx,yy,1);
    XX.f=f2;
    XX.closeness=sqrt(sum((f2(:)-f2_proj(:)).^2))/u_norm;
    XX.error=sqrt(sum((f(:)-opts.f_exact(:)).^2))/u_norm;
    solutions{ii}=XX;
    
%     if (XX.closeness<best_closeness)
%         best_closeness=XX.closeness;
%         best_f=f;
%         %figure(fA); imagesc(best_f); colormap('gray'); 
%         %title(sprintf('closeness=%g, error=%g',XX.closeness,XX.error));
%         %drawnow;
%     end;
end


end

function f=generate_random_solution(xx,yy,u,tolerance,max_iterations,opts)

u_ifft=real(ifftb(u));
u_norm=sqrt(sum(u_ifft(:).^2));

alpha1=0.9; alpha2=0.95;
beta=1.5;
mu1=alpha1*beta;
mu2=alpha2*mu1/(alpha1-alpha1*alpha2);

if (isfield(opts,'f_init'))
    f=opts.f_init;
else
    f=randn(size(u));
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

function d0=apodize(d,sigma)

N=size(d,1);
M=ceil((N+1)/2);
aa=((1:N)-M); [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
d0=d.*exp(-GR.^2/sigma^2);

end