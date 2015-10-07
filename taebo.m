function f=taebo(xx,yy,u,opts)

if (nargin==0) taebo_test; return; end;

opts.max_iterations=20000;
opts.tol=1e-8;
the_best=taebo_find_best(xx,yy,u,opts,30,300);

% opts.max_iterations=5000;
% opts.tol=1e-5;
% the_best=taebo_find_best(xx,yy,u,opts,60,500);
% opts.tol=1e-8;
% the_best=taebo_find_best(xx,yy,u,opts,60,the_best);

V=zeros(length(u(:)),length(the_best));
resids=zeros(1,length(the_best));
for j=1:length(the_best)
    resids(j)=the_best{j}.resid;
end;
[~,sort_inds]=sort(resids);
for j=1:length(the_best)
    V(:,j)=the_best{sort_inds(j)}.f(:);
end;

V(:,end+1)=opts.f_exact(:);

MM=corrcoef(V);

figure; imagesc(MM); colormap('gray');

disp('test');

%FF=ss_eventfeatures(reshape(V,1,size(V,1),size(V,2)));

%figure; plot(resids,FF(1,:),'b.',resids,FF(2,:),'r.',resids,FF(3,:),'g.');

%FF=FF(1:3,:);
%labels=isosplit(FF);
%FF(3,:)=resids;
%ss_view_clusters(FF,labels);


% opts.tol=1e-5;
% the_best=taebo_find_best(xx,yy,u,opts,20,the_best);
% 
% opts.tol=1e-7;
% the_best=taebo_find_best(xx,yy,u,opts,10,the_best);
% 
% opts.tol=1e-9;
% the_best=taebo_find_best(xx,yy,u,opts,1,the_best);

end

function the_best=taebo_find_best(xx,yy,u,opts,num_best,num_tries_or_init)

if (iscell(num_tries_or_init))
    init=num_tries_or_init;
    num_tries=length(init);
else
    num_tries=num_tries_or_init;
    init={};
end;

%Set algorithm parameters
alpha1=0.5; alpha2=0.5;
beta=1.5;
mu1=alpha1*beta;
mu2=alpha2*mu1/(alpha1-alpha1*alpha2);
tol=opts.tol;

%fA=figure;
%fB=figure;
fC=figure;
fD=figure;

the_best={};

f=opts.f_exact;
f2=pi_1(f,xx,yy,alpha1);
f2hat=fftb(f2);
f3hat=pi_2(f2hat,u,alpha2);
f3=real(ifftb(f3hat));
exact_resid=mu1*sum(abs(f(:)-f2(:)).^2)/length(f(:)) + mu2*sum(abs(f2hat(:)-f3hat(:)))/length(f2hat(:));
f_tmp=pi_1(ifftb(pi_2(fftb(opts.f_exact),u,1)),xx,yy,1);
exact_pos_error=sum(abs(opts.f_exact(:)-f_tmp(:)).^2);
fprintf('exact resid = %g\n',exact_resid);
fprintf('exact pos error = %g\n',exact_pos_error);

resids=ones(1,num_tries)*inf;
for tt=1:num_tries
    fprintf('.');
    if (mod(tt,10)==0) fprintf('%d \n',tt); end;
    
    %Initialize with a random guess
    if (tt<=length(init))
        f=init{tt}.f;
    else
        %f=opts.f_exact;
        f=randn(size(u));
    end;
        
    resids_this_try=[];
    changes=[];
    it=1;
    fig_timer=tic;
    while it<=opts.max_iterations
        f2=pi_1(f,xx,yy,alpha1);
        f2hat=fftb(f2);
        f3hat=pi_2(f2hat,u,alpha2);
        f3=real(ifftb(f3hat));
        if (mean(f3(:))<0) f3=-f3; end; %Not sure why this necessary. We seem to be getting negative of the correct image much of the time!
        
        resid0=mu1*sum(abs(f(:)-f2(:)).^2)/length(f(:)) + mu2*sum(abs(f2hat(:)-f3hat(:)))/length(f2hat(:));
        resids_this_try(it)=resid0;
        if (it>10) changes(it)=1-abs((resids_this_try(it)/resids_this_try(it-10))^(1/10));
        else changes(it)=inf; end;
        
        if ((it==1)||(toc(fig_timer)>1))
            %figure(fA);
            %semilogy(1:it,resids_this_try,'b'); hold on;
            %semilogy(1:it,changes,'r');
            %hold off;
            %figure(fB); imagesc(f3); colormap('gray');
            %drawnow;
            %fig_timer=tic;
        end;
        
        f=f3;
        
        it=it+1;
        if (changes(it-1)<tol) break; end;
    end;    
    it=it-1;
    
    f=register_to_reference(f,opts.f_exact);
    f_tmp=pi_1(ifftb(pi_2(fftb(f),u,1)),xx,yy,1);
    pos_error=sum(abs(f(:)-f_tmp(:)).^2);
    
    resid0=resids_this_try(it);
    resids(tt)=resid0;
    XX.f=f;
    XX.resid=resid0;
    XX.pos_error=pos_error;
    if (length(the_best)<num_best)
        the_best{end+1}=XX;
    else
        worst_ind=1;
        worst_resid=0;
        for j=1:length(the_best)
            if (the_best{j}.resid>worst_resid)
                worst_ind=j; worst_resid=the_best{j}.resid;
            end;
        end;
        if (resid0<worst_resid)
            the_best{worst_ind}=XX;
        end;
    end;
    
    figure(fC); hist(resids(1:tt),1000); drawnow;
    
    best_ind=1;
    best_resid=inf;
    for j=1:length(the_best)
        if (the_best{j}.resid<best_resid)
            best_ind=j; best_resid=the_best{j}.resid;
        end;
    end;
    figure(fD);
    imagesc(the_best{best_ind}.f); colormap('gray'); title(sprintf('%g (%g)\n',the_best{best_ind}.resid,the_best{best_ind}.pos_error));    
end;
fprintf('\n');

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

