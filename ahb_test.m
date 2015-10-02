% Fienup's iteration in 2D for phase retrieval, as in Miao 1998. Barnett 9/16/14
clear; close all;
n = 256;  % pixels on size of image support box
h = 1/n;  % physical grid spacing
x = (-n/2+.5:n/2-.5)*h; [xx yy] = meshgrid(x);  % physical coords in (-1/2,1/2)
a = 2.5;  % upsampling (padding) ratio for going to Fourier - no idea how to set
N = ceil(a*n); if mod(N,2), N = N+1; end  % make even
%ue = cos(20*sqrt((xx+.2).^2+(yy+.1).^2)).^2; % exact Re non-neg img u
%ue(xx>0. & xx<.35 & yy>-.1 & yy<.2) = 2;     % add hard-edged box
ue=exp((-xx.^2-yy.^2)/0.05^2); ue=ue+randn(size(ue))*0.00000000000001;
p = (-N/2+.5:N/2-.5)*h; [xp yp] = meshgrid(p); % padded physical 1d grid
ii = (xp>=-.5 & xp<.5 & yp>=-.5 & yp<.5);  % ii=boolean whether in support box
uep = zeros(N,N); uep(ii) = ue;  % u exact padded: note image centered on N/2+1
%uepi = uep([1 N:-1:2],[1 N:-1:2]);   % uep inverted through origin (N/2+1)
uepi = uep(N:-1:1,N:-1:1);

U = abs(fft2(fftshift(uep)));  % the meas data
%U = U.*(1+0.01*randn(size(U))); U = (U+U([1 N:-1:2],[1 N:-1:2]))/2; % noise?
f = fftshift(ifft2(U.*exp(2i*pi*rand(N,N))));  % init f, best guess to image
f = real(f);   % real image case (iteration preserves reality)
be = 0.5;      % decay const beta, in (.5,1) apparently. 0.5 better than 0.8
maxit = 1e3;   % how many iters
[kxx kyy] = meshgrid(-N/2:N/2-1,-N/2:N/2-1); rr2=fftshift(kxx.^2+kyy.^2); kw = nan; % for k filter
figure(2); set(gcf,'position',[200 200 1000 800]);
kb = 40; ib = N/2+1+(-kb:kb); k=-kb:kb; % k box half-size to show
tic;
for j=1:maxit  % ....... Fienup iteration
  F = fft2(fftshift(f));    % step 1   (note fft data has zero-freq at index 1)
  F2 = U.*F./abs(F);        % 2: use phase info from current f with known ampl
  %kw = 40; filter=exp(-.5*rr2/kw^2); F2 = F2.*filter; % fails for any width!
  f2 = fftshift(ifft2(F2)); % 3
  iv = ii & (f2>=0);        % boolean for if pixels "valid" (in box & +ve)

  if 1&mod(j,1)==0           % show everything every few iters...
    subplot(2,3,1); imagesc(x,x,ue); vx=caxis; axis xy equal tight;
    title('true u in box'); subplot(2,3,4); t=fftshift(U);
    imagesc(k,k,t(ib,ib)); axis xy equal tight; vk=caxis;
    title('data U(k) (zoom)');
    subplot(2,3,2); imagesc(x,x,reshape(f2(ii),[n n])); caxis(vx);
    axis xy equal tight; title(sprintf('Re f'' in box, iter %d',j));
    subplot(2,3,5); t=abs(fftshift(F));
    imagesc(k,k,t(ib,ib)); axis xy equal tight; caxis(vk);
    title(sprintf('|F(k)| (zoom): k_w=%.3g',kw));
    e = norm(f2(:)-uep(:))/N; ei = norm(f2(:)-uepi(:))/N; % rms err inv & not
    if e<ei, fe = f2-uep; else, fe=f2-uepi; end
    subplot(2,3,3); imagesc(p,p,fe); axis xy equal tight; v=max(abs(caxis));
    caxis(v*[-1 1]); colorbar; title('f'' err (full array)');
    subplot(2,3,6); E = norm(f2(~iv))/norm(f2(iv)); semilogy(j,E,'.'); hold on;
    semilogy(j,min(e,ei),'r.'); axis([0 maxit 1e-5 1]); legend('E','u rms err');
    pause(1);
  drawnow, end
    
  f(iv) = f2(iv);                % 4, plain update valid pixels
  %f(~iv) = 0;                    % 4, kill invalid pixels - v slow
  f(~iv) = f(~iv) - be*f2(~iv);  % 4, update invalid pixels, Miao variant
end, toc