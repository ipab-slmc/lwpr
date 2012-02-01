function model = test_lwpr_2D

n = 500;

% a random training set using the CROSS function
X = (rand(n,2)-.5)*2;
Y = max([exp(-X(:,1).^2 * 10),exp(-X(:,2).^2 * 50),1.25*exp(-(X(:,1).^2+X(:,2).^2)*5)]');
Y = Y' + randn(n,1)*0.1;

% a systematic test set on a grid
Xt = [];
for i=-1:0.05:1,
	for j=-1:0.05:1,
		Xt = [Xt; i j];
	end
end
Yt = max([exp(-Xt(:,1).^2 * 10),exp(-Xt(:,2).^2 * 50),1.25*exp(-(Xt(:,1).^2+Xt(:,2).^2)*5)]');
Yt = Yt';

tic


% initialize LWPR
model = lwpr_init(2,1,'name','lwpr_test');

model = lwpr_set(model,'init_D',[25 0; 0 25]);    
model = lwpr_set(model,'init_alpha',ones(2)*250);
model = lwpr_set(model,'w_gen',0.2);
model = lwpr_set(model,'diag_only',0);   
model = lwpr_set(model,'meta',1);
model = lwpr_set(model,'meta_rate',250);
model = lwpr_set(model,'kernel','Gaussian');   

%%%%%%%%%%%%%%%%%%%%%%%%
%  Transfer model into mex-internal storage
   model = lwpr_storage('Store',model);
%%%%%%%%%%%%%%%%%%%%%%%%

% train the model
for j=1:20
   inds = randperm(n);

   mse = 0;
   for i=1:n,
	   [model,yp,w] = lwpr_update(model,X(inds(i),:)',Y(inds(i),:)');         
	   mse = mse + (Y(inds(i),:)-yp).^2;
   end

   nMSE = mse/n/var(Y,1);
   fprintf(1,'#Data=%d #rfs=%d nMSE=%5.3f\n',lwpr_num_data(model),lwpr_num_rfs(model),nMSE);
   if exist('fflush') % for Octave output only
      fflush(1);
   end   
end


% create predictions for the test data
Yp = zeros(size(Yt));
for i=1:length(Xt),
	[yp,w]=lwpr_predict(model,Xt(i,:)',0.001);
	Yp(i,1) = yp;
end
%[yp,w]=lwpr_predict(model,Xt',0.001);
%Yp = yp';

ep   = Yt-Yp;
mse  = mean(ep.^2);
nmse = mse/var(Yt,1);

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Transfer model back from mex-internal storage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = lwpr_storage('GetFree',model);

figure(1);
clf;

% plot the raw noisy data
subplot(2,2,1);
plot3(X(:,1),X(:,2),Y,'*');
title('Noisy data samples');

% plot the fitted surface
axis([-1 1 -1 1 -.5 1.5]);
subplot(2,2,2);
[x,y,z]=makesurf([Xt,Yp],sqrt(length(Xt)));
if ~exist('surfl')
   mesh(x,y,z);
else
   surfl(x,y,z);
end
axis([-1 1 -1 1 -.5 1.5]);
title(sprintf('The fitted function: nMSE=%5.3f',nmse));

% plot the true surface
subplot(2,2,3);
[x,y,z]=makesurf([Xt,Yt],sqrt(length(Xt)));
if ~exist('surfl')
   mesh(x,y,z);
else
   surfl(x,y,z);
end
axis([-1 1 -1 1 -.5 1.5]);
title('The true function');

% plot the local models
subplot(2,2,4);
for i=1:length(model.sub(1).rfs),
	draw_ellipse(model.sub(1).rfs(i).D,model.sub(1).rfs(i).c,0.1,'Gaussian');
	hold on;
end
hold off;
axis('equal');
title('Input space view of RFs');

% --------------------------------------------------------------------------------
function [X,Y,Z]=makesurf(data,nx)
% [X,Y,Z]=makesurf(data,nx) converts the 3D data file data into
% three matices as need by surf(). nx tells how long the row of the
% output matrices are

[m,n]=size(data);

n=0;
for i=1:nx:m,
	n = n+1;
	X(:,n) = data(i:i+nx-1,1);
	Y(:,n) = data(i:i+nx-1,2);
	Z(:,n) = data(i:i+nx-1,3);
end;


% --------------------------------------------------------------------------------
function []=draw_ellipse(M,C,w,kernel)
% function draw ellipse draws the ellipse corresponding to the
% eigenvalues of M at the location c.

[V,E] = eig(M);

E = E;
d1 = E(1,1);
d2 = E(2,2);

steps = 50;
switch kernel
case 'Gaussian'
	start = sqrt(-2*log(w)/d1);
case 'BiSquare'
	start = sqrt(4*(1-sqrt(w))/d1);
end


for i=0:steps,
	Xp(i+1,1) = -start + i*(2*start)/steps;
	switch kernel
	case 'Gaussian'
		arg = (-2*log(w)-Xp(i+1,1)^2*d1)/d2;
	case 'BiSquare'
		arg = (2*(1-sqrt(w))-Xp(i+1,1)^2*d1)/d2;
	end
	if (arg < 0), 
		arg = 0; 
	end; % should be numerical error
	Yp(i+1,1) = sqrt(arg);
end;

for i=1:steps+1;
	Xp(steps+1+i,1) = Xp(steps+1-i+1,1);
	Yp(steps+1+i,1) = -Yp(steps+1-i+1,1);
end;

% transform the rf

M = [Xp,Yp]*V(1:2,1:2)';

Xp = M(:,1) + C(1);
Yp = M(:,2) + C(2);

plot(C(1),C(2),'ro',Xp,Yp,'c');

% LWPR: A MATLAB library for incremental online learning
% Copyright (C) 2007  Stefan Klanke, Sethu Vijayakumar, Stefan Schaal
% Contact: sethu.vijayakumar@ed.ac.uk
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Library General Public
% License along with this library; if not, write to the Free
% Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
