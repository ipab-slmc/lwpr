% TEST_LWPR_1D
%
% Simple script to demonstrate the LWPR algorithm
% We train a model on toy data, and later retrieve
% predictions and confidence bounds from the model


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In the next lines, we create a toy dataset with Ntr=500 samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ntr = 500;
Xtr = 10*rand(Ntr,1);

testfunc = inline('10*sin(7.8*log(1+x))./(1+0.1*x.^2)');

Ytr = 5 + testfunc(Xtr) + 0.1*randn(Ntr,1).*Xtr;

clf
plot(Xtr,Ytr,'r.');
drawnow;
hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we create a new LWPR model  using "lwpr_init"
% The parameters (1,1) mean  1 input dimension,  1 output dimension
%
% Using "lwpr_set", we set appropriate values for model parameters
% The most important parameter is "init_D", the initial distance
% metric that is assigned to new receptive fields (local models).
% In general init_D is a square matrix corresponding to the input
% dimensionality. For 1-D it is just a scalar.
% Smaller values for init_D  yield wider receptive fields, which
% might oversmooth at the start of training and lead to slow 
% convergence.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = lwpr_init(1,1);
model = lwpr_set(model,'init_D',20);
model = lwpr_set(model,'update_D',1);
model = lwpr_set(model,'diag_only',1);
model = lwpr_set(model,'penalty',0.0001);
model = lwpr_set(model,'init_alpha', 40);
%model = lwpr_set(model,'kernel','BiSquare');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% With the following line, we transfer the LWPR model (which is a 
% Matlab struct until now) into MEX-internal memory. This drastically
% speeds up predictions und updates, but has the slight disadvantage
% that its parameters can not be directly accessed anymore.
% After the call to "lwpr_storage",   "model" is effectively a 
% C-pointer (int32 or int64 within Matlab). Never change that variable,
% or you may loose the LWPR model!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = lwpr_storage('Store',model);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we start training the model. We just call "lwpr_update" with one
% input/output tuple after another. The function also returns the
% current prediction for the input data, which we use to keep track
% of the MSE on the training data
%
% For printing how the training proceeds, we also use
%  a) lwpr_num_data   to report the number of training data the model
%                     has seen
%  b) lwpr_num_rfs    to report the number of receptive fields that
%                     have been allocated so far
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:20
   ind = randperm(Ntr);
   mse = 0;
   
   for i=1:Ntr
      [model,yp] = lwpr_update(model,Xtr(ind(i)),Ytr(ind(i)));
	   mse = mse + (Ytr(ind(i),:)-yp)^2;
   end
   
   nMSE = mse/Ntr/var(Ytr,1);
   fprintf(1,'#Data: %5d  #RFs: %3d  nMSE=%5.3f\n',lwpr_num_data(model),lwpr_num_rfs(model),nMSE);   
   if exist('fflush') % for Octave output only
      fflush(1);
   end   
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In the next lines, we retrieve predictions from the LWPR model
% on the regularly sampled interval [0;10]
% This is achieved by calling   "lwpr_predict"
% The first output is the prediction, the second is a 
% one-standard-deviation confidence bound.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ntest = 500;
Xtest = linspace(0,10,Ntest);
Ytest = zeros(Ntest,1);
Conf = zeros(Ntest,1);

for k=1:500
   [Ytest(k), Conf(k)] = lwpr_predict(model,Xtest(k));
end

% Plot the predictions and confidence bounds
plot(Xtest,Ytest,'b-');
hold on
plot(Xtest,Ytest+Conf,'c-');
plot(Xtest,Ytest-Conf,'c-');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% With the following line, we transform the MEX-internal LWPR model
% back into a Matlab struct, and also free the internal memory.
% After this, we can still use "model" to calculate predictions,
% but this would be slower as before. However, we can now access all
% internal variables, which we exploit to visualize the receptive
% fields. 
% Note that this step is also necessary if you wish to save the 
% LWPR model together with the rest of you Matlab workspace, since
% Matlab does not "know" about the internal storage scheme.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = lwpr_storage('GetFree',model);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Any receptive field can be accessed by 
%   model.sub(outDim).rfs(index)
% where outDim is the output dimension (here only =1)  and index 
% selects the RF within that output dimension. 
% Thus, model.sub(1).rfs(5).D   corresponds to the distance metric
% of the 5th receptive field.  The field "c" denotes the center
% of that field,  beta0 is the offset, and beta the PLS regression
% parameter (in 1-D equivalent to +/- the slope of the local model)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = linspace(0,2*pi,50);
ct = cos(t);
st = sin(t);

for k=1:length(model.sub(1).rfs); 
   r = 1/sqrt(model.sub(1).rfs(k).D);
   x = model.sub.rfs(k).c;
   y = model.sub.rfs(k).beta0;
   b = model.sub.rfs(k).beta * model.sub.rfs(k).U;
   plot(x+r*ct,r*st,'g-');
   plot([x-r,x+r],[y-r*b,y+r*b],'k-');
end

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
