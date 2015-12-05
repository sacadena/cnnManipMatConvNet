function [back, rfSize] = cnn_backward(net,layerNum,loc)
% cnn_backward.m Computes an image in the input space that represents the
% backpropagation gradient of an activation unit with respect to the input
% image. 

% INPUT: 
% net: network to be used
% layerNum: The name of the net layer where the activation unit is
% loc: vector given by [inx, iny, Nch] which are the pixel coordinates of 
%   the activation unit. If blank then it is assumed to be the central pixel 
%   of the first channel feature map. 
%
% OUPUT:
% back: An image with the image-input size to the net containing gradient
%   of the activation unit determined by layerNum and loc with respect to
%   the input.
% rfSize: side of the squared receptive field

% Options settings:
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts = vl_argparse(opts, []);

% Create a net with aprropriate size with ones in data and zeros in
% derivatives
x = single(ones(net.normalization.imageSize));
res = vl_simplenn(net, x, 1);
for l = 1:numel(res)
    res(l).x = single(ones(size(res(l).x)));
    res(l).dzdx = single(zeros(size(res(l).dzdx)));
    %res(l).dzdw = zeros(size(res(l).dzdw));
end

% Set the derivative from interest pixel with value 1.
if nargin < 3
    inx = floor(size(res(layerNum+1).x,1)/2)+1;
    iny = floor(size(res(layerNum+1).x,1)/2)+1;
    Nch = 1;
else
    inx = loc(1);
    iny = loc(2);
    Nch = loc(3);
end

res(layerNum+1).dzdx(inx,iny,Nch) = single(1);

% Do back propagation.
n = layerNum;
%n = numel(net.layers) ;
%res(n+1).dzdx = dzdy ;
for i=n:-1:1
    l = net.layers{i} ;
switch l.type
  case 'conv'
    [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
        vl_nnconv(res(i).x, l.filters, l.biases, ...
                  res(i+1).dzdx, ...
                  'pad', l.pad, 'stride', l.stride) ;
  case 'pool'
    res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
      'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
  case 'normalize'
    res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
  case 'softmax'
    res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
  case 'loss'
    res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
  case 'softmaxloss'
    res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
  case 'relu'
    res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
  case 'noffset'
    res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
  case 'dropout'
    if opts.disableDropout
      res(i).dzdx = res(i+1).dzdx ;
    else
      res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux) ;
    end
  case 'custom'
    res(i) = l.backward(l, res(i), res(i+1)) ;
end
if opts.conserveMemory
  res(i+1).dzdx = [] ;
end
end

back = res(1).dzdx;
rfSize = sqrt(sum(sum(back(:,:,1)~=0)));


