% p_test.m

%% Loading net

NET_NAME = 'imagenet-vgg-verydeep-19.mat';
%NET_NAME = 'imagenet-caffe-ref.mat';
run('/Users/santiagoandrescadenaceron/Documents/MATLAB/matconvnet-1.0-beta9/matlab/vl_setupnn');
net = load(NET_NAME) ;

%% Loading an Image and Normalizing 

IMAGE_NAME = 'car.jpg';

im = imread(IMAGE_NAME); % Read Image
im_ = single(im);       % Single type.
im_ = imresize(im_, net.normalization.imageSize(1:2)) ; % Resizing for net
im_ = im_ - net.normalization.averageImage ; % Substracting the mean image

%% Simple Prediction

res = vl_simplenn(net, im_) ;
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.classes.description{best}, best, bestScore)) ;

%% Number of paricular Layer name:

lName = 'conv3_1'; % Name of a layer of interest
lNum = 0;
for i = 1:numel(net.layers)
    if strcmp(net.layers{i}.name, lName)
        lNum = i;    % Number of the layer
        break
    end
end

%% Visualization of Filters

% If there are only 3 channels (like in the first convolutional layer:
vis_square(permute(net.layers{1}.filters,[4,1,2,3]));

% If there are more than 3 channels the filters are viewed as different
% filters in each row. The row is formed by the different channels of the
% filter. For example, here I image the filters of layer lNum. Only the
% first NumChannles are shown where NCh are the 

[h,w,NCh,fo] = size(net.layers{lNum}.filters);
foo = permute(net.layers{lNum}.filters,[4,1,2,3]);
vis_square(reshape(permute(foo(1:NCh,:,:,:),[4,1,2,3]),NCh^2,h,w));


%% Calculation of output of particular layer

out = cnn_forward(net, im_, lNum);

%% Visualization of Feature Maps 
foo = permute(out(:,:,1:100),[3,1,2]);
vis_square(foo);

%% Noise input Data set
Nstim = 1000;
noise = 255*rand([net.normalization.imageSize(1:2),3,Nstim]);
noise = single(noise);
noise = noise - repmat(net.normalization.averageImage,1,1,1,Nstim) ; % Normalization

%% Record output of particular feature maps given noise stimuli

% Takes some of the feature maps and store responses given certain stimuli.

load('unifRandStimVGG.mat'); % Load uniformly distributed normalized noise in ns
tic
% Record output every 10  noisy stimuli and stores only the feature maps in vector fm
fm = [1,2];
out = cnn_forward(net, ns(:,:,:,1), lNum);
fmaps = zeros(size(out,1),size(out,2),length(fm),size(ns,4));

freqRec = 10;
n = size(ns,4);
for i = 1:freqRec:n-(freqRec-1)
    ind = i:i+freqRec-1;
    out = cnn_forward(net, ns(:,:,:,ind), lNum);
    fmaps(:,:,:,ind) = out(:,:,fm,:);
end

toc

%foo = permute(fmaps(:,:,:,10),[3,1,2]);
%vis_square(foo);


%% Selection of activation unit on the feature maps

clear ;close all;clc

load('unifRandStimVGG.mat'); % Stimulus
% 'ns' as the variable containing the noise stimulus
[h,w,NCh,n] = size(ns);
X = reshape(ns,h*w*NCh,n); % Examples in columns
clear ns; % Clear memory


load('featMapsVGG_Conv1_1.mat'); % Feature maps extracted from conv1_1:
% 'fm' The number of f.Maps. 
% 'fmaps': Values

% arbitrary selection of activation units to analyze
pInd_1 = [1,1,1]; % [hight, width, feature map index]
pInd_2 = [round(size(fmaps,1)/2),round(size(fmaps,2)/2),1];
pInd_3 = [round(size(fmaps,1)/2),round(size(fmaps,2)/2),2];
pInd_4 = [round(size(fmaps,1)/2),round(size(fmaps,2)/2),3];
pInd_5 = [round(size(fmaps,1)/2),round(size(fmaps,2)/2),4];

acUnit_1 = squeeze(fmaps(pInd_1(1),pInd_1(2),pInd_1(3),:));
acUnit_2 = squeeze(fmaps(pInd_2(1),pInd_2(2),pInd_2(3),:));
acUnit_3 = squeeze(fmaps(pInd_3(1),pInd_3(2),pInd_3(3),:));
acUnit_4 = squeeze(fmaps(pInd_4(1),pInd_4(2),pInd_4(3),:));
acUnit_5 = squeeze(fmaps(pInd_5(1),pInd_5(2),pInd_5(3),:));

clear fmaps; % clear memory.

%% Receptive Field Estimation by Averaging stimuli

y = acUnit_3;
thr = std(y)/2;
v_Rfield = mean(X(:,find(y>thr)'),2);
RField = reshape(v_Rfield,h,w,NCh);
RFmean = squeeze(mean(mean(RField)));
figure, imagesc(RField(:,:,1)-RFmean(1)),colormap 'gray'




%% Receptive field sizes across layers

rfSize = zeros(1,37);
for i = 1:37
    [~, rfSize(i)] = cnn_backward(net,i);
    fprintf('RF size of layer %d is %d\n',i,rfSize(i))
end

%% Distance between adjacent activation units RF across feature maps

rfStride = zeros(1,37);
for i = 1:37
    out = cnn_forward(net, single(ones(size(im_))), i);
    ix=floor(size(out,1)/2)+1;
    back = cnn_backward(net,i,[ix,ix,1]);
    a =find(back(:,:,1)~=0);
    back = cnn_backward(net,i,[ix+1,ix+1,1]);
    [a1,b1] =find(back(:,:,1)~=0);
    rfStride(i)=a1(1)-a(1);
    fprintf('RF stride of layer %d is %d\n',i,rfStride(i))
end

%% Indices of Complete RF in feature maps
inp = single(ones(size(im_)));
lNum;
out = cnn_forward(net,inp,lNum);
[r,c,~]=size(out);
back = cnn_backward(net,inp, 1, lNum,floor(r/2)+1,floor(c/2)+1);
[a,b] =find(back.dzdx(:,:,1)~=0);
temp1 = floor(a(1)/rfStride(lNum));
if rfStride(lNum)==1  
    indCompRf = floor(r/2)+1-temp1+1:1:r-(floor(r/2)-temp1+1);
else
    indCompRf = floor(r/2)+1-temp1:1:r-(floor(r/2)-temp1);
end

%% Indices of RF input Image

back = cnn_backward(net, inp, 1, lNum,indCompRf(1),indCompRf(1));
[a,b] =find(back.dzdx(:,:,1)~=0);
back = cnn_backward(net, inp, 1, lNum,indCompRf(end),indCompRf(end));
[aEnd,bEnd] =find(back.dzdx(:,:,1)~=0);
indRf = a(1):rfStride(lNum):aEnd(1);

%% Noise Gaussian
sigmaNs = 125/1.645;
Nstim = 100;
tic,
ns = single(sigmaNs*randn([net.normalization.imageSize(1:2),3,Nstim]));
toc

%% Receptive Fields
lNum;
rfSize;
rfStride;
indRf;
indCompRf;
RF = zeros(rfSize(lNum),rfSize(lNum),3);
%tempSum = zeros(1,1,rfSize(lNum));
tempSum = 0;
for i = 1:100
    out = cnn_forward(net, ns(:,:,:,i), lNum);
    for ix = 1:length(indRf)
        for iy  = 1:length(indRf)
            aUnits = out(indCompRf(ix),indCompRf(iy),1);
            stim = ns(indRf(ix):indRf(ix)+rfSize(lNum)-1,indRf(iy):indRf(iy)+rfSize(lNum)-1,:,i);
            tempSum = tempSum + aUnits;
            RF = RF + aUnits*stim;
            %RF = RF + bsxfun(@times,aUnits,stim);
            
        end
    end
    display('One stimulus done')
end

%RF = bsxfun(@rdivide,RF,tempSum);
RF = RF/tempSum;


RFscaled = RF;
RFscaled(:,:,1) = RFscaled(:,:,1)-min(min(RFscaled(:,:,1)));
RFscaled(:,:,2) = RFscaled(:,:,2)-min(min(RFscaled(:,:,2)));
RFscaled(:,:,3) = RFscaled(:,:,3)-min(min(RFscaled(:,:,3)));
RFscaled(:,:,1) = RFscaled(:,:,1)/max(max(RFscaled(:,:,1)));
RFscaled(:,:,2) = RFscaled(:,:,2)/max(max(RFscaled(:,:,2)));
RFscaled(:,:,3) = RFscaled(:,:,3)/max(max(RFscaled(:,:,3)));

figure, imagesc(RFscaled)
% 
%% === Getting samples from two layers belonging to two different networks.
% clear;close all;clc
% NET_NAME = 'imagenet-vgg-verydeep-19.mat';
% run('/Users/santiagoandrescadenaceron/Documents/MATLAB/matconvnet-1.0-beta9/matlab/vl_setupnn');
% net = load(NET_NAME) ;
% NET_NAME2 = 'imagenet-caffe-ref.mat';
% net2 = load(NET_NAME2);
%% 
tic
% Pick layers from each network for future regression
% Layer numbers
lName1 = 'conv1_2';
lName2 = 'conv3';    
for num = 1:numel(net.layers)
    if strcmp(net.layers{num}.name, lName1)
        lNum1 = num;    % Number of the layer
        break
    end
end
for num = 1:numel(net2.layers)
    if strcmp(net2.layers{num}.name, lName2)
        lNum2 = num;    % Number of the layer
        break
    end
end

% Forward random images through each of the nets:
imagesNames = dir('Images/');
ind = randperm(numel(imagesNames));  % Obtain random indices of images
ind(ind==1)=[];

% pick Nimag random images from imagenet dataset
Nimag = 30;
uMatU_X = [];
uMatU_Y = [];
for i = 1:Nimag
    IMAGE_NAME = strcat('Images/',imagesNames(ind(i)).name); %Name of Image
    im = imread(IMAGE_NAME); % Read Image
    % Preprocess Images
    im_ = single(im);       %  VGG
    im_ = imresize(im_, net.normalization.imageSize(1:2)) ; % Resizing for net
    im_ = im_ - net.normalization.averageImage ; % Substracting the mean image
    
    im_2 = single(im);       % ALEX
    im_2 = imresize(im_2, net2.normalization.imageSize(1:2)) ; % Resizing for net
    im_2 = im_2 - net2.normalization.averageImage ; % Substracting the mean image

    % Compute output
    out1 = cnn_forward(net, im_, lNum1); % For VGG
    out2 = cnn_forward(net2, im_2, lNum2);% For ALEX
    
    fmSize1 = size(out1,1);
    fmSize2 = size(out2,1);
    depth1 = size(out1,3);
    depth2 = size(out2,3);
    % size of patch of VGG feature map predicting units in Alex 
    sizePatch = floor(fmSize1/fmSize2);
    % indices in VGG feature map correspongin to units in ALEX
    indices = round(1:(fmSize1/fmSize2):fmSize1);
    indices(end)=indices(end)-1;
    unitsMatrix1 = zeros((sizePatch^2)*depth1,fmSize2^2); % In VGG
    unitsMatrix2 = zeros(depth2,fmSize2^2);
    count = 0;
    for ix = 1:length(indices)
        for iy = 1:length(indices)
            temp1 = out1(indices(ix):indices(ix)+sizePatch-1,indices(iy):indices(iy)+sizePatch-1,:);
            temp2 = out2(ix,iy,:);
            count = count+1;
            unitsMatrix1(:,count) = temp1(:);
            unitsMatrix2(:,count) = temp2(:);
        end
    end   
    uMatU_X = [uMatU_X,unitsMatrix1];
    uMatU_Y = [uMatU_Y,unitsMatrix2];
    fprintf('Image number %d\n',i)
end
toc

%% Do multivariate linear regression
X = [ones(size(uMatU_X,2),1),uMatU_X'];
Y = uMatU_Y';
% tic
% [beta,Sigma,E,CovB,logL] = mvregress(X,Y);
% toc
filename = strcat('Regression/VGG_',lName1,'_Alex_',lName2);
save(filename,'X','Y','Nimag');

