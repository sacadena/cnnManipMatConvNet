% rfields.m
NET_NAME = 'imagenet-vgg-verydeep-19.mat';
%NET_NAME = 'imagenet-caffe-ref.mat';
run('/Users/santiagoandrescadenaceron/Documents/MATLAB/matconvnet-1.0-beta9/matlab/vl_setupnn');
net = load(NET_NAME) ;

lNames = {'pool1','pool2','pool3','pool4'};
for nam = 1:length(lNames)
lName = lNames{nam}; % Name of a layer of interest
lNum = 0;
for i = 1:numel(net.layers)
    if strcmp(net.layers{i}.name, lName)
        lNum = i;    % Number of the layer
        break
    end
end

load('RfParamsVgg19.mat');

%% Indices of Complete RF in feature maps
inp = single(ones(net.normalization.imageSize));
out = cnn_forward(net,inp,lNum);
[r,c,~]=size(out);
back = cnn_backward(net,inp, 1, lNum,floor(r/2)+1,floor(c/2)+1);
[a,~] =find(back.dzdx(:,:,1)~=0);
temp = floor(a(1)/rfStride(lNum));
if rfStride(lNum)==1  
    indCompRf = floor(r/2)+1-temp+1:1:r-(floor(r/2)-temp+1);
else
    indCompRf = floor(r/2)+1-temp:1:r-(floor(r/2)-temp);
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
ns = single(sigmaNs*randn([net.normalization.imageSize(1:2),3,Nstim]));

%% Receptive Fields

%[~,~,~,numRF] = size(net.layers{lNum}.filters); 
out = cnn_forward(net, inp, lNum);
numRF = size(out,3);
RFMat = zeros(rfSize(lNum),rfSize(lNum),3,numRF);
for indR = 1:numRF
    RF = zeros(rfSize(lNum),rfSize(lNum),3);
    tempSum = 0;
    for i = 1:100
        out = cnn_forward(net, ns(:,:,:,i), lNum);
        for ix = 1:length(indRf)
            for iy  = 1:length(indRf)
                aUnits = out(indCompRf(ix),indCompRf(iy),indR);
                stim = ns(indRf(ix):indRf(ix)+rfSize(lNum)-1,indRf(iy):indRf(iy)+rfSize(lNum)-1,:,i);
                tempSum = tempSum + aUnits;
                RF = RF + aUnits*stim;       
            end
        end
        fprintf('Receptive Field number %d iteration number %d\n',indR,i)
    end
    RF = RF/tempSum;
    RFMat(:,:,:,indR)= RF;
end
FILENAME = strcat('RF_VGG19_',lName);
save(FILENAME,'RFMat');
end


%% RScaling
RF = load('RF_VGG19_conv4_1.mat');
RF = RF.RFMat;
RFsc = zeros(size(RF));
for i = 1:size(RF,4)
RFscaled = RF(:,:,:,i);
RFscaled(:,:,1) = RFscaled(:,:,1)-min(min(RFscaled(:,:,1)));
RFscaled(:,:,2) = RFscaled(:,:,2)-min(min(RFscaled(:,:,2)));
RFscaled(:,:,3) = RFscaled(:,:,3)-min(min(RFscaled(:,:,3)));
RFscaled(:,:,1) = RFscaled(:,:,1)/max(max(RFscaled(:,:,1)));
RFscaled(:,:,2) = RFscaled(:,:,2)/max(max(RFscaled(:,:,2)));
RFscaled(:,:,3) = RFscaled(:,:,3)/max(max(RFscaled(:,:,3)));
RFsc(:,:,:,i)=RFscaled;
end

% Try normalization
RFn = zeros(size(RF));
for i = 1:size(RF,4)
    RFnorm = RF(:,:,:,i);
    norma = norm(RFnorm(:));
    RFnorm = RFnorm/norma;
    RFn(:,:,:,i)=RFnorm;
end
    

vis_square(permute(RFsc,[4,1,2,3]));
