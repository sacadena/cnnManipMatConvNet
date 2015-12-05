function ImFinal = vis_square(data,Norm)

%take an array of shape (n, height, width) or (n, height, width, NumChannels)
% and visualize each (height, width) matrix in a grid of size approx.sqrt(n) by sqrt(n)


%  === Instructions for filters:
%
% if NumChannels = 3 or 1:
% vis_square(permute(net.layers{1}.filters,[4,1,2,3]))
% if Num Channels > 3:
% [h,w,NCh,n] = size(net.layers{lNum}.filters);
% foo = permute(net.layers{lNum}.filters,[4,1,2,3]);
% Imm = vis_square(reshape(permute(foo(1:NCh,:,:,:),[4,1,2,3]),NCh^2,h,w));

%
%  ==== Instructions for outputs or blobs:
%
% foo = permute(cnn_forward(net, x, layerNum),[3,1,2,4]);
% vis_square(foo(1:36,:,:,1))

if nargin<2
    Norm = 0;
end
    
FigHandle = figure('Position', [1, 1, 800, 800]);
colormap 'gray'

data = data-min(data(:));
data = data/max(data(:));
n = ceil(sqrt(size(data,1)));
h = size(data,2);
w = size(data,3);
Nch = size(data,4);
data = padarray(data,[n^2-size(data,1),0,0,0],'post'); 
data = permute(reshape(data,n^2,w*h,Nch),[2,1,3]); % I have each image in columns 


buf=1;
ImFinal = zeros(buf+n*(w+buf),buf+n*(w+buf),Nch);
for nCh = 1:Nch
    A = data(:,:,nCh);
    array= 0*ones(buf+n*(w+buf),buf+n*(w+buf));
    k=1;
    for i=1:n
      for j=1:n
        clim=max(abs(A(:,k)));
        if Norm == 0
            array(buf+(i-1)*(w+buf)+(1:w),buf+(j-1)*(w+buf)+(1:w))=...
            reshape(A(:,k),w,w);
        else
            array(buf+(i-1)*(w+buf)+(1:w),buf+(j-1)*(w+buf)+(1:w))=...
            reshape(A(:,k)-mean(A(:,k)),w,w)/clim;
        end
        k=k+1;
      end
    end
    ImFinal(:,:,nCh) = array;
end

imagesc(ImFinal);