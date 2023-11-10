
% rng(12345)

close all;

d=2;
hidden_dims = [16, 32, 32, 32, 16];
softplus = @(x) log(1+exp(x));
identity = @(x) x;
% activations = {softplus, softplus, softplus, softplus, softplus, softplus};
activations = {@tanh, @tanh, @tanh, @tanh, @tanh, @tanh};
% activations = {@tanh, @tanh, @tanh, @tanh, softplus};
sigma_init = 'x';

nlayers = 1 + length(hidden_dims);
Ws = cell(1, nlayers);
Bs = cell(1, nlayers);

Ws{1} = makematrix(d, hidden_dims(1), sigma_init);
Bs{1} = zeros([hidden_dims(1), 1]);  %sigma_init * randn([hidden_dims(1), 1]);
for i=2:nlayers-1
    Ws{i} = makematrix(hidden_dims(i-1), hidden_dims(i), sigma_init);
    Bs{i} = zeros([hidden_dims(i), 1]);  %sigma_init * randn([hidden_dims(i), 1]);
end
Ws{end} = makematrix(hidden_dims(end), 1, sigma_init);
Bs{end} = zeros([1,1]);  % sigma_init * randn([1, 1]);


r = 100;
res = 1;
xs = (-r:res:r);
ys = (-r:res:r);
[X,Y] = meshgrid(xs,ys);
XY = nan([d,length(xs),length(ys)]);
XY(1,:,:) = X;
XY(2,:,:) = Y;
XY = reshape(XY, [d, length(xs)*length(ys)]);

phis = NN(XY, Ws, Bs, activations);
minphi = min(phis);
maxphi = max(phis);
phis = reshape(phis, [length(xs),length(ys)]);

fprintf("Phi range: [%.3g, %.3g]\n", minphi, maxphi);

figure;
contour(X,Y,phis)
xlabel('$x$', Interpreter='latex')
ylabel('$y$', Interpreter='latex')
colorbar;

figure;
imagesc(xs, ys, phis)
set(gca,'YDir','normal') 
xlabel('$x$', Interpreter='latex')
ylabel('$y$', Interpreter='latex')
colorbar;

Rs = (1:r);
minPhis = nan(size(Rs));
maxPhis = nan(size(Rs));
for i=1:length(Rs)
    r = Rs(i);
    xscreen=-r < xs & xs < r;
    yscreen=-r < ys & ys < r;
    phitest=phis(xscreen, yscreen);
    minPhis(i) = min(phitest, [], 'all');
    maxPhis(i) = max(phitest, [], 'all');
end


figure;
surf(X, Y, phis)
xlabel('$x$', Interpreter='latex')
ylabel('$y$', Interpreter='latex')
zlabel('$\phi$', Interpreter='latex')
colorbar;

% Scaling
figure;
subplot(2,1,1)
plot(Rs, maxPhis, 'o-')
xlabel("$r$", Interpreter="latex")
ylabel("$\max{\phi}$", Interpreter="latex")
title("Max")

subplot(2,1,2)
plot(Rs, minPhis, 'o-')
xlabel("$r$", Interpreter="latex")
ylabel("$\min{\phi}$", Interpreter="latex")
title("Min")



function M=makematrix(din, dout, sigma)
    if sigma == 'x'
        sigma = 1/sqrt(din*dout);
    end
    M = sigma * randn([dout, din]);
end

function y=NN(x, Ws, bs, activations)
    for i=1:length(Ws)
        W = Ws{i};
        b = bs{i};
        act = activations{i};
        x = act(W*x+b);
    end
    y=x;
end
