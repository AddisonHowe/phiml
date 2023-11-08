
% rng(12345)

d=2;
hidden_dims = [16, 32, 32, 16];
softplus = @(x) log(1+exp(x));
identity = @(x) x;
% activations = {softplus, softplus, softplus, softplus, identity};
activations = {@tanh, @tanh, @tanh, @tanh, identity};
sigma_init = 1;

nlayers = 1 + length(hidden_dims);
Ws = cell(1, nlayers);
Bs = cell(1, nlayers);

Ws{1} = makematrix(d, hidden_dims(1), sigma_init);
Bs{1} = sigma_init * randn([hidden_dims(1), 1]);
for i=2:nlayers-1
    Ws{i} = makematrix(hidden_dims(i-1), hidden_dims(i), sigma_init);
    Bs{i} = sigma_init * randn([hidden_dims(i), 1]);
end
Ws{end} = makematrix(hidden_dims(end), 1, sigma_init);
Bs{end} = sigma_init * randn([1, 1]);

x=[1; 1];

y=NN(x, Ws, Bs, activations);

r = 5;
res = 1e-2;
xs = (-r:res:r);
ys = (-r:res:r);
[X,Y] = meshgrid(xs,ys);

XY = nan([d,length(xs),length(ys)]);
XY(1,:,:) = X;
XY(2,:,:) = Y;

XY = reshape(XY, [d, length(xs)*length(ys)]);

phis = NN(XY, Ws, Bs, activations);
phis = reshape(phis, [length(xs),length(ys)]);

disp(min(phis, [], 'all'))
disp(max(phis, [], 'all'))

surf(X, Y, phis)
contour(X,Y,phis)

function M=makematrix(din, dout, sigma)
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
