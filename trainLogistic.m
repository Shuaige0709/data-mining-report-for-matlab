function [w, b, mu, sigma] = trainLogistic(train_data, train_label, learn_rate, max_iter)
    [N, D] = size(train_data);
    mu = mean(train_data, 1);
    sigma = std(train_data, 0, 1);
    train_data = (train_data - mu) ./ sigma;
    w = zeros(D, 1);
    b = 0;

    % 梯度下降
    for i = 1 : max_iter
        z = train_data * w + b;
        f = 1 ./ (1 + exp(-z));
        dw = (1/N)*(train_data' * (f - train_label));
        db = mean(f - train_label);
        w = w - learn_rate * dw;
        b = b - learn_rate * db;
    end
end