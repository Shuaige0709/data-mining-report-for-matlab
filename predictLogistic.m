function prediction = predictLogistic(train_data, w, b, mu, sigma)
    train_data = (train_data - mu) ./ sigma;
    z = train_data * w + b;
    p = 1 ./ (1 + exp(-z));
    prediction = double(p >= 0.5);
end