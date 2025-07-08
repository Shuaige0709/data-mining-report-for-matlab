function prediction = KNN(train_data, train_label, test_data, k)
    M = size(test_data, 1);
    prediction = zeros(M, 1);

    for i = 1 : M
        % calculate ith node distance from other nodes

        sample = test_data(i, :);
        dists = sqrt(sum( (train_data - sample) .^ 2, 2));
        [~, idx] = sort(dists);
        k_idx = idx(1 : k);
        prediction(i) = mode(train_label(k_idx));
    end
end