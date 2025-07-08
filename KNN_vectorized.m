 function prediction = KNN_vectorized(train_data, train_label, test_data, k)
    % 利用公式：||a-b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
    
    train_square = sum(train_data.^2, 2);
    test_square = sum(test_data.^2, 2)';
    dists = bsxfun(@plus, train_square, test_square) - 2*(train_data*test_data');
    
    [~, idx] = sort(dists, 1);
    k_idx = idx(1:k, :);
    prediction = mode(train_label(k_idx), 1)';
end