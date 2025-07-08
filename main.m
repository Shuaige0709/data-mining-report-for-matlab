function main()
train_table = readtable("實驗A\train_data.csv");
train_data = train_table{:, 1:end-1};
train_ans = train_table{:, end};

test_table = readtable("實驗A\test_data.csv");
test_data = test_table{:, 1:end-1};
test_ans = test_table{:, end};

% clear the constant row
data = var(train_data);
keep_idx = data > 0;
train_data = train_data(:, keep_idx);
test_data = test_data(:, keep_idx);

% standariztion
mu = mean(train_data);
sigma = std(train_data);
train_data = (train_data - mu) ./ sigma;
test_data = (test_data - mu) ./ sigma;

% oridinary KNN
k = 5;
tic;
pred_knn = KNN(train_data, train_ans, test_data, k);
time_knn = toc;
acc_knn = mean(pred_knn == test_ans);

% vectorized KNN
tic;
pred_vec = KNN_vectorized(train_data, train_ans, test_data, k);
time_vec = toc;
acc_vec = mean(pred_vec == test_ans);

fprintf('KNN : time %.3f s, acc %.2f%%\n', time_knn, acc_knn*100);
fprintf('vectorized KNN : time %.3f s, acc %.2f%%\n', time_vec, acc_vec*100);

% other built-in algorithm
models = {
    fitcsvm(train_data, train_ans);
    fitctree(train_data, train_ans);
    fitcensemble(train_data, train_ans);  % random forest
    fitcnb(train_data, train_ans) % guassian naive bayes
};

names = {'SVM'; 'Decision Tree'; 'Random Forest'; 'Gaussian Naive Bayes'};

n = numel(models);
time_pred = zeros(n, 1);
acc_pred = zeros(n, 1);

for i = 1 : numel(names)
    model = models{i};
    tic;
    pred = predict(model, test_data);
    time_pred(i) = toc;
    acc_pred(i) = mean(pred == test_ans);
    fprintf('%s : time %.3f s, acc %.2f%%\n', names{i}, time_pred(i), acc_pred(i)*100);
end

% Logistic Regression
learnRate = 1e-4;
maxIter = 1000;
tic;
[w, b, mu, sigma] = trainLogistic(train_data, train_ans, learnRate, maxIter);
pred_log = predictLogistic(test_data, w, b, mu, sigma);
time_log = toc;
acc_log = mean(pred_log == test_ans);
fprintf('Logistic Regression : time %.3f s, acc %.2f%%\n', time_log, acc_log*100);

Algorithm = [{'KNN'; 'KNN_vector'; 'Logistic Regression'}; names];
Time = [time_knn; time_vec; time_log;time_pred];
Accuracy = [acc_knn; acc_vec; acc_log;acc_pred];
T = table(Time, Accuracy, 'RowNames', Algorithm);
disp(T);

k_values = 1 : 2 : 49;
num_k = numel(k_values);
acc = zeros(size(k_values));

for idx = 1 : num_k
    k = k_values(idx);
    pred = KNN_vectorized(train_data, train_ans, test_data, k);
    acc(idx) = mean(pred == test_ans);
end

% draw the k value to KNN accuracy
figure;
plot(k_values, acc, '-o');
xlabel('K value');
ylabel('Accuracy');
title('KNN Accuracy v.s. K value');
grid on;
ylim([min(acc)-0.01, max(acc)+0.01]);

% accuracy compare
acc_cmp = Accuracy * 100;
figure;
barh(acc_cmp, 'FaceColor', [0.2 0.6 0.9]);
yticks(1:numel(Algorithm));
yticklabels(Algorithm);
set(gca, 'YDir', 'Reverse');
xlabel('Accuracy (%)');
title('Accuracy in Different Algorithm');
grid on;
xlim([min(acc_cmp)-2, max(acc_cmp)+2]);

end